"""Evaluate a tokenizer of this repository:

    python src/evaluate.py recon --weights W --data imagenet/val                 PSNR, SSIM and LPIPS
    torchrun --nproc_per_node 4 src/evaluate.py probe --weights W --train imagenet/train --val imagenet/val
    torchrun --nproc_per_node 4 src/evaluate.py swap --weights W --steps 250 --out samples.npz

`swap` decodes samples of the SiT-XL released with MAETok by the given decoder, for the ADM evaluator.
iFID, rFID and the SiT-B generators use the code of Xu et al. (2026); see the README.
Without --weights, every command evaluates MAETok itself.
"""
import argparse
import math
import os
from pathlib import Path

import lpips
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import datasets, transforms

import data
import model as tokenizer


def ssim(a, b, k=7):
    """SSIM as MAETok computes it, skimage's default uniform 7x7 window at data range 2, on images in [0, 1]."""
    w = torch.ones(a.shape[1], 1, k, k, device=a.device) / k ** 2

    def mean(t):
        return F.conv2d(t, w, groups=a.shape[1])

    ma, mb, n = mean(a), mean(b), k * k / (k * k - 1)
    va, vb, cov = n * (mean(a * a) - ma ** 2), n * (mean(b * b) - mb ** 2), n * (mean(a * b) - ma * mb)
    c1, c2 = 0.02 ** 2, 0.06 ** 2
    return (((2 * ma * mb + c1) * (2 * cov + c2)) / ((ma ** 2 + mb ** 2 + c1) * (va + vb + c2))).flatten(1).mean(1)


@torch.no_grad()
def recon(args, tok, device):
    """PSNR, SSIM and LPIPS (AlexNet) between every image of args.data and its reconstruction, averaged."""
    alex = lpips.LPIPS(net="alex", verbose=False).to(device).eval()
    scores = {"psnr": [], "ssim": [], "lpips": []}
    for x, _ in DataLoader(data.Images(args.data), batch_size=50, num_workers=8):
        x = x.to(device)
        with torch.autocast("cuda", torch.bfloat16):
            y = tokenizer.decode(tok, tokenizer.encode(tok, x))
        y = y.float().clamp(-1, 1)
        a, b = (x + 1) / 2, (y + 1) / 2
        scores["psnr"] += (10 * torch.log10(1 / ((b - a) ** 2).flatten(1).mean(1).clamp_min(1e-12))).tolist()
        scores["ssim"] += ssim(b, a).tolist()
        scores["lpips"] += alex(y, x).flatten().tolist()
    print(f"{len(scores['psnr'])} images: " + " ".join(f"{k} {np.mean(v):.4f}" for k, v in scores.items()))


class LARS(torch.optim.Optimizer):
    """LARS with momentum, as in MAE's linear probe; biases and normalization parameters skip the trust ratio."""

    def __init__(self, params, lr=0.0, momentum=0.9, trust=0.001):
        super().__init__(params, {"lr": lr, "momentum": momentum, "trust": trust})

    @torch.no_grad()
    def step(self):
        for g in self.param_groups:
            for p in g["params"]:
                if p.grad is None:
                    continue
                d = p.grad
                if p.ndim > 1:
                    pn, gn = p.norm(), d.norm()
                    d = d * torch.where((pn > 0) & (gn > 0), g["trust"] * pn / gn, torch.ones_like(pn))
                mu = self.state[p].setdefault("mu", torch.zeros_like(p))
                mu.mul_(g["momentum"]).add_(d)
                p.add_(mu, alpha=-g["lr"])


def center_crop(image):
    """ADM's center crop of a PIL image, defined at module level so that spawned data workers can load it."""
    return data.center_crop(image.convert("RGB"))


def probe(args, tok, device, rank, world):
    """Linear probe of the flattened codes with MAE's recipe: 90 epochs, LARS, batch 16384, base lr 0.1."""
    normalize = transforms.Normalize([0.5] * 3, [0.5] * 3)
    crop = transforms.RandomResizedCrop(256, scale=(0.08, 1.0), interpolation=transforms.InterpolationMode.BICUBIC)
    train = datasets.ImageFolder(args.train, transforms.Compose(
        [crop, transforms.RandomHorizontalFlip(), transforms.ToTensor(), normalize]))
    val = datasets.ImageFolder(args.val, transforms.Compose(
        [transforms.Lambda(center_crop), transforms.ToTensor(), normalize]))
    sampler = DistributedSampler(train, world, rank, shuffle=True, drop_last=True)
    workers = {"num_workers": 10, "persistent_workers": True, "multiprocessing_context": "spawn"}
    loader = DataLoader(train, batch_size=128, sampler=sampler, pin_memory=True, drop_last=True, **workers)
    val_loader = DataLoader(val, batch_size=128, sampler=DistributedSampler(val, world, rank, shuffle=False), **workers)
    accum = max(1, 16384 // (128 * world))
    lr, epochs, warmup = 0.1 * 128 * world * accum / 256, 90, 10
    torch.manual_seed(0)  # seeds the head and the augmentations of the data workers, as MAE's recipe does
    head = nn.Sequential(nn.BatchNorm1d(128 * 32, affine=False, eps=1e-6), nn.Linear(128 * 32, 1000)).to(device)
    nn.init.trunc_normal_(head[1].weight, std=0.01)
    nn.init.zeros_(head[1].bias)
    head = nn.parallel.DistributedDataParallel(head, device_ids=[device.index]) if world > 1 else head
    opt = LARS(head.parameters(), lr=lr)

    def codes(x):
        with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
            return tokenizer.encode(tok, x.to(device, non_blocking=True)).float().flatten(1)

    for epoch in range(epochs):
        head.train()
        sampler.set_epoch(epoch)
        for i, (x, y) in enumerate(loader):
            if i % accum == 0:
                t = epoch + i / len(loader)
                opt.param_groups[0]["lr"] = lr * t / warmup if t < warmup else \
                    lr * 0.5 * (1 + math.cos(math.pi * (t - warmup) / (epochs - warmup)))
                opt.zero_grad(set_to_none=True)
            (F.cross_entropy(head(codes(x)), y.to(device)) / accum).backward()
            if i % accum == accum - 1:
                opt.step()
        head.eval()
        correct, seen = torch.zeros((), device=device), torch.zeros((), device=device)
        with torch.no_grad():
            for x, y in val_loader:
                correct += (head(codes(x)).argmax(1) == y.to(device)).sum()
                seen += len(y)
        if world > 1:
            dist.all_reduce(correct)
            dist.all_reduce(seen)
        if rank == 0:
            print(f"epoch {epoch + 1}: top-1 {100 * float(correct / seen):.2f}", flush=True)


@torch.no_grad()
def swap(args, tok, device, rank, world):
    """MAETok's sampling procedure (inference/generate_sit.py) for its SiT-XL in float32 with TF32 matmuls,
    64 samples per GPU, decoded by the given decoder. The paper sampled on 4 GPUs with seed 0. Writes the
    samples as uint8 of shape (N, 256, 256, 3) to an .npz file, the input of the ADM evaluator."""
    from modelling.samplers import euler_maruyama_sampler
    from modelling.sit import SiT
    torch.backends.cuda.matmul.allow_tf32 = True
    sit = SiT.from_pretrained("MAETok/sit-xl_maetok-b-128").to(device).eval()
    torch.manual_seed(args.seed * world + rank)
    n, rounds = 64, math.ceil(args.samples / (64 * world))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    shard = np.lib.format.open_memmap(f"{out}.rank{rank}.npy", mode="w+", dtype=np.uint8,
                                      shape=(rounds * n, 256, 256, 3))
    for r in range(rounds):
        z = torch.randn(n, 128, 32, device=device)
        y = torch.randint(0, 1000, (n,), device=device)
        z = euler_maruyama_sampler(model=sit, latents=z, y=y, num_steps=args.steps, cfg_scale=1.0,
                                   guidance_low=0.0, guidance_high=1.0, path_type="linear").float()
        x = (tokenizer.decode(tok, z * tok.vq_std + tok.vq_mean).clamp(-1, 1) + 1) / 2
        shard[r * n:(r + 1) * n] = (255 * x).clamp(0, 255).permute(0, 2, 3, 1).to("cpu", torch.uint8).numpy()
    shard.flush()
    del shard
    if world > 1:
        dist.barrier()
    if rank == 0:  # sample i of rank q is sample i * world + q of the run, as in MAETok's script
        samples = np.lib.format.open_memmap(f"{out}.all.npy", mode="w+", dtype=np.uint8,
                                            shape=(args.samples, 256, 256, 3))
        for q in range(world):
            part = np.load(f"{out}.rank{q}.npy", mmap_mode="r")
            samples[q::world] = part[:len(range(q, args.samples, world))]
            del part
        np.savez(out, arr_0=samples)
        del samples
        for name in [f"{out}.rank{q}.npy" for q in range(world)] + [f"{out}.all.npy"]:
            os.remove(name)
        print(f"{args.samples} samples at {args.steps} steps -> {out}")


def main():
    p = argparse.ArgumentParser(description="Evaluate a tokenizer.")
    commands = p.add_subparsers(dest="command", required=True)
    recon_ = commands.add_parser("recon", help="PSNR, SSIM and LPIPS of reconstructions")
    recon_.add_argument("--data", required=True, help="ImageNet validation split, as <class>/<image> folders")
    probe_ = commands.add_parser("probe", help="linear probe of the codes")
    probe_.add_argument("--train", required=True, help="ImageNet train split")
    probe_.add_argument("--val", required=True, help="ImageNet validation split")
    swap_ = commands.add_parser("swap", help="samples of MAETok's SiT-XL, decoded by the given decoder")
    swap_.add_argument("--steps", type=int, default=250, help="sampling steps")
    swap_.add_argument("--samples", type=int, default=50000)
    swap_.add_argument("--seed", type=int, default=0)
    swap_.add_argument("--out", default="samples.npz", help="output for the ADM evaluator")
    for command in (recon_, probe_, swap_):
        command.add_argument("--weights", help="tokenizer weights (.safetensors), MAETok without them")
    args = p.parse_args()
    device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", "0")))
    torch.cuda.set_device(device)
    distributed = "RANK" in os.environ
    if distributed:
        dist.init_process_group("nccl")
    rank, world = (dist.get_rank(), dist.get_world_size()) if distributed else (0, 1)
    tok = tokenizer.load(args.weights).to(device).eval().requires_grad_(False)
    if args.command == "recon":
        recon(args, tok, device)
    elif args.command == "probe":
        torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = True
        probe(args, tok, device, rank, world)
    else:
        swap(args, tok, device, rank, world)
    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
