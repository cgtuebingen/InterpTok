"""Precompute what training reads: the DINOv3 CLS feature and the 64 nearest neighbours of every training
image, and the feature moments of the training images for the representation Fréchet loss. Runs on the GPUs
of one node:

    torchrun --nproc_per_node 8 src/prepare.py --data imagenet/train --out prepared
"""
import argparse
import os
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from transformers import AutoModel

import data
import losses


@torch.no_grad()
def main():
    p = argparse.ArgumentParser(description="Precompute teacher features, neighbours and Fréchet reference moments.")
    p.add_argument("--data", required=True, help="ImageNet train split, as <class>/<image> folders")
    p.add_argument("--out", required=True)
    p.add_argument("--neighbours", type=int, default=64)
    p.add_argument("--batch", type=int, default=128)
    args = p.parse_args()
    device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", "0")))
    torch.cuda.set_device(device)
    distributed = "RANK" in os.environ
    if distributed:
        dist.init_process_group("nccl")
    rank, world = (dist.get_rank(), dist.get_world_size()) if distributed else (0, 1)
    barrier = dist.barrier if distributed else (lambda: None)

    images = data.Images(args.data)
    files = images.files
    if not 0 < args.neighbours < len(files) or len(files) < world:
        raise SystemExit(f"{len(files)} images: need more than --neighbours {args.neighbours} and at least {world}")
    out = Path(args.out)
    (out / "frechet").mkdir(parents=True, exist_ok=True)
    if rank == 0:
        (out / "files.txt").unlink(missing_ok=True)  # written last, it marks a complete preparation
        np.lib.format.open_memmap(out / "teacher.npy", mode="w+", dtype=np.float16, shape=(len(files), 768)).flush()
    barrier()
    features = np.load(out / "teacher.npy", mmap_mode="r+")

    teacher = AutoModel.from_pretrained(losses.TEACHER).to(device).eval()
    encoders = {name: make().to(device).eval() for name, make in losses.ENCODERS.items()}
    sums = {}
    mean = torch.tensor(losses.IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(losses.IMAGENET_STD, device=device).view(1, 3, 1, 1)
    shard = Subset(images, range(rank, len(files), world))
    loader = DataLoader(shard, batch_size=args.batch, num_workers=8, pin_memory=True, multiprocessing_context="spawn")
    for i, (x, idx) in enumerate(loader):
        x = (x.to(device, non_blocking=True) + 1) / 2
        with torch.autocast("cuda", torch.bfloat16):
            cls = teacher(pixel_values=(x - mean) / std).last_hidden_state[:, 0]
            feats = {name: encoder(x) for name, encoder in encoders.items()}
        features[idx.numpy()] = F.normalize(cls.float(), dim=-1).half().cpu().numpy()
        for name, f in feats.items():
            f = f.double()
            s1, s2 = sums.setdefault(name, (torch.zeros(f.shape[1], dtype=torch.float64, device=device),
                                            torch.zeros(f.shape[1], f.shape[1], dtype=torch.float64, device=device)))
            s1 += f.sum(0)
            s2 += f.T @ f
        if rank == 0 and i % 100 == 0:
            print(f"{(i + 1) * args.batch * world} / {len(files)} images", flush=True)
    features.flush()

    for name, (s1, s2) in sums.items():
        if distributed:
            dist.all_reduce(s1)
            dist.all_reduce(s2)
        mu = s1 / len(files)
        sigma = s2 / len(files) - mu[:, None] * mu[None, :]
        if rank == 0:
            np.savez(out / "frechet" / f"{name}.npz", mu=mu.float().cpu().numpy(),
                     sigma=(0.5 * (sigma + sigma.T)).float().cpu().numpy())
    barrier()

    if rank == 0:  # nearest neighbours by cosine similarity of the CLS features, without the image itself
        emb = torch.from_numpy(np.load(out / "teacher.npy")).to(device)
        neighbours = np.empty((len(files), args.neighbours), dtype=np.int32)
        for s in range(0, len(files), 1024):
            sims = (emb[s:s + 1024] @ emb.T).float()
            sims[torch.arange(len(sims)), torch.arange(s, s + len(sims))] = -float("inf")
            neighbours[s:s + 1024] = sims.topk(args.neighbours, dim=1).indices.int().cpu().numpy()
        np.save(out / "neighbours.npy", neighbours)
        (out / "files.txt").write_text("\n".join(files), encoding="utf-8")
    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
