"""Fine-tune MAETok between images into InterpTok, on one GPU or several:

    torchrun --nproc_per_node 8 src/train.py configs/interptok.yaml [--resume runs/interptok/last.pt]

Every `save_every` steps the run writes the moving average of the tokenizer, interptok_<step>.safetensors, and
the checkpoint last.pt, from which --resume continues.
"""
import argparse
import os
from dataclasses import dataclass
from pathlib import Path

import lpips
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
import yaml
from safetensors.torch import save_file
from torch.utils.checkpoint import checkpoint
from torch.utils.data import DataLoader
from transformers import AutoModel

import data
import losses
import model as tokenizer
from discriminator import Discriminator, diff_augment

IMAGES = 8                    # images per micro-batch; every pair of them is mixed, 28 mixtures
DISC_WEIGHT = 0.4             # adversarial weight, times the adaptive weight
DISC_WARMUP = 1500            # steps before the adversarial loss reaches the tokenizer
CR_WEIGHT, LECAM_WEIGHT = 4.0, 1e-3
EMA_DECAY = 0.998
PATCH_BLOCK = 6               # decoder block read by the decoder-patch alignment
ROWS = 8                      # mixtures in the feature mixing ratio loss, images in the Fréchet loss, per micro-batch
AMP = {"device_type": "cuda", "dtype": torch.bfloat16}


@dataclass
class Config:
    """The settings of a run, read from a file like configs/interptok.yaml, which explains each of them."""
    data: str
    prepared: str
    out: str
    steps: int
    batch: int
    seed: int
    save_every: int
    decoder_only: bool
    grow: int
    interp: bool
    conditional: bool
    code_noise: float
    pixel_ratio: float
    feature_ratio: float
    axis: float
    pred: float
    patch: float
    frechet: float
    frechet_encoders: list

    @classmethod
    def load(cls, path):
        """Read a config file; every field above must be set in it."""
        return cls(**yaml.safe_load(Path(path).read_text(encoding="utf-8")))


def beta22(u):
    """Beta(2, 2) by its exact inverse CDF, the root of 3a^2 - 2a^3 = u in [0, 1]."""
    return np.clip(0.5 + np.cos((np.arccos(1.0 - 2.0 * u) + 4.0 * np.pi) / 3.0), 0.0, 1.0)


def reduce_sum(t):
    """Sum a tensor over the GPUs, in place; a no-op on one GPU."""
    if dist.is_initialized():
        dist.all_reduce(t)
    return t


def reduce_mean(grads, world):
    """Average gradients over the GPUs through one flat buffer."""
    if world == 1 or not grads:
        return
    flat = reduce_sum(torch.cat([g.flatten() for g in grads])) / world
    for g, part in zip(grads, flat.split([g.numel() for g in grads]), strict=True):
        g.copy_(part.view_as(g))


class Trainer:
    """The tokenizer, both discriminators and all losses; `micro_batch` accumulates gradients, `optimizer_step`
    applies them."""

    def __init__(self, cfg, device, world=1):
        self.cfg, self.device, self.world = cfg, device, world
        self.model = tokenizer.load(grow=cfg.grow).to(device).train()
        if cfg.decoder_only:
            self.model.encoder.requires_grad_(False)
            self.model.quant_conv.requires_grad_(False)
        self.pred_head = losses.mlp(128 * 32, 768).to(device) if cfg.pred > 0 else None
        self.patch_head = losses.mlp(768, 768).to(device) if cfg.patch > 0 else None
        self.heads = torch.nn.ModuleList(h for h in (self.pred_head, self.patch_head) if h is not None)
        self.g_params = [p for p in self.model.parameters() if p.requires_grad] + list(self.heads.parameters())
        self.ema = {k: v.detach().clone().float() for k, v in self.model.state_dict().items() if v.is_floating_point()}
        self.disc = Discriminator(conditional=cfg.conditional).to(device).train()
        self.d_params = [p for p in self.disc.parameters() if p.requires_grad]
        self.opt_g = torch.optim.AdamW(self.g_params, lr=1e-5, betas=(0.9, 0.99), weight_decay=0.01)
        self.opt_d = torch.optim.AdamW(self.d_params, lr=1e-4, betas=(0.9, 0.95), weight_decay=0.0)
        self.lecam, self.lecam_sums = losses.LeCam(), np.zeros(3)

        if (cfg.interp and cfg.feature_ratio > 0) or cfg.patch > 0:  # the losses that run the vision model
            self.teacher = AutoModel.from_pretrained(losses.TEACHER).to(device).eval().requires_grad_(False)
            self.prefix = 1 + self.teacher.config.num_register_tokens
            self.mean = torch.tensor(losses.IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
            self.std = torch.tensor(losses.IMAGENET_STD, device=device).view(1, 3, 1, 1)
        self.lpips = lpips.LPIPS(net="vgg", verbose=False).to(device).eval().requires_grad_(False)
        self.frechet = (losses.FrechetLoss(f"{cfg.prepared}/frechet", tuple(cfg.frechet_encoders)).to(device)
                        if cfg.frechet > 0 else None)
        self.hidden = {}
        if self.patch_head is not None:
            self.model.decoder.model.blocks[PATCH_BLOCK].register_forward_hook(lambda m, i, o: self.hidden.update(h=o))

    def teacher_tokens(self, x):
        """DINOv3 tokens of images x in [-1, 1]: CLS, registers, then 16 x 16 patches, each 768-dimensional."""
        x = ((x.float().clamp(-1, 1) + 1) / 2 - self.mean) / self.std
        with torch.autocast(**AMP):
            return self.teacher(pixel_values=x).last_hidden_state

    def lpips_loss(self, y, x, chunk=4):
        """LPIPS (VGG) between y and x, in chunks under activation checkpointing to save memory."""
        return torch.cat([checkpoint(self.lpips, y[i:i + chunk], x[i:i + chunk], use_reentrant=False).flatten()
                          for i in range(0, len(y), chunk)]).mean()

    def micro_batch(self, x, t_cls, alpha, rng, step, accum):
        """One micro-batch of images x in [-1, 1], their teacher CLS features and the mixing ratios alpha of all
        pairs. Accumulates the gradients of both discriminators and of the tokenizer, and returns the losses."""
        cfg, model, n = self.cfg, self.model, x.shape[0]
        ia, ib = (torch.from_numpy(i).to(self.device) for i in np.triu_indices(n, k=1))

        # Encode the images, mix the codes of every pair, and decode reconstructions and mixtures in one call.
        z = tokenizer.encode(model, x)
        z_rec = z
        if cfg.code_noise > 0:
            sigma = torch.as_tensor(np.abs(rng.normal(0.0, cfg.code_noise, size=n)), dtype=z.dtype, device=z.device)
            eps = torch.as_tensor(rng.standard_normal(size=tuple(z.shape)), dtype=z.dtype, device=z.device)
            z_rec = z + sigma.view(-1, 1, 1) * model.vq_std * eps
        mix = alpha.view(-1, 1, 1)
        decoded = tokenizer.decode(model, torch.cat([z_rec, (1 - mix) * z[ia] + mix * z[ib]]))
        y_rec, y = decoded[:n], decoded[n:]
        rec = (y_rec - x).abs().mean() + self.lpips_loss(y_rec, x)

        # Without source conditioning, the interpolation discriminator's real images are random sources.
        x_real = None if cfg.conditional else x[torch.from_numpy(rng.integers(0, n, len(ia))).to(self.device)]
        sources, view, log = self.train_discriminators(x, x[ia], x[ib], x_real, y, y_rec, accum)

        self.set_disc_trainable(False)
        adversarial, weight = self.adversarial_loss(y, y_rec, x_real, sources, view, rec, step)
        loss = rec + weight * adversarial
        log.update(rec=float(rec), g=float(adversarial), w=weight)

        # Mixing ratio losses on the mixtures.
        if cfg.interp and cfg.pixel_ratio > 0:
            ratio = losses.pixel_ratio_loss(y, y_rec[ia].detach(), y_rec[ib].detach(), alpha)
            loss = loss + cfg.pixel_ratio * ratio
            log["ratio"] = float(ratio)
        if cfg.interp and cfg.feature_ratio > 0:
            pick = torch.from_numpy(rng.choice(len(ia), min(ROWS, len(ia)), replace=False)).to(self.device)
            f = self.teacher_tokens(y[pick])[:, 0].float()
            feature_ratio = losses.feature_ratio_loss(f, t_cls[ia[pick]], t_cls[ib[pick]], alpha[pick])
            loss = loss + cfg.feature_ratio * feature_ratio
            log["feature_ratio"] = float(feature_ratio)

        # Alignment losses with the frozen vision model on single images.
        codes = z.float().flatten(1)
        if cfg.axis > 0:
            loss = loss + cfg.axis * losses.axis_loss(codes, t_cls)
        if self.pred_head is not None:
            loss = loss + cfg.pred * losses.cosine_loss(self.pred_head(codes), t_cls)
        if self.patch_head is not None:
            with torch.no_grad():
                target = self.teacher_tokens(x)[:, self.prefix:].float()
            hidden = self.hidden.pop("h")[:n, :model.decoder.num_img_tokens].float()
            loss = loss + cfg.patch * losses.cosine_loss(self.patch_head(hidden), target)

        # Representation Fréchet loss on a subsample of the decoded images, mixtures included.
        if self.frechet is not None:
            rows = decoded if cfg.interp else y_rec
            pick = torch.from_numpy(rng.choice(len(rows), min(ROWS, len(rows)), replace=False)).to(self.device)
            loss = loss + cfg.frechet * self.frechet((rows[pick] + 1) / 2)

        (loss / accum).backward()
        self.set_disc_trainable(True)
        return log

    def train_discriminators(self, x, x_a, x_b, x_real, y, y_rec, accum):
        """Accumulates the gradients of both discriminators: the interpolation discriminator judges each decoded
        mixture y beside its sources x_a and x_b, the reconstruction discriminator the images x beside y_rec."""
        disc, y = self.disc, y.detach()
        with torch.autocast(**AMP):
            if disc.conditional:
                fake_in = diff_augment(y)
                view = disc.backbone.draw_view()
                sources = (disc.sources(diff_augment(x_a), view), disc.sources(diff_augment(x_b), view))
                real, fake = disc.interp_logits(fake_in, sources, view)
            else:
                sources = view = None
                real_in, fake_in = diff_augment(x_real), diff_augment(y)
                real, fake = disc.logits(real_in, recon=False), disc.logits(fake_in, recon=False)
        view_rec = disc.backbone.draw_view() if disc.conditional else "draw"  # shared crop, or one per call
        rec_real = disc.logits(diff_augment(x), view_rec)
        rec_fake = disc.logits(diff_augment(y_rec.detach()), view_rec)
        loss = (losses.interp_d(real, fake) if disc.conditional else losses.hinge_d(real, fake)) \
            + losses.hinge_d(rec_real, rec_fake)
        log = {"d": float(loss)}
        self.lecam_sums += [float(real.mean()), float(fake.mean()), 1.0]
        loss = DISC_WEIGHT * (loss + LECAM_WEIGHT * self.lecam(real, fake))
        with torch.autocast(**AMP):  # consistency regularization against a stronger augmentation
            policy = "color,translation,cutout_0.5"
            if disc.conditional:
                cr = disc.interp_logits(diff_augment(y, policy, 1.0), sources, view)
            else:
                cr = (disc.logits(diff_augment(x_real, policy, 1.0), recon=False),
                      disc.logits(diff_augment(y, policy, 1.0), recon=False))
        loss = loss + CR_WEIGHT * F.mse_loss(torch.cat([real, fake]), torch.cat(cr))
        (loss / accum).backward()
        return sources, view, log

    def adversarial_loss(self, y, y_rec, x_real, sources, view, rec, step):
        """The tokenizer's adversarial loss and its weight, the adaptive weight once the discriminators have
        warmed up. Without interpolation the mixtures still pass the discriminator, at weight 0, so that every
        variant runs the same passes."""
        disc = self.disc
        with torch.autocast(**AMP):
            if disc.conditional:
                interp = losses.interp_g(*disc.interp_logits(diff_augment(y), sources, view))
            else:
                interp = -disc.logits(diff_augment(y), recon=False).mean()
        view_rec = disc.backbone.draw_view() if disc.conditional else "draw"
        loss = (1.0 if self.cfg.interp else 0.0) * interp - disc.logits(diff_augment(y_rec), view_rec).mean()
        if step < DISC_WARMUP:
            return loss, 0.0
        return loss, losses.adaptive_weight(rec, loss, self.model.decoder.last_layer) * DISC_WEIGHT

    def set_disc_trainable(self, flag):
        """Freeze the discriminators while the tokenizer's losses pass through them, and unfreeze them after."""
        for p in self.d_params:
            p.requires_grad_(flag)

    def optimizer_step(self):
        """Average the accumulated gradients over the GPUs, clip and apply them, then update LeCAM, the moments
        of the Fréchet loss and the moving average of the tokenizer."""
        for params in (self.g_params, self.d_params):
            reduce_mean([p.grad for p in params if p.grad is not None], self.world)
        torch.nn.utils.clip_grad_norm_(self.g_params, 1.0)
        self.opt_g.step()
        torch.nn.utils.clip_grad_norm_(self.d_params, 1.0)
        self.opt_d.step()
        self.opt_g.zero_grad(set_to_none=True)
        self.opt_d.zero_grad(set_to_none=True)
        real, fake, count = reduce_sum(torch.tensor(self.lecam_sums, device=self.device)).tolist()
        self.lecam.update(real / count, fake / count)
        self.lecam_sums = np.zeros(3)
        if self.frechet is not None:
            self.frechet.step()
        with torch.no_grad():
            for k, v in self.model.state_dict().items():
                if k in self.ema:
                    self.ema[k].mul_(EMA_DECAY).add_(v.float(), alpha=1 - EMA_DECAY)

    def ema_state(self):
        """The moving average of the tokenizer, the model we evaluate and release."""
        return {k: (self.ema[k].to(v.dtype) if k in self.ema else v).detach().contiguous()
                for k, v in self.model.state_dict().items()}

    def state_dict(self):
        """Everything a resumed run needs, without the frozen networks."""
        frechet = self.frechet.state_dict() if self.frechet else {}
        return {"model": self.model.state_dict(), "ema": self.ema, "heads": self.heads.state_dict(),
                "disc": {k: v for k, v in self.disc.state_dict().items() if not k.startswith("backbone.")},
                "opt_g": self.opt_g.state_dict(), "opt_d": self.opt_d.state_dict(),
                "frechet": {k: v for k, v in frechet.items() if not k.startswith("encoders.")},
                "lecam": (self.lecam.real, self.lecam.fake)}

    def load_state_dict(self, state):
        """Restore what `state_dict` saved; the frozen networks come from their pretrained weights."""
        self.model.load_state_dict(state["model"])
        self.ema = {k: v.to(self.device) for k, v in state["ema"].items()}
        self.heads.load_state_dict(state["heads"])
        self.disc.load_state_dict(state["disc"], strict=False)
        self.opt_g.load_state_dict(state["opt_g"])
        self.opt_d.load_state_dict(state["opt_d"])
        if self.frechet is not None:
            self.frechet.load_state_dict(state["frechet"], strict=False)
        self.lecam.real, self.lecam.fake = state["lecam"]


def main():
    p = argparse.ArgumentParser(description="Fine-tune MAETok between images into InterpTok.")
    p.add_argument("config", help="a file like configs/interptok.yaml")
    p.add_argument("--resume", help="checkpoint to continue from, such as <out>/last.pt")
    args = p.parse_args()
    cfg = Config.load(args.config)

    device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", "0")))
    torch.cuda.set_device(device)
    distributed = "RANK" in os.environ
    if distributed:
        dist.init_process_group("nccl")
    rank, world = (dist.get_rank(), dist.get_world_size()) if distributed else (0, 1)
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = True
    if cfg.batch % (IMAGES * world):
        raise SystemExit(f"batch {cfg.batch} is not a multiple of {IMAGES} images x {world} GPUs")
    accum = cfg.batch // (IMAGES * world)
    out = Path(cfg.out)
    out.mkdir(parents=True, exist_ok=True)

    images = data.Images(cfg.data)
    if (Path(cfg.prepared) / "files.txt").read_text(encoding="utf-8").split("\n") != images.files:
        raise SystemExit(f"{cfg.prepared} was prepared for other images than {cfg.data}; run src/prepare.py on them")
    teacher_cls = np.load(f"{cfg.prepared}/teacher.npy", mmap_mode="r")
    neighbours = np.load(f"{cfg.prepared}/neighbours.npy")

    torch.manual_seed(cfg.seed)
    trainer = Trainer(cfg, device, world)
    start = 0
    if args.resume:
        state = torch.load(args.resume, map_location=device, weights_only=False)
        if state["batch"] != cfg.batch:
            raise SystemExit(f"{args.resume} was trained at batch {state['batch']}, the config sets {cfg.batch}")
        trainer.load_state_dict(state)
        start = state["step"]
    if distributed:
        for module in (trainer.model, trainer.heads, trainer.disc):
            for t in list(module.parameters()) + list(module.buffers()):
                dist.broadcast(t.data, 0)
    torch.manual_seed(cfg.seed + rank + 7919 * start)

    sampler = data.PairBatches(neighbours, IMAGES // 2, rank, world, skip=start * accum)
    batches = iter(DataLoader(images, batch_sampler=sampler, num_workers=8, pin_memory=True, prefetch_factor=4,
                              multiprocessing_context="spawn"))
    for step in range(start + 1, cfg.steps + 1):
        logs = []
        for micro in range(accum):
            x, idx = next(batches)
            t_cls = torch.from_numpy(teacher_cls[idx.numpy()].astype(np.float32)).to(device)
            rng = np.random.default_rng([cfg.seed, step, rank * accum + micro])
            alpha = torch.as_tensor(beta22(rng.random(IMAGES * (IMAGES - 1) // 2)), dtype=torch.float32, device=device)
            logs.append(trainer.micro_batch(x.to(device, non_blocking=True), t_cls, alpha, rng, step, accum))
        trainer.optimizer_step()

        if rank == 0 and step % 25 == 0:
            print(f"step {step}", *(f"{k} {np.mean([log[k] for log in logs]):.4f}" for k in logs[0]), flush=True)
        if rank == 0 and (step % cfg.save_every == 0 or step == cfg.steps):
            save_file(trainer.ema_state(), out / f"interptok_{step:07d}.safetensors")
            torch.save({"step": step, "batch": cfg.batch, **trainer.state_dict()}, out / "last.pt.tmp")
            os.replace(out / "last.pt.tmp", out / "last.pt")
    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
