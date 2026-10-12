"""The interpolation and reconstruction discriminators, on the frozen DINO ViT-S/16 of MAETok's discriminator."""
import math

import timm
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

DINO_URL = "https://dl.fbaipublicfiles.com/dino/dino_deitsmall16_pretrain/dino_deitsmall16_pretrain.pth"
TAPS = (2, 5, 8, 11)
MEAN, STD = torch.tensor([0.485, 0.456, 0.406]), torch.tensor([0.229, 0.224, 0.225])


def _color(x):
    """DiffAugment's brightness, saturation and contrast jitter, per image."""

    def rand():
        return torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device)

    x = x + (rand() - 0.5)
    m = x.mean(dim=1, keepdim=True)
    x = (x - m) * (rand() * 2) + m
    m = x.mean(dim=[1, 2, 3], keepdim=True)
    return (x - m) * (rand() + 0.5) + m


def _translation(x, ratio=0.125):
    """DiffAugment's translation by up to `ratio` of the image size, per image, filling with zeros."""
    B, _, H, W = x.shape
    sx, sy = int(H * ratio + 0.5), int(W * ratio + 0.5)
    tx = torch.randint(-sx, sx + 1, size=[B, 1, 1], device=x.device)
    ty = torch.randint(-sy, sy + 1, size=[B, 1, 1], device=x.device)
    b, gx, gy = torch.meshgrid(*(torch.arange(n, device=x.device) for n in (B, H, W)), indexing="ij")
    gx, gy = (gx + tx + 1).clamp(0, H + 1), (gy + ty + 1).clamp(0, W + 1)
    return F.pad(x, [1, 1, 1, 1]).permute(0, 2, 3, 1)[b, gx, gy].permute(0, 3, 1, 2)


def _cutout(x, ratio):
    """DiffAugment's cutout of a square of `ratio` of the image size at a random position, per image."""
    B, _, H, W = x.shape
    cw, ch = int(H * ratio + 0.5), int(W * ratio + 0.5)
    ox = torch.randint(0, H + (1 - cw % 2), size=[B, 1, 1], device=x.device)
    oy = torch.randint(0, W + (1 - ch % 2), size=[B, 1, 1], device=x.device)
    b, gx, gy = torch.meshgrid(*(torch.arange(n, device=x.device) for n in (B, cw, ch)), indexing="ij")
    mask = torch.ones(B, H, W, dtype=x.dtype, device=x.device)
    mask[b, (gx + ox - cw // 2).clamp(0, H - 1), (gy + oy - ch // 2).clamp(0, W - 1)] = 0
    return x * mask.unsqueeze(1)


AUGMENTS = {"color": _color, "translation": _translation,
            "cutout_0.2": lambda x: _cutout(x, 0.2), "cutout_0.5": lambda x: _cutout(x, 0.5)}


def diff_augment(x, policy="color,translation,cutout_0.2", p=0.5):
    """DiffAugment (Zhao et al., 2020): one coin per transform for the batch, magnitudes per image."""
    for name in policy.split(","):
        if torch.rand(1) <= p:
            x = AUGMENTS[name](x)
    return x.contiguous()


class Dino(nn.Module):
    """Frozen DINO ViT-S/16. Returns five token maps, before the first block and after blocks 2, 5, 8 and 11,
    each (B, 384, 196) with the CLS token added to every patch token."""

    def __init__(self):
        super().__init__()
        self.vit = timm.create_model("vit_small_patch16_224", num_classes=0, act_layer="gelu_tanh")
        state = torch.hub.load_state_dict_from_url(DINO_URL, map_location="cpu")
        for key in state:
            if key.endswith("attn.qkv.bias"):
                state[key][384:768] = 0  # the key bias, which softmax ignores, zeroed as in MAETok
        self.vit.load_state_dict(state)
        self.vit.requires_grad_(False)
        self.register_buffer("scale", (0.5 / STD).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("shift", ((0.5 - MEAN) / STD).view(1, 3, 1, 1), persistent=False)

    @staticmethod
    def draw_view(size=256):
        """A random 224 crop (i, j) with probability 0.5, otherwise None, a resize to 224."""
        if size > 224 and float(torch.rand(1)) <= 0.5:
            return int(torch.randint(0, size - 223, (1,))), int(torch.randint(0, size - 223, (1,)))
        return None

    def forward(self, x, view="draw"):
        """x in [-1, 1] at 256 px. view: a crop (i, j), None for the resize, or "draw" to draw one now."""
        if view == "draw":
            view = self.draw_view(x.shape[-1])
        with torch.autocast(x.device.type, enabled=False):
            x = self.scale * x.float() + self.shift
            if view is None:
                x = F.interpolate(x, size=(224, 224), mode="area" if x.shape[-1] > 224 else "bicubic")
            else:
                x = x[..., view[0]:view[0] + 224, view[1]:view[1] + 224]
        x = self.vit._pos_embed(self.vit.patch_embed(x).float())
        maps = [x]
        for i, block in enumerate(self.vit.blocks):
            x = block(x)
            if i in TAPS:
                maps.append(x)
        return [(m[:, 1:].float() + m[:, :1].float()).transpose(1, 2) for m in maps]


def _sn(module):
    return nn.utils.spectral_norm(module, n_power_iterations=1, eps=1e-12, dim=0)


class BatchNormLocal(nn.Module):
    """Batch normalization over groups of at most 8 images, without running statistics."""

    def __init__(self, channels, group=8, eps=1e-6):
        super().__init__()
        self.group, self.eps = group, eps
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))

    def forward(self, x):
        groups = torch.tensor_split(x.float(), math.ceil(x.size(0) / self.group))
        x = torch.cat([(g - g.mean((0, 2), keepdim=True)) / torch.sqrt(g.var((0, 2), keepdim=True, unbiased=False)
                                                                        + self.eps) for g in groups])
        return x * self.weight[None, :, None] + self.bias[None, :, None]


class Residual(nn.Module):
    """(fn(x) + x) / sqrt(2), a residual connection that keeps the variance."""

    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return (self.fn(x) + x) * (1 / math.sqrt(2))


def _block(c, k):
    conv = _sn(nn.Conv1d(c, c, k, padding=k // 2, padding_mode="circular"))
    return nn.Sequential(conv, BatchNormLocal(c), nn.LeakyReLU(0.2))


def _head(c):
    """One logit per token: a k=1 block, a residual block of circular k=9 convolutions, a k=1 projection."""
    return nn.Sequential(_block(c, 1), Residual(_block(c, 9)), _sn(nn.Conv1d(c, 1, 1)))


class SourceAttention(nn.Module):
    """The tokens of the scored image attend to the tokens of the other image at the same tap, in a pre-norm
    block with LayerScale, and a k=1 convolution fuses [x; att; x - att]."""

    def __init__(self, c, heads=6):
        super().__init__()
        self.heads = heads
        self.norm_q, self.norm_kv, self.norm_mlp = (nn.LayerNorm(c, eps=1e-6) for _ in range(3))
        self.to_q = _sn(nn.Linear(c, c, bias=False))
        self.to_kv = _sn(nn.Linear(c, 2 * c, bias=False))
        self.proj = _sn(nn.Linear(c, c, bias=False))
        self.mlp = nn.Sequential(_sn(nn.Linear(c, 4 * c)), nn.GELU(approximate="tanh"), _sn(nn.Linear(4 * c, c)))
        self.ls_attn = nn.Parameter(torch.full((c,), 1e-4))
        self.ls_mlp = nn.Parameter(torch.full((c,), 1e-4))
        self.fuse = _sn(nn.Conv1d(3 * c, c, 1))

    def forward(self, x, y):
        B, C, L = x.shape
        q = self.to_q(self.norm_q(x.transpose(1, 2)))
        k, v = self.to_kv(self.norm_kv(y.transpose(1, 2))).chunk(2, dim=-1)
        q, k, v = (t.view(B, -1, self.heads, C // self.heads).transpose(1, 2) for t in (q, k, v))
        att = self.proj(F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(B, L, C))
        h = x.transpose(1, 2) + self.ls_attn * att
        h = h + self.ls_mlp * self.mlp(self.norm_mlp(h))
        att = att.transpose(1, 2)
        return self.fuse(torch.cat([h.transpose(1, 2), att, x - att], dim=1))


class Discriminator(nn.Module):
    """Both discriminators on one frozen backbone, with one head per tap each, so 5 x 196 logits per image.

    The interpolation discriminator scores a decoded mixture beside each of its two sources. Without source
    conditioning (`conditional=False`) it scores the decoded mixture alone, like the reconstruction one."""

    def __init__(self, conditional=True, width=384):
        super().__init__()
        self.conditional = conditional
        self.backbone = Dino()
        n = len(TAPS) + 1
        self.heads = nn.ModuleList(_head(width) for _ in range(n))
        self.heads_recon = nn.ModuleList(_head(width) for _ in range(n))
        self.cond = nn.ModuleList(SourceAttention(width) for _ in range(n)) if conditional else None

    def sources(self, x, view):
        """The backbone's token maps of the source images, which the decoded mixtures attend to."""
        with torch.no_grad():
            return self.backbone(x, view)

    def interp_logits(self, y, sources, view):
        """The score belongs to the query: s(A|AB) and s(B|AB) are real, s(AB|A) and s(AB|B) fake.
        Returns the real and the fake logits, each of shape (2, B, 980)."""
        B, real, fake = y.size(0), [], []
        for head, attend, m, a, b in zip(self.heads, self.cond, self.backbone(y, view), *sources, strict=True):
            q, k = torch.cat([a, b, m, m]), torch.cat([m, m, a, b])
            if self.training and torch.is_grad_enabled():
                f = checkpoint(attend, q, k, use_reentrant=False).chunk(4)
            else:
                f = attend(q, k).chunk(4)
            real.append(torch.stack([head(f[0]), head(f[1])]).view(2, B, -1))
            fake.append(torch.stack([head(f[2]), head(f[3])]).view(2, B, -1))
        return torch.cat(real, -1), torch.cat(fake, -1)

    def logits(self, x, view="draw", recon=True):
        """Unconditional logits, (B, 980), of the reconstruction heads or of the interpolation heads without
        source conditioning. Each call draws its own view unless it is given one."""
        heads = self.heads_recon if recon else self.heads
        return torch.cat([h(m).view(x.size(0), -1) for h, m in zip(heads, self.backbone(x, view), strict=True)], 1)
