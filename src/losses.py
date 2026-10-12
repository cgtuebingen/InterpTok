"""The losses of InterpTok beside L1 and LPIPS: adversarial, mixing ratio, alignment and representation Fréchet."""
import numpy as np
import timm
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn

TEACHER = "facebook/dinov3-vitb16-pretrain-lvd1689m"  # the frozen vision model of the alignment losses
IMAGENET_MEAN, IMAGENET_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


def hinge_d(real, fake):
    """Discriminator hinge on every logit."""
    return 0.5 * (F.relu(1.0 - real).mean() + F.relu(1.0 + fake).mean())


def interp_d(real, fake):
    """Interpolation discriminator hinge, with the two real and the two fake passes averaged first."""
    return hinge_d(real.mean(0), fake.mean(0))


def interp_g(real, fake):
    """Relativistic generator loss over all four passes, in each of which the decoded mixture is query or key."""
    return real.mean(0).mean() - fake.mean(0).mean()


class LeCam:
    """LeCAM regularization (Tseng et al., 2021) against moving averages of the discriminator's outputs."""

    def __init__(self, decay=0.999):
        self.decay, self.real, self.fake = decay, 0.0, 0.0

    def __call__(self, real, fake):
        return F.relu(real - self.fake).pow(2).mean() + F.relu(self.real - fake).pow(2).mean()

    def update(self, real, fake):
        self.real = self.real * self.decay + real * (1.0 - self.decay)
        self.fake = self.fake * self.decay + fake * (1.0 - self.decay)


def adaptive_weight(rec, adv, last_layer, cap=1.0):
    """VQGAN's ratio of the reconstruction and adversarial gradient norms at the decoder's last layer."""
    g_rec = torch.autograd.grad(rec, last_layer, retain_graph=True)[0]
    g_adv = torch.autograd.grad(adv, last_layer, retain_graph=True)[0]
    return min(float(torch.clamp(g_rec.norm() / (g_adv.norm() + 1e-4), 0.0, 1e4)), cap)


def pixel_ratio_loss(y, y_a, y_b, alpha):
    """Pixel mixing ratio loss: the decoded mixture lies a fraction alpha of the way from D(a) to D(b), in L1."""
    d_a = (y - y_a).abs().flatten(1).mean(1)
    d_b = (y - y_b).abs().flatten(1).mean(1)
    return ((d_a / (d_a + d_b).clamp_min(1e-8) - alpha) ** 2).mean()


def angle(u, v):
    """Angle between unit vectors, exact at 0 and at pi, where an arc cosine loses precision."""
    return 2.0 * torch.atan2(torch.sqrt(((u - v) ** 2).sum(-1) + 1e-12), torch.sqrt(((u + v) ** 2).sum(-1) + 1e-12))


def feature_ratio_loss(f, f_a, f_b, alpha):
    """Feature mixing ratio loss: the angles from f to the sources' features split their angle at alpha."""
    f, f_a, f_b = (F.normalize(t.float(), dim=-1) for t in (f, f_a, f_b))
    ab = angle(f_a, f_b)
    return ((angle(f, f_a) - alpha * ab) ** 2 + (angle(f, f_b) - (1.0 - alpha) * ab) ** 2).mean()


def axis_loss(z, t):
    """Distance-matrix alignment: the cosine similarities between codes match those between features."""
    z, t = F.normalize(z.float(), dim=-1), F.normalize(t.float(), dim=-1)
    return F.mse_loss(z @ z.T, t @ t.T)


def cosine_loss(pred, target):
    """One minus the cosine similarity of each row of pred to the same row of target, averaged."""
    return (1.0 - (F.normalize(pred.float(), dim=-1) * F.normalize(target.float(), dim=-1)).sum(-1)).mean()


def mlp(dim_in, dim_out, hidden=1024):
    """The two-layer head that maps codes or decoder tokens to the vision model's features."""
    return nn.Sequential(nn.Linear(dim_in, hidden), nn.SiLU(), nn.Linear(hidden, dim_out))


class SigLIP(nn.Module):
    """SigLIP2-L/16 at 256 px, pooled output."""

    def __init__(self):
        super().__init__()
        from transformers import SiglipVisionModel
        from transformers.utils import logging
        level = logging.get_verbosity()
        logging.set_verbosity_error()  # the checkpoint also holds the text tower, which the vision model skips
        self.model = SiglipVisionModel.from_pretrained("google/siglip2-large-patch16-256")
        logging.set_verbosity(level)

    def forward(self, x):
        return self.model(pixel_values=(x - 0.5) / 0.5).pooler_output


class TimmEncoder(nn.Module):
    """A pretrained timm network as a feature extractor, on images in [0, 1] resized to its input size."""

    def __init__(self, name, size, **kwargs):
        super().__init__()
        self.model, self.size = timm.create_model(name, pretrained=True, num_classes=0, **kwargs), size
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1), persistent=False)

    def forward(self, x):
        x = F.interpolate(x, size=(self.size, self.size), mode="bilinear", align_corners=False)
        return self.model((x - self.mean) / self.std)


ENCODERS = {"siglip": SigLIP,
            "inception": lambda: TimmEncoder("inception_v3.tv_in1k", 299),
            "mae": lambda: TimmEncoder("vit_large_patch16_224.mae", 224, global_pool="token")}


def frechet_distance(mu, sigma, mu_r, sqrt_sigma_r, trace_sigma_r):
    """Squared Fréchet distance in float64; tr sqrt(S S_r) is the sum of the roots of the eigenvalues of
    S_r^1/2 S S_r^1/2, so one symmetric eigendecomposition suffices and gradients reach mu and sigma."""
    mu, sigma = mu.double(), sigma.double()
    s = sqrt_sigma_r @ sigma @ sqrt_sigma_r
    cross = torch.linalg.eigvalsh(0.5 * (s + s.T)).clamp_min(1e-12).sqrt().sum()
    return (((mu - mu_r) ** 2).sum() + sigma.diagonal().sum() + trace_sigma_r - 2.0 * cross).clamp_min(0.0)


class Moments(nn.Module):
    """One feature space: the reference moments of the training images, the moving moments of the decoded
    images, and the feature sums of the current optimizer step."""

    def __init__(self, mu, sigma):
        super().__init__()
        sigma = torch.as_tensor(sigma, dtype=torch.float64)
        eig, vec = torch.linalg.eigh(0.5 * (sigma + sigma.T))
        self.register_buffer("mu_r", torch.as_tensor(mu, dtype=torch.float64))
        self.register_buffer("sqrt_r", (vec * eig.clamp_min(0.0).sqrt()) @ vec.T)
        self.register_buffer("trace_r", sigma.diagonal().sum())
        d = sigma.shape[0]
        for name, shape in (("mean", (d,)), ("second", (d, d)), ("sum", (d,)), ("sum2", (d, d))):
            self.register_buffer(name, torch.zeros(shape))


class FrechetLoss(nn.Module):
    """Representation Fréchet loss (Yang et al., 2026). In each feature space, the Fréchet distance between the
    decoded images and the training images, normalized by its own detached value. The moments of the decoded
    images are a bias-corrected moving average over optimizer steps, blended with the current micro-batch, and
    the loss starts once they have accumulated over `warmup` steps."""

    def __init__(self, reference, names=("siglip", "inception", "mae"), beta=0.999, warmup=500):
        super().__init__()
        self.beta, self.warmup = beta, warmup
        self.encoders = nn.ModuleDict({n: ENCODERS[n]() for n in names}).requires_grad_(False).eval()
        self.moments = nn.ModuleDict({n: Moments(**np.load(f"{reference}/{n}.npz")) for n in names})
        self.register_buffer("count", torch.zeros(()))
        self.register_buffer("steps", torch.zeros((), dtype=torch.long))

    def train(self, mode=True):
        """Switch the module's mode; the frozen encoders always stay in eval mode."""
        super().train(mode)
        self.encoders.eval()
        return self

    def forward(self, x):
        """x: decoded images in [0, 1]. Returns the loss, zero during the warm-up."""
        terms, t = [], int(self.steps)
        for name, m in self.moments.items():
            with torch.autocast(x.device.type, torch.bfloat16):
                f = self.encoders[name](x)
            f = f.float()
            with torch.no_grad():
                m.sum.add_(f.sum(0))
                m.sum2.add_(f.t() @ f)
            if t >= self.warmup:
                c = 1.0 - self.beta ** t
                mu = self.beta * (m.mean / c) + (1.0 - self.beta) * f.mean(0)
                second = self.beta * (m.second / c) + (1.0 - self.beta) * (f.t() @ f) / f.shape[0]
                fd = frechet_distance(mu, second - mu[:, None] * mu[None, :], m.mu_r, m.sqrt_r, m.trace_r).float()
                terms.append(fd / (fd.detach() + 0.01))
        self.count += x.shape[0]
        return torch.stack(terms).sum() if terms else x.new_zeros(())

    @torch.no_grad()
    def step(self):
        """At each optimizer step: fold the step's feature sums, from every GPU, into the moving moments."""
        sums = [s for m in self.moments.values() for s in (m.sum, m.sum2)] + [self.count]
        if dist.is_initialized():
            for s in sums:
                dist.all_reduce(s)
        for m in self.moments.values():
            m.mean.mul_(self.beta).add_(m.sum / self.count, alpha=1.0 - self.beta)
            m.second.mul_(self.beta).add_(m.sum2 / self.count, alpha=1.0 - self.beta)
        for s in sums:
            s.zero_()
        self.steps += 1
