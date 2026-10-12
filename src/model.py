"""MAETok-B-128 and InterpTok, which appends six zero-gated blocks to MAETok's decoder."""
import copy
import json
import logging
import sys
import warnings
from pathlib import Path

import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from torch import nn
from torch.utils.checkpoint import checkpoint

sys.path.append(str(Path(__file__).resolve().parents[1] / "maetok"))  # MAETok's code, a git submodule
with warnings.catch_warnings():  # MAETok registers its ViTs under timm's names, and timm warns once per name
    warnings.filterwarnings("ignore", "Overwriting .* in registry", UserWarning)
    from modelling.modules.timm_vit import vision_transformer
    from modelling.modules.timm_vit.rope_utils import apply_rotary_emb
    from modelling.tokenizer import AEModel, ModelArgs
# MAETok's rotary embedding calls a deprecated autocast, which torch warns about at the first decode
warnings.filterwarnings("ignore", r"`torch\.cuda\.amp\.autocast", FutureWarning, r"modelling\.")
# The Hub asks anonymous downloads to log in, which MAETok's public weights do not need
logging.getLogger("huggingface_hub.utils._http").addFilter(lambda r: "unauthenticated requests" not in r.getMessage())

MAETOK = "MAETok/maetok-b-128"


def _attention(self, x, attn_mask=None, freqs_cis=None, num_prefix_tokens=1, num_latent_tokens=32):
    """MAETok's attention, with the rotary embedding applied out of place so that autograd can train it."""
    B, N, C = x.shape
    q, k, v = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4).contiguous().unbind(0)
    q, k = self.q_norm(q), self.k_norm(k)
    if freqs_cis is not None:
        s, e = num_prefix_tokens, N - num_latent_tokens
        q_rot, k_rot = apply_rotary_emb(q[:, :, s:e], k[:, :, s:e], freqs_cis)
        q = torch.cat([q[:, :, :s], q_rot, q[:, :, e:]], dim=2)
        k = torch.cat([k[:, :, :s], k_rot, k[:, :, e:]], dim=2)
    x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p if self.training else 0.0)
    return self.proj_drop(self.proj(x.transpose(1, 2).reshape(B, N, C)))


vision_transformer.Attention.forward = _attention


class GatedBlock(nn.Module):
    """x + gate * (block(x) - x), with the gate at zero, so that a grown decoder starts as MAETok's."""

    def __init__(self, block):
        super().__init__()
        self.block = block
        self.gate = nn.Parameter(torch.zeros(()))

    def forward(self, x, *args, **kwargs):
        if self.training and torch.is_grad_enabled():
            out = checkpoint(self.block, x, *args, use_reentrant=False, **kwargs)
        else:
            out = self.block(x, *args, **kwargs)
        return x + self.gate * (out - x)


def grow_decoder(model, n):
    """Append gated copies of the decoder's last n blocks, each with the rotary frequencies of its source."""
    if n <= 0:
        return model
    decoder, blocks = model.decoder, model.decoder.model.blocks
    source = list(range(len(blocks) - n, len(blocks)))
    for i in source:
        blocks.append(GatedBlock(copy.deepcopy(blocks[i])))
    decoder.freqs = nn.Parameter(torch.cat([decoder.freqs.data, decoder.freqs.data[:, source]], dim=1))
    return model


def load(weights=None, grow=0):
    """MAETok-B-128 grown by `grow` decoder blocks, or the tokenizer stored in `weights` (.safetensors). The latter
    is built from MAETok's configuration, without downloading MAETok's weights."""
    if weights is None:
        return grow_decoder(AEModel.from_pretrained(MAETOK), grow)
    state = load_file(weights)
    model = AEModel(ModelArgs(**json.loads(Path(hf_hub_download(MAETOK, "config.json")).read_text())))
    depth = len({k.split(".")[3] for k in state if k.startswith("decoder.model.blocks.")})
    grow_decoder(model, depth - len(model.decoder.model.blocks))
    model.load_state_dict(state)
    return model


def encode(model, x):
    """Images in [-1, 1] of shape (B, 3, 256, 256) to codes of shape (B, 128, 32)."""
    return model.encode(x)[0]


def decode(model, z):
    """Codes of shape (B, 128, 32) to images in [-1, 1]."""
    return model.decode(z)


class Tokenizer(nn.Module):
    """A frozen tokenizer with the interface of the evaluation code of Xu et al. (2026)."""

    supports_y, num_classes = False, 0

    def __init__(self, weights=None):
        super().__init__()
        self.model = load(weights).eval().requires_grad_(False)

    def encode(self, x, *args, **kwargs):
        """Codes of images x; further arguments of the caller, such as class labels, are ignored."""
        return encode(self.model, x)

    def decode(self, z, *args, **kwargs):
        """Images of codes z; further arguments of the caller are ignored."""
        return decode(self.model, z)
