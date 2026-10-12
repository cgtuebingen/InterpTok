"""Decode the straight line between the codes of two images, as in the paper's figures:

    python src/interpolate.py a.jpg b.jpg --weights interptok_41k.safetensors --out a_to_b.png

Writes one row of 256 px tiles: the first image, the decoded codes (1 - t) a + t b at evenly spaced t from 0 to 1,
and the second image. Without --weights, MAETok decodes the line.
"""
import argparse

import numpy as np
import torch
from PIL import Image

import data
import model as tokenizer


@torch.no_grad()
def main():
    p = argparse.ArgumentParser(description="Decode the line between the codes of two images.")
    p.add_argument("images", nargs=2, help="two images, center-cropped to 256 px")
    p.add_argument("--weights", help="tokenizer weights (.safetensors), MAETok without them")
    p.add_argument("--steps", type=int, default=9, help="decoded codes on the line, both ends included")
    p.add_argument("--out", required=True, help="output image, such as a_to_b.png")
    args = p.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = tokenizer.load(args.weights).to(device).eval()

    photos = [data.center_crop(Image.open(f).convert("RGB")) for f in args.images]
    x = torch.from_numpy(np.stack(photos)).permute(0, 3, 1, 2).float().div(127.5).sub(1).to(device)
    z = tokenizer.encode(tok, x)
    t = torch.linspace(0, 1, args.steps, device=device).view(-1, 1, 1)
    y = tokenizer.decode(tok, (1 - t) * z[:1] + t * z[1:])
    tiles = ((y.clamp(-1, 1) + 1) * 127.5).round().byte().permute(0, 2, 3, 1).cpu().numpy()
    Image.fromarray(np.concatenate([photos[0], *tiles, photos[1]], axis=1)).save(args.out)
    print(f"{args.out}: {args.images[0]}, {args.steps} decoded codes, {args.images[1]}")


if __name__ == "__main__":
    main()
