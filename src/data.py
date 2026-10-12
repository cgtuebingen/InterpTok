"""ImageNet at 256 px, and the per-epoch pairing of similar training images."""
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

PAIR_SEED = 1234


def image_files(root):
    """The images of an ImageNet split stored as root/<class>/<image>, as sorted paths relative to root."""
    root = Path(root)
    images = (p for p in root.glob("*/*") if p.suffix.lower() in (".jpeg", ".jpg", ".png"))
    return sorted(p.relative_to(root).as_posix() for p in images)


def center_crop(image, size=256):
    """ADM's center crop (Dhariwal and Nichol, 2021)."""
    while min(*image.size) >= 2 * size:
        image = image.resize(tuple(s // 2 for s in image.size), resample=Image.BOX)
    scale = size / min(*image.size)
    image = image.resize(tuple(round(s * scale) for s in image.size), resample=Image.BICUBIC)
    a = np.array(image)
    top, left = (a.shape[0] - size) // 2, (a.shape[1] - size) // 2
    return a[top:top + size, left:left + size]


class Images(Dataset):
    """The images of an ImageNet split in [-1, 1], of shape (3, 256, 256), returned with their index."""

    def __init__(self, root, size=256):
        self.root, self.files, self.size = Path(root), image_files(root), size
        if not self.files:
            raise FileNotFoundError(f"no images in {root}, which should hold them as <class>/<image>")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        a = center_crop(Image.open(self.root / self.files[i]).convert("RGB"), self.size)
        return torch.from_numpy(a).permute(2, 0, 1).float() / 127.5 - 1.0, i


def epoch_pairs(neighbours, epoch, k_start=8, widen_every=2):
    """Disjoint pairs for one epoch. In a seeded order, every unpaired image is paired with its most similar
    unpaired image among its k nearest neighbours, where k doubles every `widen_every` epochs. With K
    neighbours precomputed and k > K, an image takes such a neighbour with probability K / k and a random
    unpaired image otherwise. Images whose k neighbours are all paired stay unpaired in that epoch."""
    n, depth = neighbours.shape
    k = min(k_start * 2 ** (epoch // widen_every), n)
    wide = k > depth
    rng = np.random.default_rng(PAIR_SEED * 1_000_003 + epoch)
    order = rng.permutation(n)
    used = np.zeros(n, dtype=bool)
    free, where, n_free = np.arange(n), np.arange(n), n  # unpaired images, for drawing one at random

    def remove(i):
        nonlocal n_free
        last = free[n_free - 1]
        free[where[i]], where[last] = last, where[i]
        n_free -= 1

    pairs = []
    for i in order:
        if used[i]:
            continue
        j = -1
        if not wide or rng.random() < depth / k:
            j = next((int(c) for c in neighbours[i, :min(k, depth)] if c != i and not used[c]), -1)
        if j < 0 and wide and n_free > 1:
            while (j := int(free[rng.integers(n_free)])) == i:
                pass
        if j >= 0:
            pairs.append((int(i), j))
            used[i] = used[j] = True
            remove(i)
            remove(j)
    return np.asarray(pairs, dtype=np.int64).reshape(-1, 2)


class PairBatches:
    """Micro-batches of `n_pairs` pairs from the current epoch's pairing, as dataset indices in pair order.
    The GPUs read disjoint slices of the same sequence, and `skip` resumes it after that many micro-batches."""

    def __init__(self, neighbours, n_pairs=4, rank=0, world=1, skip=0):
        self.neighbours, self.n_pairs, self.rank, self.world, self.skip = neighbours, n_pairs, rank, world, skip

    def __iter__(self):
        step, epoch, skip = self.n_pairs * self.world, 0, self.skip
        while True:
            pairs = epoch_pairs(self.neighbours, epoch)
            epoch += 1
            batches = len(pairs) // step
            if batches == 0:
                raise ValueError(f"{len(pairs)} pairs in an epoch, too few for micro-batches of {step} pairs")
            if skip >= batches:
                skip -= batches
                continue
            for b in range(skip, batches):
                start = b * step + self.rank * self.n_pairs
                yield pairs[start:start + self.n_pairs].reshape(-1).tolist()
            skip = 0
