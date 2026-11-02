"""Loading the Valdenegro-Toro watertank FLS dataset and turning it into batches."""

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

DATA_ROOT = (
    Path(__file__).resolve().parents[2]
    / "data/raw/marine-debris-fls-datasets/md_fls_dataset/data/watertank-cropped"
)


def load_images(root: Path = DATA_ROOT):
    """Return (list of HxW uint8 arrays, int labels, class names).

    The PNGs are RGBA with R=G=B and constant alpha, so we keep one channel.
    """
    classes = sorted(p.name for p in root.iterdir() if p.is_dir())
    images, labels = [], []
    for ci, c in enumerate(classes):
        for f in sorted((root / c).glob("*.png")):
            images.append(np.asarray(Image.open(f).convert("L")))
            labels.append(ci)
    return images, np.array(labels), classes


class FixedPadDataset(Dataset):
    """Center zero-pad every image to size x size, as the paper does.

    silhouette=True replaces the sonar content with 1s, so the only information
    left is where the image ends and the zero border begins. It's a probe for
    whether the padding gives the class away.
    """

    def __init__(self, images, labels, size=305, silhouette=False):
        self.images, self.labels = images, labels
        self.size, self.silhouette = size, silhouette

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        img = self.images[i]
        h, w = img.shape
        out = np.zeros((self.size, self.size), np.float32)
        top, left = (self.size - h) // 2, (self.size - w) // 2
        out[top : top + h, left : left + w] = 1.0 if self.silhouette else img / 255.0
        return torch.from_numpy(out)[None], (h, w), int(self.labels[i])


class NativeDataset(Dataset):
    """Images at their original size. constant=True is the same size-only probe
    as the silhouette, but for the variable-size model."""

    def __init__(self, images, labels, constant=False):
        self.images, self.labels, self.constant = images, labels, constant

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        img = self.images[i]
        x = np.ones(img.shape, np.float32) if self.constant else img.astype(np.float32) / 255.0
        return torch.from_numpy(x)[None], img.shape, int(self.labels[i])


def collate(batch):
    """Stack a batch, padding to the largest image at the bottom/right if sizes differ.

    Returns (x, sizes, y). sizes[i] = (h, w) of the real image, which VarCNN uses
    to mask the padded area so it never affects the output.
    """
    xs, shapes, ys = zip(*batch)
    H = max(x.shape[1] for x in xs)
    W = max(x.shape[2] for x in xs)
    out = torch.zeros(len(xs), 1, H, W)
    for i, x in enumerate(xs):
        out[i, :, : x.shape[1], : x.shape[2]] = x
    # For FixedPadDataset, the real size is the full padded canvas.
    sizes = torch.tensor([[x.shape[1], x.shape[2]] for x in xs])
    return out, sizes, torch.tensor(ys)
