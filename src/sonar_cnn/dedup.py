"""Build a near-duplicate-free copy of the dataset.

Within each class, walk the images in a seeded random order and keep one only if
its thumbnail correlation with every image already kept is below the threshold
(see dupes.load_thumbs). Output has the same layout as watertank-cropped, so
data.load_images(root=...) reads it directly.

  uv run python -m sonar_cnn.dedup --stats                # counts at several thresholds
  uv run python -m sonar_cnn.dedup --threshold 0.9        # writes data/dedup_t90/
"""

import argparse
import csv
import shutil
from pathlib import Path

import numpy as np

from .data import DATA_ROOT
from .dupes import load_thumbs

OUT_ROOT = Path(__file__).resolve().parents[2] / "data"


def files_in_order():
    """Paths in the same order as dupes.load_thumbs / data.load_images."""
    classes = sorted(p.name for p in DATA_ROOT.iterdir() if p.is_dir())
    return [f for c in classes for f in sorted((DATA_ROOT / c).glob("*.png"))]


def greedy_keep(X, labels, threshold, seed=0):
    """Returns (keep mask, rep) where rep[i] is the kept image that i duplicates (or i)."""
    rng = np.random.default_rng(seed)
    keep = np.zeros(len(labels), bool)
    rep = np.arange(len(labels))
    for c in np.unique(labels):
        m = rng.permutation(np.where(labels == c)[0])
        kept = []
        for i in m:
            if kept:
                sims = X[kept] @ X[i]
                j = int(sims.argmax())
                if sims[j] >= threshold:
                    rep[i] = kept[j]
                    continue
            kept.append(i)
        keep[kept] = True
    return keep, rep


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--threshold", type=float, default=0.9)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--stats", action="store_true")
    args = p.parse_args()

    X, labels, _, classes = load_thumbs()

    if args.stats:
        print(f"{'thr':>5} {'total':>6}  " + " ".join(f"{c[:8]:>8}" for c in classes))
        print(f"{'orig':>5} {len(labels):6d}  " + " ".join(f"{(labels == i).sum():8d}" for i in range(len(classes))))
        for t in (0.98, 0.95, 0.92, 0.9, 0.85, 0.8):
            keep, _ = greedy_keep(X, labels, t, args.seed)
            print(f"{t:5.2f} {keep.sum():6d}  " + " ".join(f"{(keep & (labels == i)).sum():8d}" for i in range(len(classes))))
        return

    keep, rep = greedy_keep(X, labels, args.threshold, args.seed)
    files = files_in_order()
    assert len(files) == len(labels)
    out = OUT_ROOT / f"dedup_t{round(100 * args.threshold)}"
    if out.exists():
        shutil.rmtree(out)
    for i in np.where(keep)[0]:
        dst = out / classes[labels[i]] / files[i].name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(files[i], dst)
    with open(out / "manifest.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["class", "file", "kept", "duplicate_of", "similarity"])
        for i, f in enumerate(files):
            w.writerow([classes[labels[i]], f.name, int(keep[i]),
                        "" if keep[i] else files[rep[i]].name,
                        "" if keep[i] else f"{X[i] @ X[rep[i]]:.4f}"])
    print(f"kept {keep.sum()} / {len(keep)} images -> {out}")
    for ci, c in enumerate(classes):
        print(f"  {c:16s} {(keep & (labels == ci)).sum():4d} / {(labels == ci).sum():4d}")


if __name__ == "__main__":
    main()
