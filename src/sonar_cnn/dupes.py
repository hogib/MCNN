"""Near-duplicate analysis: how similar are test images to their nearest training image?

Images are shrunk to 32x32 thumbnails (size and aspect removed), mean-centered and
L2-normalized, so similarity is cosine/correlation of the content.

  uv run python -m sonar_cnn.dupes
"""

import json
import re

import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split

from .data import DATA_ROOT
from .train import RESULTS

T = 32


def load_thumbs():
    classes = sorted(p.name for p in DATA_ROOT.iterdir() if p.is_dir())
    feats, labels, idx = [], [], []
    for ci, c in enumerate(classes):
        # Same (alphabetical) order as data.load_images, so indices line up.
        # Functions that need file-number order sort by idx themselves.
        files = sorted((DATA_ROOT / c).glob("*.png"))
        for f in files:
            im = Image.open(f).convert("L").resize((T, T), Image.BILINEAR)
            v = np.asarray(im, np.float32).ravel()
            v -= v.mean()
            feats.append(v / (np.linalg.norm(v) + 1e-8))
            labels.append(ci)
            idx.append(int(re.search(r"(\d+)$", f.stem)[1]))
    return np.stack(feats), np.array(labels), np.array(idx), classes


def pseudo_sequences(X, labels, threshold=0.85):
    """Group images into approximate recording sequences: within each class,
    single-linkage clusters of thumbnails with correlation >= threshold, so any
    chain of near-duplicates ends up in one group. Returns a global group id per image."""
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform

    groups = np.zeros(len(labels), int)
    nxt = 0
    for c in np.unique(labels):
        m = np.where(labels == c)[0]
        D = np.clip(1 - X[m] @ X[m].T, 0, None)
        np.fill_diagonal(D, 0)
        lab = fcluster(linkage(squareform(D, checks=False), "single"), 1 - threshold, "distance")
        groups[m] = lab - 1 + nxt
        nxt += lab.max()
    return groups


def group_split(labels, groups, test_frac, seed):
    """Per class, put whole groups in the test set (in random order) until it holds
    ~test_frac of the class. Skips groups that would overshoot by more than 25%."""
    rng = np.random.default_rng(seed)
    te = []
    for c in np.unique(labels):
        m = np.where(labels == c)[0]
        target = test_frac * len(m)
        gs = np.unique(groups[m])
        rng.shuffle(gs)
        n = 0
        for g in gs:
            size = (groups[m] == g).sum()
            if n + size <= 1.25 * target:
                te += list(m[groups[m] == g])
                n += size
            if n >= target:
                break
    te = np.array(sorted(te))
    tr = np.setdiff1d(np.arange(len(labels)), te)
    return tr, te


def nn_stats(S, labels, tr, te):
    """1-NN accuracy and nearest-train-similarity for the given split."""
    sub = S[np.ix_(te, tr)]
    nn = sub.argmax(1)
    best = sub.max(1)
    acc = (labels[tr][nn] == labels[te]).mean()
    return acc, best


def block_split(labels, idx, test_frac, offset):
    """Per class, take a contiguous run of file numbers as test (a proxy for a
    split by sequence, if file numbers follow recording order)."""
    tr, te = [], []
    for c in np.unique(labels):
        members = np.where(labels == c)[0]
        members = members[np.argsort(idx[members])]
        n = len(members)
        k = int(round(test_frac * n))
        start = int(offset * n) % n
        test_pos = (np.arange(k) + start) % n
        mask = np.zeros(n, bool)
        mask[test_pos] = True
        te += list(members[mask])
        tr += list(members[~mask])
    return np.array(tr), np.array(te)


def main():
    X, labels, idx, classes = load_thumbs()
    S = X @ X.T
    out = {}

    # Q1: are consecutive file numbers adjacent frames?
    cons, rand_same = [], []
    rng = np.random.default_rng(0)
    for c in np.unique(labels):
        m = np.where(labels == c)[0]
        m = m[np.argsort(idx[m])]
        cons += [S[a, b] for a, b in zip(m[:-1], m[1:])]
        pairs = rng.choice(m, size=(len(m), 2))
        rand_same += [S[a, b] for a, b in pairs if a != b]
    cons, rand_same = np.array(cons), np.array(rand_same)
    print("Similarity of consecutive file numbers vs random same-class pairs:")
    print(f"  consecutive: median {np.median(cons):.3f}  >0.9: {(cons > 0.9).mean():.1%}")
    print(f"  random same-class: median {np.median(rand_same):.3f}  >0.9: {(rand_same > 0.9).mean():.1%}")
    out["consecutive_median"] = float(np.median(cons))
    out["random_same_class_median"] = float(np.median(rand_same))

    # Q2: random split (the paper's protocol) vs contiguous block split.
    print("\nNearest training image for each test image (80/20):")
    rows = []
    for name, splits in [
        ("random", [train_test_split(np.arange(len(labels)), train_size=0.8, stratify=labels, random_state=s) for s in range(3)]),
        ("block", [block_split(labels, idx, 0.2, off) for off in (0.0, 0.4, 0.8)]),
        ("group", [group_split(labels, pseudo_sequences(X, labels), 0.2, s) for s in range(3)]),
    ]:
        accs, b = [], []
        for tr, te in splits:
            a, best = nn_stats(S, labels, tr, te)
            accs.append(a)
            b.append(best)
        b = np.concatenate(b)
        r = {
            "split": name,
            "nn1_acc": float(np.mean(accs)),
            "nn1_acc_std": float(np.std(accs)),
            "median_best_sim": float(np.median(b)),
            "frac_best_sim_gt_0.95": float((b > 0.95).mean()),
            "frac_best_sim_gt_0.90": float((b > 0.90).mean()),
        }
        rows.append(r)
        print(f"  {name:6s}: 1-NN thumbnail acc {100 * r['nn1_acc']:.2f} ± {100 * r['nn1_acc_std']:.2f}  "
              f"median best sim {r['median_best_sim']:.3f}  "
              f">0.95: {r['frac_best_sim_gt_0.95']:.1%}  >0.90: {r['frac_best_sim_gt_0.90']:.1%}")
    out["splits"] = rows
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "dupes.json").write_text(json.dumps(out, indent=2))
    print(f"wrote {RESULTS / 'dupes.json'}")


if __name__ == "__main__":
    main()
