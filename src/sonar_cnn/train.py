"""Experiments.

  uv run sonar-cnn size-baseline              # classify from (h, w) alone, no pixels
  uv run sonar-cnn train --model mcnn --input pad          # paper replication
  uv run sonar-cnn train --model mcnn --input silhouette   # padding-leak probe
  uv run sonar-cnn train --model varcnn --input native     # variable-size model
  uv run sonar-cnn train --model varcnn --input native-constant  # size-leak probe
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from .data import FixedPadDataset, NativeDataset, RandomSizeDataset, collate, load_images
from .models import MODELS

RESULTS = Path(__file__).resolve().parents[2] / "results"


def split(labels, train_frac, seed):
    idx = np.arange(len(labels))
    return train_test_split(idx, train_size=train_frac, stratify=labels, random_state=seed)


def make_dataset(kind, images, labels, train=False, seed=0):
    if kind.startswith("randsize"):
        return RandomSizeDataset(
            images, labels, train,
            pad_to=305 if "pad" in kind else None,
            constant="constant" in kind,
            seed=seed + (0 if train else 10_000),
        )
    if kind == "pad":
        return FixedPadDataset(images, labels)
    if kind == "silhouette":
        return FixedPadDataset(images, labels, silhouette=True)
    if kind == "native":
        return NativeDataset(images, labels)
    if kind == "native-constant":
        return NativeDataset(images, labels, constant=True)
    raise ValueError(kind)


def evaluate(model, loader, device):
    model.eval()
    preds, ys = [], []
    with torch.no_grad():
        for x, sizes, y in loader:
            preds.append(model(x.to(device), sizes).argmax(1).cpu())
            ys.append(y)
    return torch.cat(ys).numpy(), torch.cat(preds).numpy()


def run_one(args, images, labels, seed, device):
    torch.manual_seed(seed)
    tr, te = split(labels, args.train_frac, seed)
    pick = lambda ix: ([images[i] for i in ix], labels[ix])
    train_ds = make_dataset(args.input, *pick(tr), train=True, seed=seed)
    test_ds = make_dataset(args.input, *pick(te), train=False, seed=seed)
    g = torch.Generator().manual_seed(seed)
    train_dl = DataLoader(train_ds, args.bs, shuffle=True, collate_fn=collate, generator=g)
    test_dl = DataLoader(test_ds, 64, collate_fn=collate)

    model = MODELS[args.model]().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs * len(train_dl))
    t0 = time.time()
    for ep in range(args.epochs):
        model.train()
        tot = 0.0
        for x, sizes, y in train_dl:
            x, y = x.to(device), y.to(device)
            loss = F.cross_entropy(model(x, sizes), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item() * len(y)
        if (ep + 1) % 5 == 0 or ep == 0:
            print(f"  seed {seed} ep {ep + 1:3d} loss {tot / len(train_ds):.4f}", flush=True)
    yt, yp = evaluate(model, test_dl, device)
    return {
        "seed": seed,
        "acc": accuracy_score(yt, yp),
        "macro_f1": f1_score(yt, yp, average="macro"),
        "errors": int((yt != yp).sum()),
        "n_test": len(yt),
        "params": sum(p.numel() for p in model.parameters()),
        "train_s": round(time.time() - t0, 1),
    }


def summarize(name, runs, extra):
    acc = np.array([r["acc"] for r in runs])
    f1 = np.array([r["macro_f1"] for r in runs])
    print(f"{name}: acc {100 * acc.mean():.2f} ± {100 * acc.std():.2f}  "
          f"macro-F1 {100 * f1.mean():.2f} ± {100 * f1.std():.2f}  ({len(runs)} seeds)")
    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / f"{name}.json"
    out.write_text(json.dumps({**extra, "runs": runs}, indent=2))
    print(f"wrote {out}")


def cmd_train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    images, labels, classes = load_images()
    runs = []
    for seed in args.seeds:
        r = run_one(args, images, labels, seed, device)
        print(f"  seed {seed}: acc {100 * r['acc']:.2f}  errors {r['errors']}/{r['n_test']}")
        runs.append(r)
    name = f"{args.model}_{args.input}_tr{int(100 * args.train_frac)}"
    summarize(name, runs, {"args": vars(args) | {"func": None}, "classes": classes})


def cmd_size_baseline(args):
    """How well can the class be predicted from the crop size alone?"""
    images, labels, classes = load_images()
    hw = np.array([im.shape for im in images])
    print("per-class mean (h, w) and range:")
    for ci, c in enumerate(classes):
        s = hw[labels == ci]
        print(f"  {c:16s} n={len(s):4d}  mean {s[:, 0].mean():6.1f} x {s[:, 1].mean():6.1f}  "
              f"h {s[:, 0].min()}-{s[:, 0].max()}  w {s[:, 1].min()}-{s[:, 1].max()}")
    runs = []
    for seed in args.seeds:
        tr, te = split(labels, args.train_frac, seed)
        clf = RandomForestClassifier(300, random_state=seed).fit(hw[tr], labels[tr])
        yp = clf.predict(hw[te])
        runs.append({"seed": seed, "acc": accuracy_score(labels[te], yp),
                     "macro_f1": f1_score(labels[te], yp, average="macro")})
    summarize(f"size_baseline_tr{int(100 * args.train_frac)}", runs, {"classes": classes})


def main():
    p = argparse.ArgumentParser(prog="sonar-cnn")
    sub = p.add_subparsers(required=True)

    def common(sp):
        sp.add_argument("--train-frac", type=float, default=0.8)
        sp.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])

    t = sub.add_parser("train")
    t.add_argument("--model", choices=list(MODELS), default="mcnn")
    t.add_argument("--input", choices=["pad", "silhouette", "native", "native-constant",
                            "randsize", "randsize-constant", "randsize-pad", "randsize-pad-constant"], default="pad")
    t.add_argument("--epochs", type=int, default=30)
    t.add_argument("--bs", type=int, default=32)
    t.add_argument("--lr", type=float, default=1e-3)
    common(t)
    t.set_defaults(func=cmd_train)

    b = sub.add_parser("size-baseline")
    common(b)
    b.set_defaults(func=cmd_size_baseline)

    args = p.parse_args()
    args.func(args)
