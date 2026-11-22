"""MCNN under the detector's evaluation protocol.

MCNN (Çimen & Bolat; reimplemented in ~/Projects/sonar_cnn) classifies one
object crop, zero-padded to 305x305. Its published accuracy comes from a random
80/20 split of the `watertank-cropped` set, where near-duplicate frames sit on
both sides of the split. Here it is trained and tested on crops cut from the
same full frames, with the same scene folds, as the detector:

  crops   every annotated box of the 10 object classes (Wall excluded, as in
          watertank-cropped), cut from the 320x480 frame at native resolution
  folds   splits/scene_folds_v1.json; fold k test, fold (k+1) mod 5 validation
          (epoch selection), the other three training
  train   as in sonar_cnn: AdamW lr 1e-3, wd 1e-4, cosine, 30 epochs, batch 32

  uv run python -m sonar_cnn.scene.mcnn_cv train --seeds 0 1 2      # runs/mcnn_cv_s{seed}_f{k}
  uv run python -m sonar_cnn.scene.mcnn_cv summary                  # pooled accuracy, macro-F1
  uv run python -m sonar_cnn.scene.mcnn_cv onnx runs/mcnn_cv_s0_f0  # mcnn.onnx for the Pi benchmark
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .data import class_names, load_frames
from .split import SPLIT_FILE

ROOT = Path(__file__).resolve().parents[3]
PAD = 305


class MCNN(nn.Module):
    """Copy of sonar_cnn.models.MCNN (paper Table 2; ReLU after every conv)."""

    def __init__(self, n_classes=10):
        super().__init__()

        def conv(i, o, k, s=1, p=0):
            return [nn.Conv2d(i, o, k, s, p), nn.ReLU(inplace=True)]

        self.features = nn.Sequential(
            *conv(1, 32, 5, 2, 2), *conv(32, 32, 5, 2, 2), *conv(32, 32, 5, 2, 2), nn.MaxPool2d(2),
            *conv(32, 64, 3, 1, 1), *conv(64, 64, 3, 1, 1), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 2, padding="same"), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
        )
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(9 * 9 * 128, 256), nn.ReLU(inplace=True),
                                  nn.Linear(256, n_classes))

    def forward(self, x):
        return self.head(self.features(x))


def object_crops():
    """Crops of all non-Wall boxes: (list of HxW uint8, labels, fold per crop, (frame, box) ids, classes)."""
    frames = load_frames()
    classes = [c for c in class_names(frames) if c != "Wall"]
    fold_of = {int(i): f for i, f in json.loads(SPLIT_FILE.read_text())["fold"].items()}
    crops, labels, folds, ids = [], [], [], []
    for f in frames:
        img = None
        for j, b in enumerate(f.boxes):
            if b.cls == "Wall":
                continue
            img = f.load() if img is None else img
            crops.append(img[b.y : b.y + b.h, b.x : b.x + b.w].copy())
            labels.append(classes.index(b.cls))
            folds.append(fold_of[f.index])
            ids.append((f.index, j))
    return crops, np.array(labels), np.array(folds), ids, classes


def pad(crops):
    out = np.zeros((len(crops), 1, PAD, PAD), np.float32)
    for i, c in enumerate(crops):
        h, w = c.shape
        t, l = (PAD - h) // 2, (PAD - w) // 2
        out[i, 0, t : t + h, l : l + w] = c / 255.0
    return torch.from_numpy(out)


def predict(model, X, device, bs=128):
    model.eval()
    with torch.no_grad():
        return torch.cat([model(X[a : a + bs].to(device)).argmax(1).cpu() for a in range(0, len(X), bs)]).numpy()


def train_fold(X, y, folds, k, seed, args, device):
    torch.manual_seed(seed)
    np.random.seed(seed)
    v = (k + 1) % 5
    tr = np.where((folds != k) & (folds != v))[0]
    va, te = np.where(folds == v)[0], np.where(folds == k)[0]
    model = MCNN(int(y.max()) + 1).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    steps = args.epochs * ((len(tr) + args.bs - 1) // args.bs)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    g = torch.Generator().manual_seed(seed)
    yt = torch.from_numpy(y)
    best, log, t0 = (-1.0, 0, None), [], time.time()
    for ep in range(args.epochs):
        model.train()
        for b in torch.randperm(len(tr), generator=g).split(args.bs):
            i = tr[b.numpy()]
            loss = F.cross_entropy(model(X[i].to(device)), yt[i].to(device))
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
        acc_v = float((predict(model, X[va], device) == y[va]).mean())
        log.append({"epoch": ep + 1, "loss": loss.item(), "val_acc": acc_v})
        if acc_v > best[0]:
            best = (acc_v, ep + 1, {n: t.detach().cpu().clone() for n, t in model.state_dict().items()})
    model.load_state_dict(best[2])
    pt = predict(model, X[te], device)
    return model, {"seed": seed, "test_fold": k, "val_fold": v, "best_epoch": best[1], "val_acc": best[0],
                   "test_idx": te.tolist(), "test_pred": pt.tolist(), "test_acc": float((pt == y[te]).mean()),
                   "n_train": len(tr), "train_s": round(time.time() - t0, 1), "log": log}


def cmd_train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    crops, y, folds, ids, classes = object_crops()
    X = pad(crops)
    print(f"{len(crops)} crops, classes {classes}, per fold {np.bincount(folds).tolist()}", flush=True)
    for seed in args.seeds:
        for k in range(5):
            out = ROOT / "runs" / f"mcnn_cv_s{seed}_f{k}"
            out.mkdir(parents=True, exist_ok=True)
            model, r = train_fold(X, y, folds, k, seed, args, device)
            r |= {"classes": classes, "ids": [ids[i] for i in r["test_idx"]], "labels": y[r["test_idx"]].tolist()}
            torch.save({"model": model.state_dict(), "classes": classes}, out / "best.pt")
            (out / "results.json").write_text(json.dumps(r))
            print(f"seed {seed} fold {k}: test acc {100 * r['test_acc']:.2f} (best epoch {r['best_epoch']}, "
                  f"val {100 * r['val_acc']:.2f}, {r['train_s']} s)", flush=True)


def scores(yt, yp, n):
    """accuracy, macro-F1, per-class recall."""
    f1, rec = [], []
    for c in range(n):
        tp = np.sum((yp == c) & (yt == c))
        p, s = np.sum(yp == c), np.sum(yt == c)
        rec.append(tp / s if s else np.nan)
        f1.append(2 * tp / (p + s) if p + s else np.nan)
    return float(np.mean(yt == yp)), float(np.nanmean(f1)), rec


def cmd_summary(args):
    per_seed, out = [], {}
    for seed in args.seeds:
        rs = [json.loads((ROOT / "runs" / f"mcnn_cv_s{seed}_f{k}" / "results.json").read_text()) for k in range(5)]
        classes = rs[0]["classes"]
        yt = np.concatenate([r["labels"] for r in rs])
        yp = np.concatenate([r["test_pred"] for r in rs])
        acc, mf1, rec = scores(yt, yp, len(classes))
        per_seed.append({"seed": seed, "acc": acc, "macro_f1": mf1, "recall": rec,
                         "acc_per_fold": [r["test_acc"] for r in rs], "best_epoch": [r["best_epoch"] for r in rs]})
        print(f"seed {seed}: pooled acc {100 * acc:.2f}  macro-F1 {100 * mf1:.2f}  per fold "
              + " ".join(f"{100 * r['test_acc']:.1f}" for r in rs) + f"  best epochs {[r['best_epoch'] for r in rs]}")
    a = np.array([p["acc"] for p in per_seed])
    f = np.array([p["macro_f1"] for p in per_seed])
    R = np.array([p["recall"] for p in per_seed])
    print(f"mean over {len(a)} seeds: acc {100 * a.mean():.2f} +- {100 * a.std(ddof=1) if len(a) > 1 else 0:.2f}  "
          f"macro-F1 {100 * f.mean():.2f} +- {100 * f.std(ddof=1) if len(f) > 1 else 0:.2f}")
    for c, r in zip(classes, R.mean(0)):
        print(f"   {c:16s} recall {100 * r:5.1f}")
    out = {"per_seed": per_seed, "classes": classes, "acc_mean": float(a.mean()), "macro_f1_mean": float(f.mean()),
           "acc_sd": float(a.std(ddof=1)) if len(a) > 1 else None, "macro_f1_sd": float(f.std(ddof=1)) if len(f) > 1 else None,
           "recall_mean": R.mean(0).tolist()}
    (ROOT / "runs" / "mcnn_cv_summary.json").write_text(json.dumps(out, indent=1))
    print("wrote runs/mcnn_cv_summary.json")


def cmd_onnx(args):
    ck = torch.load(ROOT / args.run / "best.pt", map_location="cpu")
    m = MCNN(len(ck["classes"]))
    m.load_state_dict(ck["model"])
    m.eval()
    dest = ROOT / args.run / "mcnn.onnx"
    torch.onnx.export(m, torch.zeros(1, 1, PAD, PAD), dest, input_names=["x"], output_names=["logits"],
                      dynamic_axes={"x": {0: "n"}, "logits": {0: "n"}}, opset_version=17, dynamo=False)
    print(f"wrote {dest.relative_to(ROOT)}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(required=True)
    t = sub.add_parser("train")
    t.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    t.add_argument("--epochs", type=int, default=30)
    t.add_argument("--bs", type=int, default=32)
    t.add_argument("--lr", type=float, default=1e-3)
    t.set_defaults(func=cmd_train)
    s = sub.add_parser("summary")
    s.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    s.set_defaults(func=cmd_summary)
    o = sub.add_parser("onnx")
    o.add_argument("run")
    o.set_defaults(func=cmd_onnx)
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
