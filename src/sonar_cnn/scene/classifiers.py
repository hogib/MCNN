"""Small ImageNet-family classifiers on MCNN's task, same protocol as sonar_cnn.scene.mcnn_cv.

Crops, padding (305x305, zero-padded, grey levels / 255), folds (test k, val k+1),
recipe (AdamW lr 1e-3 wd 1e-4, cosine over all steps, 30 epochs, batch 32, best-val
epoch) and scoring (pooled accuracy, macro-F1, per-class recall) are those of MCNN in
mcnn_cv.py, so the three networks differ only in architecture. Trained from scratch
like MCNN (main setting); the first convolution takes one grey channel.

  uv run --extra baselines python -m sonar_cnn.scene.classifiers train --arch mnv3s --seeds 0 1 2
  uv run --extra baselines python -m sonar_cnn.scene.classifiers summary --arch mnv3s
  uv run --extra baselines python -m sonar_cnn.scene.classifiers cost      # params, MACs
  MCNN_SPLIT=splits/random_folds_v1.json uv run --extra baselines python -m \
      sonar_cnn.scene.classifiers train --arch mcnn --prefix rcls   # random image-level split
"""

import argparse
import json
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .mcnn_cv import MCNN, PAD, ROOT, object_crops, pad, predict, scores
from .macs import count_macs, count_params


def make(arch, n_classes=10, canvas=PAD):
    import torchvision

    if arch == "mnv3s":
        m = torchvision.models.mobilenet_v3_small(weights=None, num_classes=n_classes)
        c = m.features[0][0]
        m.features[0][0] = nn.Conv2d(1, c.out_channels, c.kernel_size, c.stride, c.padding, bias=False)
    elif arch == "shufv2x05":
        m = torchvision.models.shufflenet_v2_x0_5(weights=None, num_classes=n_classes)
        c = m.conv1[0]
        m.conv1[0] = nn.Conv2d(1, c.out_channels, c.kernel_size, c.stride, c.padding, bias=False)
    elif arch == "mcnn":
        m = MCNN(n_classes)
    elif arch == "mcnn_t2":
        from ..models import MCNNTable2
        m = MCNNTable2(n_classes)
    elif arch.startswith("mcnn_t2_"):  # ablation: mcnn_t2_bn, mcnn_t2_gap, mcnn_t2_bn_gap
        from ..models import MCNNVariant
        opts = set(arch[len("mcnn_t2_"):].split("_"))
        assert opts <= {"bn", "gap"}, arch
        m = MCNNVariant(n_classes, bn="bn" in opts, gap="gap" in opts, canvas=canvas)
    else:
        raise ValueError(arch)
    return m


def augment(x, max_shift=16):
    """Training augmentation on a padded batch (B, 1, H, W) in [0, 1]: horizontal flip (p=0.5),
    shift of the object inside the canvas by up to max_shift pixels (limited to the empty margin, so
    the object is never cut), gain x U(0.8, 1.25) and multiplicative speckle noise (sd 0.1)."""
    B = x.shape[0]
    flip = torch.rand(B, device=x.device) < 0.5
    x = torch.where(flip[:, None, None, None], x.flip(-1), x)
    out = torch.zeros_like(x)
    for j in range(B):
        nz = x[j, 0] > 0
        rows, cols = nz.any(1).nonzero(), nz.any(0).nonzero()
        if len(rows) == 0:
            out[j] = x[j]
            continue
        r0, r1, c0, c1 = rows[0].item(), rows[-1].item(), cols[0].item(), cols[-1].item()
        H, W = x.shape[-2:]
        dy = int(torch.randint(-min(max_shift, r0), min(max_shift, H - 1 - r1) + 1, (1,)))
        dx = int(torch.randint(-min(max_shift, c0), min(max_shift, W - 1 - c1) + 1, (1,)))
        out[j] = torch.roll(x[j], (dy, dx), (-2, -1))
    gain = torch.empty(B, 1, 1, 1, device=x.device).uniform_(0.8, 1.25)
    out = out * gain * (1 + 0.1 * torch.randn_like(out))
    return out.clamp_(0, 1)


def pad_canvas(crops, canvas):
    """Zero-pad each crop, centred, to canvas x canvas; crops larger than the canvas are shrunk
    (aspect ratio kept) to fit. canvas=305 reproduces mcnn_cv.pad exactly."""
    from PIL import Image

    out = np.zeros((len(crops), 1, canvas, canvas), np.float32)
    for i, c in enumerate(crops):
        h, w = c.shape
        if max(h, w) > canvas:
            s = canvas / max(h, w)
            h, w = max(1, round(h * s)), max(1, round(w * s))
            c = np.asarray(Image.fromarray(c).resize((w, h), Image.BILINEAR))
        t, l = (canvas - h) // 2, (canvas - w) // 2
        out[i, 0, t : t + h, l : l + w] = c / 255.0
    return torch.from_numpy(out)


def train_fold(arch, X, y, folds, k, seed, args, device):
    torch.manual_seed(seed)
    np.random.seed(seed)
    v = (k + 1) % 5
    tr = np.where((folds != k) & (folds != v))[0]
    va, te = np.where(folds == v)[0], np.where(folds == k)[0]
    model = make(arch, int(y.max()) + 1, getattr(args, "canvas", PAD)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    steps = args.epochs * ((len(tr) + args.bs - 1) // args.bs)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    g = torch.Generator().manual_seed(seed)
    yt = torch.from_numpy(y)
    best, log, t0 = (-1.0, 0, None), [], time.time()
    for ep in range(args.epochs):
        model.train()
        for b in torch.randperm(len(tr), generator=g).split(args.bs):
            if len(b) < 2:  # BatchNorm needs > 1 sample per batch in training mode
                continue
            i = tr[b.numpy()]
            xb = X[i].to(device)
            if getattr(args, "aug", False):
                xb = augment(xb)
            loss = F.cross_entropy(model(xb), yt[i].to(device))
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
    return model, {"arch": arch, "seed": seed, "test_fold": k, "val_fold": v, "best_epoch": best[1], "val_acc": best[0],
                   "test_idx": te.tolist(), "test_pred": pt.tolist(), "test_acc": float((pt == y[te]).mean()),
                   "n_train": len(tr), "train_s": round(time.time() - t0, 1), "log": log}


def cmd_train(args):
    torch.set_num_threads(args.threads)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    crops, y, folds, ids, classes = object_crops()
    X = pad(crops) if args.canvas == PAD else pad_canvas(crops, args.canvas)
    print(f"{args.arch}: {len(crops)} crops, per fold {np.bincount(folds).tolist()}", flush=True)
    for seed in args.seeds:
        for k in range(5):
            out = ROOT / "runs" / f"{args.prefix}_{args.arch}_s{seed}_f{k}"
            if (out / "results.json").exists() and not args.force:
                print(f"skip {out.name} (done)", flush=True)
                continue
            out.mkdir(parents=True, exist_ok=True)
            model, r = train_fold(args.arch, X, y, folds, k, seed, args, device)
            r |= {"classes": classes, "ids": [ids[i] for i in r["test_idx"]], "labels": y[r["test_idx"]].tolist()}
            torch.save({"model": model.state_dict(), "classes": classes, "arch": args.arch}, out / "best.pt")
            (out / "results.json").write_text(json.dumps(r))
            print(f"{args.arch} seed {seed} fold {k}: test acc {100 * r['test_acc']:.2f} (best epoch {r['best_epoch']}, "
                  f"val {100 * r['val_acc']:.2f}, {r['train_s']} s)", flush=True)


def summarize(arch, seeds, run_fmt):
    per_seed = []
    for seed in seeds:
        rs = [json.loads((ROOT / "runs" / run_fmt.format(arch=arch, seed=seed, k=k) / "results.json").read_text())
              for k in range(5)]
        classes = rs[0]["classes"]
        yt = np.concatenate([r["labels"] for r in rs])
        yp = np.concatenate([r["test_pred"] for r in rs])
        acc, mf1, rec = scores(yt, yp, len(classes))
        per_seed.append({"seed": seed, "acc": acc, "macro_f1": mf1, "recall": [float(x) for x in rec],
                         "acc_per_fold": [r["test_acc"] for r in rs], "best_epoch": [r["best_epoch"] for r in rs]})
    a = np.array([p["acc"] for p in per_seed])
    f = np.array([p["macro_f1"] for p in per_seed])
    R = np.array([p["recall"] for p in per_seed])
    sd = lambda x: float(x.std(ddof=1)) if len(x) > 1 else None
    return {"arch": arch, "per_seed": per_seed, "classes": classes, "acc_mean": float(a.mean()), "acc_sd": sd(a),
            "macro_f1_mean": float(f.mean()), "macro_f1_sd": sd(f), "recall_mean": R.mean(0).tolist(),
            "acc_per_fold_mean": np.mean([p["acc_per_fold"] for p in per_seed], 0).tolist()}


def cmd_summary(args):
    fmt = ("mcnn_cv_s{seed}_f{k}" if args.arch == "mcnn" and args.prefix == "cls"
           else args.prefix + "_{arch}_s{seed}_f{k}")
    s = summarize(args.arch, args.seeds, fmt)
    m = make(args.arch, canvas=args.canvas)
    s |= {"params": count_params(m), "macs_per_crop": count_macs(m, torch.zeros(1, 1, args.canvas, args.canvas))}
    print(f"{args.arch}: acc {100 * s['acc_mean']:.2f} +- {100 * (s['acc_sd'] or 0):.2f}  macro-F1 "
          f"{100 * s['macro_f1_mean']:.2f} +- {100 * (s['macro_f1_sd'] or 0):.2f}  per fold "
          + " ".join(f"{100 * x:.1f}" for x in s["acc_per_fold_mean"])
          + f"  params {s['params']:,}  MACs {s['macs_per_crop'] / 1e6:.1f} M")
    for c, r in zip(s["classes"], s["recall_mean"]):
        print(f"   {c:16s} recall {100 * r:5.1f}")
    dest = ROOT / "results" / "scene_cv" / f"{args.prefix}_{args.arch}_summary.json"
    dest.write_text(json.dumps(s, indent=1))
    print(f"wrote {dest.relative_to(ROOT)}")


def cmd_cost(args):
    for arch in ("mcnn", "mnv3s", "shufv2x05"):
        m = make(arch)
        print(f"{arch:10s} params {count_params(m):>10,}  MACs/crop {count_macs(m, torch.zeros(1, 1, PAD, PAD)) / 1e6:8.1f} M")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(required=True)
    t = sub.add_parser("train")
    t.add_argument("--arch", required=True, choices=["mnv3s", "shufv2x05", "mcnn", "mcnn_t2", "mcnn_t2_bn", "mcnn_t2_gap", "mcnn_t2_bn_gap"])
    t.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    t.add_argument("--epochs", type=int, default=30)
    t.add_argument("--bs", type=int, default=32)
    t.add_argument("--lr", type=float, default=1e-3)
    t.add_argument("--threads", type=int, default=4)
    t.add_argument("--force", action="store_true")
    t.add_argument("--aug", action="store_true", help="training augmentation (see augment)")
    t.add_argument("--canvas", type=int, default=PAD, help="padded input size (305 = MCNN)")
    t.add_argument("--prefix", default="cls", help="run-name prefix, e.g. rcls for the random split (MCNN_SPLIT)")
    t.set_defaults(func=cmd_train)
    s = sub.add_parser("summary")
    s.add_argument("--arch", required=True, choices=["mnv3s", "shufv2x05", "mcnn", "mcnn_t2", "mcnn_t2_bn", "mcnn_t2_gap", "mcnn_t2_bn_gap"])
    s.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    s.add_argument("--prefix", default="cls")
    s.add_argument("--canvas", type=int, default=PAD)
    s.set_defaults(func=cmd_summary)
    c = sub.add_parser("cost")
    c.set_defaults(func=cmd_cost)
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
