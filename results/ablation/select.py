"""Apply the pre-registered selection rule of PLAN.md, then report test results for every variant.

  uv run python results/ablation/select.py
"""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
V = [("V0 baseline", "cls_mcnn_t2", "cls_mcnn_t2"), ("V1 +aug", "abl_aug_mcnn_t2", "abl_aug_mcnn_t2"),
     ("V2 +BN", "abl_mcnn_t2_bn", "abl_mcnn_t2_bn"), ("V3 +GAP head", "abl_mcnn_t2_gap", "abl_mcnn_t2_gap"),
     ("V4 aug+BN+GAP", "abl_aug_mcnn_t2_bn_gap", "abl_aug_mcnn_t2_bn_gap"),
     ("V5 V4 @160", "abl_aug_c160_mcnn_t2_bn_gap", "abl_aug_c160_mcnn_t2_bn_gap")]
out = []
for name, run, summ in V:
    rs = {(s, k): json.loads((ROOT / "runs" / f"{run}_s{s}_f{k}" / "results.json").read_text())
          for s in range(3) for k in range(5)}
    S = json.loads((ROOT / "results" / "scene_cv" / f"{summ}_summary.json").read_text())
    out.append({"variant": name, "runs": run, "val_score": float(np.mean([r["val_acc"] for r in rs.values()])),
                "test_acc": S["acc_mean"], "test_acc_sd": S["acc_sd"], "test_macro_f1": S["macro_f1_mean"],
                "params": S["params"], "macs": S["macs_per_crop"], "classes": S["classes"],
                "recall": S["recall_mean"], "pair_acc": {f"{s}_{k}": r["test_acc"] for (s, k), r in rs.items()},
                "best_epoch": float(np.mean([r["best_epoch"] for r in rs.values()]))})
best = max(o["val_score"] for o in out)
tied = [o for o in out if best - o["val_score"] <= 0.005]
win = min(tied, key=lambda o: (o["macs"], o["params"]))
base = out[0]
print(f"{'variant':16s} {'val':>6s} {'test acc':>12s} {'mF1':>6s} {'d acc vs V0 (15 pairs)':>24s} {'wins':>5s} "
      f"{'params':>10s} {'MACs':>7s} ep")
for o in out:
    d = np.array([o["pair_acc"][p] - base["pair_acc"][p] for p in base["pair_acc"]]) * 100
    o["paired_vs_v0"] = {"mean": float(d.mean()), "sd": float(d.std(ddof=1)) if o is not base else 0.0,
                         "wins": int((d > 0).sum())}
    tag = " <- selected" if o is win else (" (tied)" if o in tied else "")
    print(f"{o['variant']:16s} {100*o['val_score']:6.2f} {100*o['test_acc']:6.2f}±{100*o['test_acc_sd']:4.2f} "
          f"{100*o['test_macro_f1']:6.2f} {d.mean():+8.2f} ± {o['paired_vs_v0']['sd']:5.2f}        "
          f"{o['paired_vs_v0']['wins']:3d}/15 {o['params']:>10,} {o['macs']/1e6:6.1f}M {o['best_epoch']:4.1f}{tag}")
print("\nper-class recall (test):")
print(f"{'class':16s}" + "".join(f"{o['variant'][:2]:>7s}" for o in out))
for i, c in enumerate(base["classes"]):
    print(f"{c:16s}" + "".join(f"{100*o['recall'][i]:7.1f}" for o in out))
for o in out:
    o.pop("pair_acc")
(ROOT / "results" / "ablation" / "selection.json").write_text(json.dumps(
    {"rule": "PLAN.md", "best_val": best, "tied": [o["variant"] for o in tied], "selected": win["variant"],
     "variants": out}, indent=1))
