"""Microplastic sources against other debris (two-group task derived from the ten classes).

Materials from the dataset paper (Valdenegro-Toro et al. 2025, arXiv:2503.22880, App. B):
  microplastic sources (1): Shampoo-bottle, Tire, Drink-carton (assumed laminated carton)
  other debris (0):         Can, Chain, Hook, Propeller, Valve, Standing-bottle
  excluded:                 Bottle (glass and plastic bottles share one label), Wall
Merging classes removes confusions within a group, so these scores are not comparable with the
ten-class scores.

  uv run python -m sonar_cnn.scene.material                     # scene folds  -> results/scene_cv/cls_material_summary.json
  uv run python -m sonar_cnn.scene.material --random            # random split -> results/scene_cv/rcls_material_summary.json
MCNN runs: --mcnn mcnn_t2 (default; design-table pooling, runs cls_mcnn_t2_* / rcls_mcnn_t2_*)
or --mcnn mcnn (2x2-pooling variant, runs mcnn_cv_* / rcls_mcnn_*).
"""
import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
MICRO = ("Shampoo-bottle", "Tire", "Drink-carton")
OTHER = ("Can", "Chain", "Hook", "Propeller", "Valve", "Standing-bottle")
EXCLUDED = ("Bottle", "Wall")


def group_of(name):
    return 1 if name in MICRO else 0 if name in OTHER else None


def binary_scores(tg, pg):
    """tg: true group (0/1); pg: predicted group (0/1, -1 = error). -> metrics dict."""
    tg, pg = np.asarray(tg), np.asarray(pg)
    tp = np.sum((pg == 1) & (tg == 1))
    rec = tp / max(np.sum(tg == 1), 1)
    prec = tp / max(np.sum(pg == 1), 1)
    return {"acc": float(np.mean(pg == tg)), "micro_recall": float(rec), "micro_precision": float(prec),
            "micro_f1": float(2 * prec * rec / (prec + rec)) if prec + rec else 0.0,
            "other_recall": float(np.mean(pg[tg == 0] == 0)), "n": int(len(tg)), "n_micro": int(np.sum(tg == 1))}


def runs(prefix, seed):
    rs = [json.loads((ROOT / "runs" / f"{prefix}_s{seed}_f{k}" / "results.json").read_text()) for k in range(5)]
    classes = rs[0]["classes"]
    grp = np.array([-2 if group_of(c) is None else group_of(c) for c in classes])
    yt = np.concatenate([r["labels"] for r in rs])
    yp = np.concatenate([r["test_pred"] for r in rs])
    keep = grp[yt] >= 0
    p = grp[yp[keep]]
    return binary_scores(grp[yt[keep]], np.where(p < 0, -1, p))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--random", action="store_true", help="score the random-split runs (rcls_*)")
    ap.add_argument("--mcnn", choices=["mcnn_t2", "mcnn"], default="mcnn_t2", help="which MCNN runs to score")
    args = ap.parse_args()
    if args.random:
        m = f"rcls_{args.mcnn}"
        prefixes = (m, "rcls_mnv3s", "rcls_shufv2x05")
    else:
        m = "cls_mcnn_t2" if args.mcnn == "mcnn_t2" else "mcnn_cv"
        prefixes = (m, "cls_mnv3s", "cls_shufv2x05")
    out = {}
    for pre in prefixes:
        rows = [runs(pre, s) for s in range(3)]
        out[pre] = {k: (100 * np.mean([r[k] for r in rows]), 100 * np.std([r[k] for r in rows], ddof=1))
                    for k in ("acc", "micro_recall", "micro_precision", "micro_f1", "other_recall")}
        print(pre, {k: f"{m:.1f}±{s:.1f}" for k, (m, s) in out[pre].items()}, "n", rows[0]["n"])
    dest = ROOT / "results" / "scene_cv" / ("rcls_material_summary.json" if args.random else "cls_material_summary.json")
    dest.write_text(json.dumps(out, indent=1))
    print(f"wrote {dest.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
