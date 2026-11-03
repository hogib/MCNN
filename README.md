# sonar-cnn

Reimplementation of MCNN (Çimen & Bolat, "MCNN: A Customized CNN Model for Marine
Debris Classification Based on Sonar Images") on the Valdenegro-Toro watertank FLS
dataset (`watertank-cropped`, 2,364 images, 10 classes), plus a variable-size CNN
and probes for leakage through image size.

## Setup
The dataset goes in `data/raw/` (git-ignored): a sparse clone of
`mvaldenegro/marine-debris-fls-datasets`, folder `md_fls_dataset/data/watertank-cropped`.

## Models
- `MCNN`: the paper's Table 2, input zero-padded to 305x305, ~2.8M params.
- `VarCNN`: fully convolutional with masked BN and masked global mean+max pooling.
  It takes images at their original size. Batches are padded only so they can be
  stacked, and the padding is masked out after every layer, so outputs are
  identical to running each image alone. ~185k params.

## Running
    ./run_all.sh          # round 1: size baseline, MCNN/VarCNN on padded and native inputs
    ./run_randsize.sh     # round 2: random resize so crop size carries no class info
    ./run_group.sh        # round 3: pseudo-sequence group split (no near-duplicates across split)
    uv run python -m sonar_cnn.dupes                  # near-duplicate analysis -> results/dupes.json
    uv run python -m sonar_cnn.dedup --threshold 0.9  # build data/dedup_t90/
    ./run_dedup.sh        # round 4: random-size models on data/dedup_t90
    uv run pytest         # VarCNN batch == single-image equivalence test

Input modes: `pad` (centered zero-pad to 305x305, as in the paper), `silhouette`
(padded, content replaced by 1s), `native` (original size), `randsize` (random
(h, w) in [48, 160] per axis, independent of class), `randsize-pad`, and
`-constant` probes (content replaced by 1s).

Splits: `random` (stratified 80/20, the paper's protocol) and `group` (whole
pseudo-sequences, i.e. single-linkage chains of 32x32 thumbnails with
correlation >= 0.85, kept on one side).

## Results
Stratified 80/20, 3 seeds, AdamW lr 1e-3, cosine schedule, 30 epochs (50 for
randsize), accuracy at the final epoch (no validation set). All runs regenerated
on 2026-11-03 (RTX 5090); `results/*.json` carry per-class precision/recall/F1
and confusion matrices, and `weights/` (untracked) holds one checkpoint per
experiment and seed.

| Model | Input | Random split | Group split | Dedup t=0.9, random split |
|---|---|---|---|---|
| Random forest on (h, w) | crop size only | 98.17 ± 0.70 | | |
| MCNN | pad | 99.79 ± 0.17 (90/10: 99.86 ± 0.20) | 96.61 ± 1.98 | |
| MCNN | silhouette (no content) | 98.24 ± 0.55 | | |
| MCNN | randsize-pad | 99.44 ± 0.53 | 91.42 ± 2.29 | 98.44 ± 0.72 |
| MCNN | randsize-pad-constant | 19.03 (chance) | | |
| VarCNN | native | 99.86 ± 0.10 | 96.08 ± 1.29 | |
| VarCNN | native-constant (no content) | 97.32 ± 0.95 | | |
| VarCNN | randsize | 99.93 ± 0.10 | 96.31 ± 0.87 | 99.86 ± 0.20 |
| VarCNN | randsize-constant | 19.03 (chance) | | |

Near-duplicates (`results/dupes.json`): 1-NN on 32x32 thumbnails gets 100.00% on
a random split (71% of test images have a training image with correlation > 0.9)
vs 92.0% on the group split. File numbers do not follow recording order.

Takeaways:
- The paper's ~99.9% reproduces, but crop size alone gives ~98%, and padding
  exposes it to the network.
- Random splits are dominated by near-duplicate frames. On the group split,
  MCNN without size information drops to ~90%, VarCNN holds ~96.6% with 15x
  fewer parameters.
- Greedy dedup at 0.9 does not remove the leak (still ~99% on random splits):
  frames just under the threshold are still from the same stretch. Use the
  group split for honest numbers.
