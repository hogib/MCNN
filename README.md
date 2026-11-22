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

## Scene-grouped evaluation on full frames (`sonar_cnn.scene`)

Ported from [hogib/sonar_edge](https://github.com/hogib/sonar_edge), where these results were
first produced. Instead of `watertank-cropped`, the crops are cut from the 1,868 full frames of
`watertank-segmentation` (every annotated box of the ten debris classes, 2,826 crops, zero-padded
to 305x305), so they can be assigned to the scene-grouped folds of the detector study. Add the
frames to the sparse dataset clone with
`git -C data/raw/marine-debris-fls-datasets sparse-checkout add md_fls_dataset/data/watertank-segmentation`
(commit `4a77998`).

- `splits/scene_folds_v1.json`: frames grouped into scenes reconstructed from the box layouts and
  assigned to five folds (fold k test, fold k+1 validation, the rest training).
- `splits/random_folds_v1.json`: the same frames assigned to five folds of the same sizes at
  random (image level), for comparison; selected with `MCNN_SPLIT=splits/random_folds_v1.json`.
- Models: MCNN (copy of the model used in the scene-fold runs; see `scene/mcnn_cv.py`),
  MobileNetV3-Small and ShuffleNetV2 x0.5 from scratch with a one-channel first convolution, all
  with MCNN's recipe (AdamW lr 1e-3, wd 1e-4, cosine, 30 epochs, batch 32, best validation epoch).

```
uv run python -m sonar_cnn.scene.classifiers train --arch mcnn --seeds 0 1 2      # also mnv3s, shufv2x05
uv run python -m sonar_cnn.scene.classifiers summary --arch mnv3s                 # -> results/scene_cv/
MCNN_SPLIT=splits/random_folds_v1.json uv run python -m sonar_cnn.scene.classifiers train --arch mcnn --prefix rcls
uv run python -m sonar_cnn.scene.classifiers summary --arch mcnn --prefix rcls
uv run python -m sonar_cnn.scene.material [--random] [--mcnn mcnn]                 # microplastic sources vs other debris (default MCNN runs: mcnn_t2)
```

Scene folds, pooled over the five test folds, mean ± sd over three seeds:

| Model | Params | MACs/crop | Accuracy | Macro-F1 |
|---|---|---|---|---|
| MCNN, design-table pooling (`mcnn_t2`, used in the paper) | 2.80 M | 236.9 M | 76.2 ± 2.0 | 69.6 ± 3.3 |
| MCNN, 2x2 pooling (`mcnn`) | 2.80 M | 234.7 M | 77.7 ± 0.5 | 71.4 ± 1.7 |
| MobileNetV3-Small | 1.53 M | 103.0 M | 71.9 ± 0.8 | 65.3 ± 1.8 |
| ShuffleNetV2 x0.5 | 0.35 M | 68.1 M | 66.7 ± 2.1 | 58.3 ± 2.3 |

Random image-level folds of the same sizes (same frames, recipe and evaluator; 90-96% of the
test frames of each fold share their scene with training frames):

| Model | Accuracy | Macro-F1 | Scene folds → random: accuracy gain |
|---|---|---|---|
| MCNN (`mcnn_t2`) | 95.9 ± 0.0 | 95.4 ± 0.2 | +19.7 |
| MCNN (`mcnn`) | 95.9 ± 0.2 | 95.3 ± 0.2 | +18.2 |
| MobileNetV3-Small | 93.7 ± 0.8 | 92.7 ± 1.0 | +21.8 |
| ShuffleNetV2 x0.5 | 88.7 ± 1.4 | 86.2 ± 1.5 | +22.0 |

Microplastic sources vs other debris, F1 of the microplastic group: MCNN 87.4 (scene) / 98.3
(random) for `mcnn`, 86.5 / 98.2 for `mcnn_t2`, MobileNetV3-Small 81.6 / 96.8, ShuffleNetV2 x0.5 75.5 / 92.9.

(`summary --arch mcnn` with the default prefix reads the runs as `runs/mcnn_cv_s{seed}_f{k}`, the
names they were produced under.)

## License

The code is released under the MIT License (see `LICENSE`). The sonar images are not included;
they belong to the Marine Debris FLS datasets
(<https://github.com/mvaldenegro/marine-debris-fls-datasets>, doi:10.5281/zenodo.15101686),
distributed under CC BY-NC-SA 4.0. The fold files in `splits/` list frame identifiers of that
dataset; cite the dataset when using them.
