# sonar-cnn

Reimplementation of MCNN (Çimen & Bolat, "MCNN: A Customized CNN Model for Marine
Debris Classification Based on Sonar Images") on the Valdenegro-Toro watertank FLS
dataset (`watertank-cropped`, 2,364 images, 10 classes), plus a variable-size CNN
and probes for leakage through image size.

## Setup
The dataset goes in `data/raw/` (git-ignored): a sparse clone of
`mvaldenegro/marine-debris-fls-datasets`, folder `md_fls_dataset/data/watertank-cropped`.

## Running
    ./run_all.sh            # all experiments -> results/*.json, results/run_all.log
    uv run pytest           # VarCNN batch == single-image equivalence test

## Models
- `MCNN`: the paper's Table 2, input zero-padded to 305x305, ~2.8M params.
- `VarCNN`: fully convolutional with masked BN and masked global mean+max pooling.
  It takes images at their original size. Batches are padded only so they can be
  stacked, and the padding is masked out after every layer, so outputs are
  identical to running each image alone. ~185k params.

## Results (stratified random split, 3 seeds, 30 epochs, AdamW)
| Experiment | Input | Accuracy |
|---|---|---|
| Random forest on (h, w) only | crop size, no pixels | 98.17 ± 0.70 (80/20) |
| MCNN | padded 305x305 | 99.72 ± 0.26 (80/20), 99.86 ± 0.20 (90/10) |
| MCNN | padded silhouette (content replaced by 1s) | 98.31 ± 0.35 |
| VarCNN | native size | 99.86 ± 0.10 |
| VarCNN | native size, content replaced by 1s | 97.04 ± 1.20 |
