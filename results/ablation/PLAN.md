# MCNN ablation: plan fixed before any run (2026-11-22)

Protocol: scene-grouped 5-fold CV (splits/scene_folds_v1.json), fold k test, fold k+1 validation,
seeds 0 1 2, recipe of MCNN (AdamW lr 1e-3, wd 1e-4, cosine, 30 epochs, batch 32, best-validation
epoch). Code: sonar_cnn.scene.classifiers.

Variants (baseline = MCNNTable2, runs cls_mcnn_t2_*):
- V0 baseline                          --arch mcnn_t2                      (existing runs)
- V1 + augmentation                    --arch mcnn_t2 --aug
- V2 + BatchNorm after every conv      --arch mcnn_t2_bn
- V3 + global-average-pooling head     --arch mcnn_t2_gap
- V4 V1+V2+V3                          --arch mcnn_t2_bn_gap --aug
- V5 V4 on a 160x160 canvas            --arch mcnn_t2_bn_gap --aug --canvas 160

Augmentation (augment()): horizontal flip p=0.5; shift inside the canvas up to 16 px, limited to the
empty margin; gain U(0.8, 1.25); multiplicative speckle noise sd 0.1. Not tuned.

Selection rule (validation only; test folds are not looked at for the decision):
1. Score = mean over the 15 fold-seed runs of the best-epoch VALIDATION accuracy.
2. Winner = highest score; variants within 0.5 points of the best are tied, and among tied variants
   the one with the fewest MACs per crop wins (then fewest parameters).
3. Test accuracy (pooled over the five test folds, mean over seeds) is computed afterwards and
   reported for EVERY variant, not only the winner.
