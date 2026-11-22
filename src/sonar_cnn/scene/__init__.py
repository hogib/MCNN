"""MCNN and reference crop classifiers on crops cut from the full watertank-segmentation frames,
evaluated under scene-grouped 5-fold CV (splits/scene_folds_v1.json) or, for comparison, a random
image-level 5-fold split of the same frames (MCNN_SPLIT=splits/random_folds_v1.json).

Ported from hogib/sonar_edge (src/sonar_edge/{data,stats,split,mcnn_cv,material}.py and
baselines/{classifiers,macs}.py), where these results were first produced; the scene folds are
those of the sonar_edge detector paper."""
