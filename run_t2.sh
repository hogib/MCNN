#!/bin/bash
# MCNN with the pooling of the design table (models.MCNNTable2, arch/model "mcnn_t2").
cd ~/Projeler/oguz/sonar_cnn
L=results/t2_logs
uv run sonar-cnn train --model mcnn_t2 --input pad --train-frac 0.8 > $L/p1_tr80.log 2>&1
uv run sonar-cnn train --model mcnn_t2 --input pad --train-frac 0.9 > $L/p1_tr90.log 2>&1
uv run sonar-cnn train --model mcnn_t2 --input silhouette --train-frac 0.8 > $L/p2_silhouette.log 2>&1
uv run sonar-cnn train --model mcnn_t2 --input pad --train-frac 0.8 --split group > $L/p2_group.log 2>&1
echo "$(date "+%F %H:%M") p1/p2 done" >> $L/run.log
uv run python -m sonar_cnn.scene.classifiers train --arch mcnn_t2 --seeds 0 1 2 > $L/scene.log 2>&1
MCNN_SPLIT=splits/random_folds_v1.json uv run python -m sonar_cnn.scene.classifiers train --arch mcnn_t2 --prefix rcls --seeds 0 1 2 > $L/random.log 2>&1
echo "$(date "+%F %H:%M") scene/random done" >> $L/run.log
touch $L/t2.done
