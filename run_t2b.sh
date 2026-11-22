#!/bin/bash
# Remaining shortcut probes with MCNN in the design-table layout (mcnn_t2).
cd ~/Projeler/oguz/sonar_cnn
L=results/t2_logs
uv run sonar-cnn train --model mcnn_t2 --input randsize-pad --epochs 50 > $L/p2_randsize.log 2>&1
uv run sonar-cnn train --model mcnn_t2 --input randsize-pad-constant --epochs 50 > $L/p2_randsize_const.log 2>&1
uv run sonar-cnn train --split group --model mcnn_t2 --input randsize-pad --epochs 50 > $L/p2_randsize_group.log 2>&1
uv run sonar-cnn train --data data/dedup_t90 --model mcnn_t2 --input randsize-pad --epochs 50 > $L/p2_randsize_dedup.log 2>&1
echo "$(date "+%F %H:%M") randsize/dedup done" >> $L/run.log
touch $L/t2b.done
