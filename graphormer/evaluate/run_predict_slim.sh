#!/bin/bash
NAME=LogS

python graphormer/evaluate/predict.py \
    --user-dir ./graphormer \
    --num-workers 8 \
    --ddp-backend=legacy_ddp \
    --task graph_prediction \
    --criterion l1_loss \
    --arch graphormer_slim \
    --num-classes 1 \
    --batch-size 64 \
    --load-pretrained-model-output-layer \
    --user-data-dir ./graphormer/data/customized_datasets \
    --save-dir ./ckpts_slim \
    --dataset-name ${NAME}_dataset \
    --split ${SPLIT} \
    --seed 1 $@


