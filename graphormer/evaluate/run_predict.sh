#!/bin/bash
python graphormer/evaluate/predict.py \
    --user-dir ./graphormer \
    --num-workers 8 \
    --ddp-backend=legacy_ddp \
    --task graph_prediction \
    --criterion l1_loss \
    --arch graphormer_base \
    --num-classes 1 \
    --batch-size 64 \
    --pretrained-model-name pcqm4mv2_graphormer_base \
    --load-pretrained-model-output-layer \
    --user-data-dir ./graphormer/data/customized_datasets \
    --dataset-name logS_dataset \
    --seed 1

    # --gpu \
#    --split valid \

