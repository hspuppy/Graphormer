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
    --load-pretrained-model-output-layer \
    --user-data-dir ./graphormer/data/customized_datasets \
    --dataset-name ${NAME}_dataset \
    --save-dir ./ckpts/${NAME} \
    --split ${SPLIT} \
    --seed 1 $@

    #--split valid \
    # --gpu \
#    --split valid \
    # --pretrained-model-name pcqm4mv2_graphormer_base \

