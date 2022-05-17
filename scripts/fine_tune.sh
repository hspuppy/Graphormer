#!/bin/bash
set -e
set -x

n_gpu=1
epoch=50
max_epoch=$((epoch + 1))
batch_size=64
tot_updates=$((${NSAMPLE}*3/4*epoch/batch_size/n_gpu))
warmup_updates=$((tot_updates*16/100))
echo warm up and total updates: $warmup_updates, $tot_updates

rm -f ./ckpts/${NAME}/*
echo Fine tuning ${NAME} ...

CUDA_VISIBLE_DEVICES=0 fairseq-train \
    --log-file ./run_${NAME}.log \
    --user-dir ./graphormer \
    --num-workers 8 \
    --ddp-backend=legacy_ddp \
    --user-data-dir ./graphormer/data/customized_datasets \
    --dataset-name ${NAME}_dataset \
    --task graph_prediction \
    --criterion l1_loss \
    --arch graphormer_base \
    --batch-size $batch_size \
    --max-epoch $max_epoch \
    --no-epoch-checkpoints \
    --num-classes 1 \
    --save-dir ./ckpts/${NAME} \
    --lr 2e-4 --lr-scheduler polynomial_decay --power 1 --warmup-updates $warmup_updates --total-num-update $tot_updates \
    --attention-dropout 0.1 --act-dropout 0.1 --dropout 0.0 \
    --optimizer adam --adam-betas '(0.9, 0.999)' --adam-eps 1e-8 --clip-norm 5.0 --weight-decay 0.0 \
    --pretrained-model-name pcqm4mv2_graphormer_base \
    --load-pretrained-model-output-layer \
    --seed 1

    #--patience 5 \
    #--lr 1e-4 \
    # --finetune-from-model ./ckpts/checkpoint_best_pcqm4mv2.pt \
    # --tensorboard-logdir
    # --lr-scheduler polynomial_decay --power 1 --warmup-updates $warmup_updates --total-num-update $tot_updates \
    # --lr 2e-4 --end-learning-rate 1e-5 \

