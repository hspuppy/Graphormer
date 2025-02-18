python evaluate.py \
    --user-dir ../../graphormer \
    --num-workers 8 \
    --ddp-backend=legacy_ddp \
    --dataset-name pcqm4mv2 \
    --dataset-source ogb \
    --task graph_prediction \
    --criterion l1_loss \
    --arch graphormer_base \
    --num-classes 1 \
    --batch-size 64 \
    --pretrained-model-name pcqm4mv2_graphormer_base \
    --load-pretrained-model-output-layer \
    --split valid \
    --seed 1
