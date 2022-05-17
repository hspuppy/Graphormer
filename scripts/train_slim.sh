NAME=LogS
MAX_EPOCHS=200
WUP=600  # 20000*16/100 ≈ 3200 tup * 16 / 100
TUP=4000  # 1280 / 64 * 1000 ≈ 20000  sample_num / batch * epoches / ngpus

CUDA_VISIBLE_DEVICES=0 fairseq-train \
    --log-file ./run_slim.log \
    --user-dir ./graphormer \
    --num-workers 8 \
    --ddp-backend=legacy_ddp \
    --user-data-dir ./graphormer/data/customized_datasets \
    --dataset-name ${NAME}_dataset \
    --task graph_prediction \
    --criterion l1_loss \
    --arch graphormer_slim \
    --num-classes 1 \
    --no-epoch-checkpoints \
    --attention-dropout 0.1 --act-dropout 0.1 --dropout 0.0 \
    --optimizer adam --adam-betas '(0.9, 0.999)' --adam-eps 1e-8 --clip-norm 5.0 --weight-decay 0.01 \
    --lr-scheduler polynomial_decay --power 1 --warmup-updates $WUP --total-num-update $TUP \
    --lr 2e-4 --end-learning-rate 1e-9 \
    --batch-size 64 \
    --data-buffer-size 20 \
    --encoder-layers 12 \
    --encoder-embed-dim 80 \
    --encoder-ffn-embed-dim 80 \
    --encoder-attention-heads 8 \
    --max-epoch $MAX_EPOCHS \
    --patience 10 \
    --save-dir ./ckpts_slim

    # --fp16 \
