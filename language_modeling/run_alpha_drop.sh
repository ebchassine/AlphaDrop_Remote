#!/bin/bash
cd BTD-Transformer

# Make sure PYTHONPATH includes parent for utils
export PYTHONPATH=..

CUDA_VISIBLE_DEVICES=6 python train_baseline.py \
    --cuda \
    --data ../penn \
    --dataset ptb \
    --n_layer 3 \
    --d_model 256 \
    --n_head 1 \
    --d_head 40 \
    --d_inner 2100 \
    --dropout 0.3 \
    --dropatt 0.0 \
    --lr 0.00025 \
    --max_step 40000 \
    --tgt_len 32 \
    --mem_len 0 \
    --eval_tgt_len 32 \
    --batch_size 120 \
    --block_length 4 \
    --gpu0_bsz 1 \
    --max_epoch 200 \
    --log-interval 200 \
    --esd-interval 200 \
    --pl-fitting median \
    --xmin-pos 2 \
    --filter-zeros False \
    --work_dir ./runs/alphadrop \
    --debug
