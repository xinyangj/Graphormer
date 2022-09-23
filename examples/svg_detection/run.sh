#!/usr/bin/env bash
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

n_gpu=1
epoch=500
max_epoch=$((epoch + 1))
batch_size=8
tot_updates=$((600*epoch/batch_size/n_gpu))
warmup_updates=$((tot_updates*16/100))

CUDA_VISIBLE_DEVICES=2 fairseq-train \
--user-dir ../../graphormer \
--num-workers 16 \
--ddp-backend=legacy_ddp \
--dataset-name svg_diagram \
--dataset-source svg \
--task svg_detection \
--criterion detr_loss \
--arch graphormer_slim \
--num-classes 21 \
--attention-dropout 0.1 --act-dropout 0.1 --dropout 0.0 \
--optimizer adam --adam-betas '(0.9, 0.999)' --adam-eps 1e-8 --clip-norm 5.0 --weight-decay 0.0 \
--lr-scheduler polynomial_decay --power 1 --total-num-update $tot_updates \
--lr 2e-4 --end-learning-rate 2e-5 \
--batch-size $batch_size \
--data-buffer-size 20 \
--encoder-layers 12 \
--encoder-embed-dim 768 \
--encoder-ffn-embed-dim 768 \
--encoder-attention-heads 32 \
--max-epoch $max_epoch \
--save-dir ./ckpts01 \
--seed 1 \
--pre-layernorm \
--disable-validation
#--warmup-updates $warmup_updates \
#--fp16 \
#--flag-m 3 \
#--flag-step-size 0.01 \
#--flag-mag 0 \
#--pretrained-model-name pcqm4mv1_graphormer_base_for_molhiv \
