#!/usr/bin/fish
env CUDA_VISIBLE_DEVICES=2,3  python train_pipeline.py \
    -sl 32 -sld 16 -df 4 -np 10 -lr 4 -ct fifo -cr bce -l False \
    -n _hilo_32_16_4_10_4_fifo_bce > logs/hilo_32_16_4_10_4_fifo_bce.txt

env CUDA_VISIBLE_DEVICES=2,3  python train_pipeline.py \
    -sl 32 -sld 16 -df 4 -np 10 -lr 3 -ct fifo -cr bce -l False \
    -n _hilo_32_16_4_10_3_fifo_bce > logs/hilo_32_16_4_10_3_fifo_bce.txt
