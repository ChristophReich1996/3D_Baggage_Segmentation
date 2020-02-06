#!/usr/bin/fish
env CUDA_VISIBLE_DEVICES=0,2  python test_pipeline.py \
    -sl 32 -sld 16 -df 4 -np 10 -lr 4 -cr bce -a test \
    -n _hilo_32_16_4_10_4_fifo_bce > logs/TEST_hilo_32_16_4_10_4_fifo_bce.txt
