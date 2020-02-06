#env CUDA_VISIBLE_DEVICES=0  python train_pipeline.py \
#    -sl 48 -sld 24 -df 2 -np 10 -lr 4 -ct fifo -cr bce -l False \
#    -n _hilo_48_24_2_10_4_fifo_bce > logs/hilo_48_24_2_10_4_fifo_bce.txt
env CUDA_VISIBLE_DEVICES=0  python train_pipeline.py \
        -sl 16 -sld 8 -df 2 -np 10 -lr 3 -ct fifo -cr bce -l False \
        -n _hilo_16_8_2_10_3_fifo_bce > logs/hilo_16_8_2_10_3_fifo_bce.txt
