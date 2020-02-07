env CUDA_VISIBLE_DEVICES=3 python train_pipeline_unet.py \
        -sl 16 -sld 8 -df 2 -lr 3 -ct fifo -cr focal -l False \
        -n _hilounet_16_8_2_3_fifo_focal_new > logs/hilounet_16_8_2_3_fifo_focal_new_selu.txt
