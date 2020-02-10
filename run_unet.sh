env CUDA_VISIBLE_DEVICES=3 python train_pipeline_unet.py \
        -sl 32 -sld 16 -df 2 -lr 3 -ct fifo -cr focal -l False \
        -n _hilounet_32_16_2_3_fifo_focal_wide_rtx > logs/hilounet_32_16_2_3_fifo_focal_wide_rtx.txt
