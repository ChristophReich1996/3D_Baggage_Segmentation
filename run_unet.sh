env CUDA_VISIBLE_DEVICES=0,2 python train_pipeline_unet.py \
        -sl 16 -sld 8 -df 2 -lr 3 -ct fifo -cr focal -l True \
        -n _hilounet_16_8_2_3_fifo_focal_new > logs/hilounet_16_8_2_3_fifo_focal_new.txt
