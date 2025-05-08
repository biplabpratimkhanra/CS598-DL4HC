CUDA_VISIBLE_DEVICES=0 CUDA_LAUNCH_BLOCKING=1 python "/content/drive/MyDrive/CS 598 DFLH - Final Project - Medfuse/03medfuse/fusion_main.py" --dim 256 \
--dropout 0.3 --mode train \
--epochs 10 --pretrained \
--vision-backbone resnet34 --data_pairs radiology \
--batch_size 16 --align 0.0 --labels_set radiology --save_dir "/content/drive/MyDrive/CS 598 DFLH - Final Project - Medfuse/03medfuse/checkpoints/cxr_rad_full" \
--fusion_type uni_cxr --layers 2 --vision_num_classes 14 \
