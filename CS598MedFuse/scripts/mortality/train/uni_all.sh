CUDA_VISIBLE_DEVICES=0 CUDA_LAUNCH_BLOCKING=1 python "/content/drive/MyDrive/CS 598 DFLH - Final Project - Medfuse/03medfuse/fusion_main.py" \
--dim 256 --dropout 0.3 --layers 2 \
--vision-backbone resnet34 \
--mode train \
--epochs 25 --batch_size 16 --lr 0.00053985 \
--vision_num_classes 1 --num_classes 1 \
--data_pairs partial_ehr \
--fusion_type uni_ehr --task in-hospital-mortality \
--save_dir "/content/drive/MyDrive/CS 598 DFLH - Final Project - Medfuse/03medfuse/checkpoints/mortality/uni_ehr_full" \
--labels_set mortality
