#!/bin/bash
python train.py \
  -b 1 \
  -e 40 \
  -l 0.0001 \
  -r 10 \
  -n 'test_Varnet/first_train/swinunet_1_cont' \
  -t 'Data/train/' \
  -v 'Data/val/' \
  --use-transformer \
  --kspace-augment-config configs/kspace_mask_augment_2_exp.yaml \
  --resume test_Varnet/swinunet_1/checkpoints/best_model.pt