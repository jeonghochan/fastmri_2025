#!/bin/bash
python train.py \
  -b 1 \
  -e 40 \
  -l 0.0001 \
  -r 10 \
  -n 'fastmri/second_cont/unet_2' \
  -t 'Data/train/' \
  -v 'Data/val/' \
  --chans 18 \
  --kspace-augment-config configs/kspace_mask_augment_2_exp.yaml \
  --resume test_Varnet/unet_2_to32/checkpoints/model.pt
  