python train.py \
  -b 1 \
  -e 200 \
  -l 0.001 \
  -r 10 \
  -n 'test_Varnet/swinunet' \
  -t '../Data/train/' \
  -v '../Data/val/' \
  --use-transformer \
  --kspace-augment-config configs/kspace_mask_augment.yaml