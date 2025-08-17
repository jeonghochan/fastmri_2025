python train.py \
  -b 1 \
  -e 80 \
  -l 0.001 \
  -r 10 \
  -n 'test_Varnet/unet' \
  -t '/storage/junsoo/train/' \
  -v '/storage/junsoo/val/' \
  --chans 16 \
  --kspace-augment-config configs/kspace_augment.yaml