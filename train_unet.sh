python train.py \
  -b 1 \
  -e 40 \
  -l 0.001 \
  -r 10 \
  -n 'test_Varnet/unet' \
  -t '/storage/junsoo/train/' \
  -v '/storage/junsoo/val/' \
  --chans 18 \
  --kspace-augment-config configs/kspace_augment.yaml