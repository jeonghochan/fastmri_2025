from utils.data.load_data import create_data_loaders
from utils.augment.config import AugmentConfig
import argparse

# Create a simple args object
class Args:
    def __init__(self):
        self.data_path_train = '/Data/train/'
        self.batch_size = 1
        self.input_key = 'kspace'
        self.target_key = 'image_label'  
        self.max_key = 'max'

args = Args()
config = AugmentConfig.from_preset('minimal')

try:
    train_loader = create_data_loaders(args.data_path_train, args, augment_config=config)
    print('Data loader created successfully')
    
    # Get one batch
    for batch in train_loader:
        mask, kspace, target, maximum, fname, slice_num = batch
        print(f'K-space shape: {kspace.shape}')
        print(f'Target shape: {target.shape}')
        break
        
except Exception as e:
    print(f'Error: {e}')
    import traceback
    traceback.print_exc()