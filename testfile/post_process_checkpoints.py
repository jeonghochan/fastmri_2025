#!/usr/bin/env python3
"""
Post-process checkpoint files to keep only top 5 models by validation loss.
Reads validation loss from checkpoint metadata - no re-computation needed.
"""

import torch
from pathlib import Path
import argparse
import sys


def load_checkpoint_info(checkpoint_path):
    """Load validation loss and epoch from checkpoint metadata."""
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        
        epoch = checkpoint.get('epoch', 0)
        val_loss = checkpoint.get('best_val_loss', float('inf'))
        
        # Handle tensor values
        if hasattr(val_loss, 'item'):
            val_loss = val_loss.item()
        elif torch.is_tensor(val_loss):
            val_loss = float(val_loss)
            
        return {
            'path': checkpoint_path,
            'epoch': epoch,
            'val_loss': val_loss,
            'valid': True
        }
    except Exception as e:
        print(f"Warning: Could not read {checkpoint_path.name}: {e}")
        return {
            'path': checkpoint_path,
            'epoch': 0,
            'val_loss': float('inf'),
            'valid': False
        }


def post_process_checkpoints(checkpoint_dir, keep_count=5, dry_run=False):
    """
    Post-process checkpoints to keep only the best models.
    
    Args:
        checkpoint_dir: Path to checkpoint directory
        keep_count: Number of best models to keep
        dry_run: If True, show what would be done without actually doing it
    """
    checkpoint_dir = Path(checkpoint_dir)
    
    if not checkpoint_dir.exists():
        print(f"Error: Checkpoint directory {checkpoint_dir} does not exist!")
        return False
    
    print(f"Processing checkpoint directory: {checkpoint_dir}")
    print(f"Target: Keep top {keep_count} models by validation loss")
    print("-" * 60)
    
    # Find all epoch checkpoint files
    epoch_files = sorted(checkpoint_dir.glob("model_epoch_*.pt"))
    
    if not epoch_files:
        print("No epoch checkpoint files found (model_epoch_*.pt)")
        return False
    
    print(f"Found {len(epoch_files)} epoch checkpoint files")
    
    # Load checkpoint information
    print("\nReading checkpoint metadata...")
    model_info = []
    total_size = 0
    
    for checkpoint_path in epoch_files:
        info = load_checkpoint_info(checkpoint_path)
        if info['valid']:
            model_info.append(info)
            size_mb = checkpoint_path.stat().st_size / (1024**2)
            total_size += size_mb
            print(f"  Epoch {info['epoch']:2d}: val_loss = {info['val_loss']:.6f} ({size_mb:.1f} MB)")
    
    if not model_info:
        print("No valid checkpoint files found!")
        return False
    
    # Sort by validation loss (best first)
    model_info.sort(key=lambda x: x['val_loss'])
    
    # Determine which models to keep and remove
    models_to_keep = model_info[:keep_count]
    models_to_remove = model_info[keep_count:]
    
    print(f"\nTop {len(models_to_keep)} models by validation loss (TO KEEP):")
    keep_size = 0
    for i, info in enumerate(models_to_keep):
        size_mb = info['path'].stat().st_size / (1024**2)
        keep_size += size_mb
        print(f"  {i+1}. Epoch {info['epoch']:2d}: val_loss = {info['val_loss']:.6f} - {info['path'].name} ({size_mb:.1f} MB)")
    
    if models_to_remove:
        print(f"\nModels to remove ({len(models_to_remove)}):")
        remove_size = 0
        for info in models_to_remove:
            size_mb = info['path'].stat().st_size / (1024**2)
            remove_size += size_mb
            print(f"  Epoch {info['epoch']:2d}: val_loss = {info['val_loss']:.6f} - {info['path'].name} ({size_mb:.1f} MB)")
        
        print(f"\nStorage summary:")
        print(f"  Current total: {total_size:.1f} MB")
        print(f"  After cleanup: {keep_size:.1f} MB")
        print(f"  Space saved:   {remove_size:.1f} MB ({remove_size/total_size*100:.1f}%)")
        
        if dry_run:
            print(f"\n[DRY RUN] Would delete {len(models_to_remove)} files")
            print("Run without --dry-run to actually delete files")
        else:
            # Confirm deletion
            response = input(f"\nProceed with deleting {len(models_to_remove)} checkpoint files? (y/N): ")
            if response.lower() in ['y', 'yes']:
                deleted_count = 0
                for info in models_to_remove:
                    try:
                        info['path'].unlink()
                        deleted_count += 1
                        print(f"  Deleted: {info['path'].name}")
                    except Exception as e:
                        print(f"  Error deleting {info['path'].name}: {e}")
                
                print(f"\nCompleted: Deleted {deleted_count}/{len(models_to_remove)} files")
            else:
                print("Operation cancelled")
                return False
    else:
        print(f"\nAll {len(model_info)} models are within the top {keep_count}")
        print("No files need to be removed")
    
    # Check for other important files
    print(f"\nOther files in checkpoint directory:")
    other_files = ['model.pt', 'best_model.pt']
    for filename in other_files:
        filepath = checkpoint_dir / filename
        if filepath.exists():
            size_mb = filepath.stat().st_size / (1024**2)
            if filename == 'best_model.pt':
                # Get info from best model
                info = load_checkpoint_info(filepath)
                print(f"  {filename}: Epoch {info['epoch']}, val_loss = {info['val_loss']:.6f} ({size_mb:.1f} MB) [PRESERVED]")
            else:
                print(f"  {filename}: {size_mb:.1f} MB [PRESERVED]")
        else:
            print(f"  {filename}: NOT FOUND")
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description='Post-process FastMRI checkpoints to keep only top N models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python post_process_checkpoints.py /home/junsoo/result/test_Varnet/unet/checkpoints
  python post_process_checkpoints.py /home/junsoo/result/test_Varnet/unet/checkpoints --keep 3
  python post_process_checkpoints.py /home/junsoo/result/test_Varnet/unet/checkpoints --dry-run
        """
    )
    
    parser.add_argument('checkpoint_dir',
                       help='Path to checkpoint directory')
    parser.add_argument('--keep', type=int, default=5,
                       help='Number of best models to keep (default: 5)')
    parser.add_argument('--dry-run', action='store_true',
                       help='Show what would be done without actually deleting files')
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.keep < 1:
        print("Error: --keep must be at least 1")
        sys.exit(1)
    
    # Run post-processing
    success = post_process_checkpoints(args.checkpoint_dir, args.keep, args.dry_run)
    
    if not success:
        sys.exit(1)
    
    print("\nPost-processing completed successfully!")


if __name__ == '__main__':
    main()