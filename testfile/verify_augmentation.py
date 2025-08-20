#!/usr/bin/env python3
"""
Visual verification tool for k-space augmentation.
Shows before/after augmentation results to verify mathematical consistency.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import h5py
import argparse
from pathlib import Path
import sys
import os

# Add utils to path
if os.getcwd() + '/utils/model/' not in sys.path:
    sys.path.insert(1, os.getcwd() + '/utils/model/')

from utils.model.fastmri.fftc import ifft2c_new
from utils.augment.kspace_augment import KSpaceAugmentor
from utils.data.transforms import DataTransform

def load_sample_data(data_path: Path, max_samples: int = 5):
    """Load sample data for verification."""
    samples = []
    data_files = list(data_path.glob('*.h5'))[:max_samples]
    
    for file_path in data_files:
        with h5py.File(file_path, 'r') as f:
            # Get first slice
            kspace = f['kspace'][0]  # [coils, height, width]
            
            # Convert to tensor format
            kspace_tensor = torch.from_numpy(kspace).float()
            
            # Convert complex to real/imag format [coils, height, width, 2]
            kspace_fastmri = torch.stack([kspace_tensor.real, kspace_tensor.imag], dim=-1)
            
            # Create target by taking RSS of full reconstruction
            target = torch.sqrt(torch.sum(torch.abs(kspace_tensor) ** 2, dim=0))
            target = ifft2c_new(target.unsqueeze(0)).squeeze(0).abs()
            
            samples.append({
                'filename': file_path.name,
                'kspace': kspace_fastmri,
                'target': target,
                'attrs': {'max': target.max().item()}
            })
    
    return samples

def create_comparison_plot(original_kspace, augmented_kspace, original_target, augmented_target, 
                         transforms_applied, filename, save_path):
    """Create side-by-side comparison plot."""
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle(f'{filename}\nTransforms: {", ".join(transforms_applied)}', fontsize=14)
    
    # Original k-space magnitude (log scale)
    kspace_orig_mag = torch.sqrt(original_kspace[..., 0]**2 + original_kspace[..., 1]**2)
    axes[0, 0].imshow(torch.log(kspace_orig_mag[0] + 1e-8).numpy(), cmap='gray')
    axes[0, 0].set_title('Original K-space (log)')
    axes[0, 0].axis('off')
    
    # Augmented k-space magnitude (log scale)
    kspace_aug_mag = torch.sqrt(augmented_kspace[..., 0]**2 + augmented_kspace[..., 1]**2)
    axes[0, 1].imshow(torch.log(kspace_aug_mag[0] + 1e-8).numpy(), cmap='gray')
    axes[0, 1].set_title('Augmented K-space (log)')
    axes[0, 1].axis('off')
    
    # K-space difference
    kspace_diff = torch.abs(kspace_orig_mag[0] - kspace_aug_mag[0])
    axes[0, 2].imshow(kspace_diff.numpy(), cmap='hot')
    axes[0, 2].set_title(f'K-space Diff (max: {kspace_diff.max():.3f})')
    axes[0, 2].axis('off')
    
    # K-space phase difference
    orig_phase = torch.atan2(original_kspace[0, ..., 1], original_kspace[0, ..., 0])
    aug_phase = torch.atan2(augmented_kspace[0, ..., 1], augmented_kspace[0, ..., 0])
    phase_diff = torch.abs(orig_phase - aug_phase)
    axes[0, 3].imshow(phase_diff.numpy(), cmap='hsv')
    axes[0, 3].set_title(f'Phase Diff (max: {phase_diff.max():.3f})')
    axes[0, 3].axis('off')
    
    # Original target image
    axes[1, 0].imshow(original_target.numpy(), cmap='gray')
    axes[1, 0].set_title('Original Target')
    axes[1, 0].axis('off')
    
    # Augmented target image
    axes[1, 1].imshow(augmented_target.numpy(), cmap='gray')
    axes[1, 1].set_title('Augmented Target')
    axes[1, 1].axis('off')
    
    # Image difference
    img_diff = torch.abs(original_target - augmented_target)
    axes[1, 2].imshow(img_diff.numpy(), cmap='hot')
    axes[1, 2].set_title(f'Image Diff (max: {img_diff.max():.3f})')
    axes[1, 2].axis('off')
    
    # Reconstructed from augmented k-space
    reconstructed = ifft2c_new(torch.complex(augmented_kspace[0, ..., 0], augmented_kspace[0, ..., 1]).unsqueeze(0))
    reconstructed = reconstructed.squeeze(0).abs()
    recon_diff = torch.abs(reconstructed - augmented_target)
    axes[1, 3].imshow(recon_diff.numpy(), cmap='hot')
    axes[1, 3].set_title(f'Consistency Check (max: {recon_diff.max():.6f})')
    axes[1, 3].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return {
        'kspace_diff_max': kspace_diff.max().item(),
        'image_diff_max': img_diff.max().item(),
        'consistency_error': recon_diff.max().item(),
        'consistency_mean': recon_diff.mean().item()
    }

def verify_mathematical_consistency(kspace_tensor, target_tensor):
    """Verify that IFFT(kspace) matches target."""
    # Reconstruct image from k-space
    kspace_complex = torch.complex(kspace_tensor[0, ..., 0], kspace_tensor[0, ..., 1])
    reconstructed = ifft2c_new(kspace_complex.unsqueeze(0)).squeeze(0).abs()
    
    # Calculate difference
    diff = torch.abs(reconstructed - target_tensor)
    
    return {
        'max_error': diff.max().item(),
        'mean_error': diff.mean().item(),
        'std_error': diff.std().item(),
        'consistent': diff.max().item() < 1e-5
    }

def main():
    parser = argparse.ArgumentParser(description='Verify k-space augmentation visually')
    parser.add_argument('--data-path', type=Path, required=True,
                       help='Path to data directory with .h5 files')
    parser.add_argument('--config', type=str, default='configs/kspace_augment.yaml',
                       help='Path to augmentation config')
    parser.add_argument('--output-dir', type=Path, default='verification_output',
                       help='Directory to save verification plots')
    parser.add_argument('--num-samples', type=int, default=3,
                       help='Number of samples to verify')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Loading k-space augmentation verification tool...")
    
    # Create augmentor
    import yaml
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    augmentor = KSpaceAugmentor(
        prob_hflip=config.get('prob_hflip', 0.5),
        prob_vflip=config.get('prob_vflip', 0.5), 
        prob_shift=config.get('prob_shift', 0.3),
        max_shift_fraction=config.get('max_shift_fraction', 0.05),
        seed=args.seed
    )
    
    print(f"Augmentor config: hflip={augmentor.prob_hflip}, vflip={augmentor.prob_vflip}, "
          f"shift={augmentor.prob_shift}, max_shift={augmentor.max_shift_fraction}")
    
    # Load sample data
    print(f"Loading samples from {args.data_path}...")
    samples = load_sample_data(args.data_path, args.num_samples)
    print(f"Loaded {len(samples)} samples")
    
    # Verification results
    results = []
    
    for i, sample in enumerate(samples):
        print(f"\nProcessing sample {i+1}/{len(samples)}: {sample['filename']}")
        
        # Original data
        original_kspace = sample['kspace']
        original_target = sample['target']
        
        # Verify original consistency
        orig_consistency = verify_mathematical_consistency(original_kspace, original_target)
        print(f"Original consistency: max_error={orig_consistency['max_error']:.6f}, "
              f"consistent={orig_consistency['consistent']}")
        
        # Apply augmentation
        augmented_kspace, augmented_target = augmentor.augment_kspace(
            original_kspace, original_target, sample['filename'], 0
        )
        
        # Verify augmented consistency
        aug_consistency = verify_mathematical_consistency(augmented_kspace, augmented_target)
        print(f"Augmented consistency: max_error={aug_consistency['max_error']:.6f}, "
              f"consistent={aug_consistency['consistent']}")
        
        # Determine what transforms were applied by comparing
        transforms_applied = []
        
        # Check for horizontal flip
        if not torch.allclose(original_target, augmented_target, atol=1e-5):
            # Check if it matches horizontal flip
            if torch.allclose(torch.flip(original_target, dims=[-2]), augmented_target, atol=1e-5):
                transforms_applied.append('hflip')
            elif torch.allclose(torch.flip(original_target, dims=[-1]), augmented_target, atol=1e-5):
                transforms_applied.append('vflip')
            else:
                # Might be a combination or shift
                transforms_applied.append('complex_transform')
        
        if not transforms_applied:
            transforms_applied = ['none']
        
        # Create visualization
        save_path = args.output_dir / f"verification_{i+1:02d}_{sample['filename']}.png"
        comparison_stats = create_comparison_plot(
            original_kspace, augmented_kspace, original_target, augmented_target,
            transforms_applied, sample['filename'], save_path
        )
        
        result = {
            'filename': sample['filename'],
            'transforms': transforms_applied,
            'original_consistency': orig_consistency,
            'augmented_consistency': aug_consistency,
            'comparison_stats': comparison_stats
        }
        results.append(result)
        
        print(f"  Transforms applied: {', '.join(transforms_applied)}")
        print(f"  Consistency preserved: {aug_consistency['consistent']}")
        print(f"  Saved visualization: {save_path}")
    
    # Summary report
    print(f"\n{'='*60}")
    print("VERIFICATION SUMMARY")
    print(f"{'='*60}")
    
    consistent_count = sum(1 for r in results if r['augmented_consistency']['consistent'])
    print(f"Mathematical consistency: {consistent_count}/{len(results)} samples passed")
    
    max_consistency_error = max(r['augmented_consistency']['max_error'] for r in results)
    mean_consistency_error = np.mean([r['augmented_consistency']['mean_error'] for r in results])
    
    print(f"Max consistency error: {max_consistency_error:.8f}")
    print(f"Mean consistency error: {mean_consistency_error:.8f}")
    print(f"Consistency threshold: 1e-5")
    
    if consistent_count == len(results):
        print("✅ ALL SAMPLES PASSED - Mathematical consistency verified!")
    else:
        print("❌ SOME SAMPLES FAILED - Check individual results")
    
    # Save detailed report
    report_path = args.output_dir / "verification_report.txt"
    with open(report_path, 'w') as f:
        f.write("K-space Augmentation Verification Report\n")
        f.write("="*50 + "\n\n")
        f.write(f"Configuration: {args.config}\n")
        f.write(f"Samples processed: {len(results)}\n")
        f.write(f"Mathematically consistent: {consistent_count}/{len(results)}\n\n")
        
        for result in results:
            f.write(f"Sample: {result['filename']}\n")
            f.write(f"  Transforms: {', '.join(result['transforms'])}\n")
            f.write(f"  Original consistency error: {result['original_consistency']['max_error']:.8f}\n")
            f.write(f"  Augmented consistency error: {result['augmented_consistency']['max_error']:.8f}\n")
            f.write(f"  Consistent: {result['augmented_consistency']['consistent']}\n\n")
    
    print(f"\nDetailed report saved to: {report_path}")
    print(f"Visualization plots saved to: {args.output_dir}")

if __name__ == '__main__':
    main()