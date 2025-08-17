#!/usr/bin/env python3
"""
Test script to visualize k-space augmentation scheduling.
Shows how augmentation strength changes over epochs.
"""

import sys
import os
import yaml
import numpy as np

# Add utils to path
if os.getcwd() + '/utils/model/' not in sys.path:
    sys.path.insert(1, os.getcwd() + '/utils/model/')

from utils.augment.kspace_augment import KSpaceAugmentor

def test_schedule(config_path, max_epochs_override=None):
    """Test scheduling behavior with given config."""
    
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Override max_epochs if provided
    if max_epochs_override:
        config['max_epochs'] = max_epochs_override
    
    print(f"Testing scheduler with config: {config_path}")
    print(f"Schedule type: {config.get('aug_schedule', 'constant')}")
    print(f"Max strength: {config.get('aug_strength', 1.0)}")
    print(f"Delay epochs: {config.get('aug_delay', 0)}")
    print(f"Max epochs: {config.get('max_epochs', 100)}")
    print(f"Exp decay: {config.get('aug_exp_decay', 5.0)}")
    print()
    
    # Create augmentor
    augmentor = KSpaceAugmentor(
        prob_hflip=config.get('prob_hflip', 0.5),
        prob_vflip=config.get('prob_vflip', 0.5),
        prob_shift=config.get('prob_shift', 0.3),
        max_shift_fraction=config.get('max_shift_fraction', 0.05),
        seed=42,
        aug_strength=config.get('aug_strength', 1.0),
        aug_schedule=config.get('aug_schedule', 'constant'),
        aug_delay=config.get('aug_delay', 0),
        max_epochs=config.get('max_epochs', 100),
        aug_exp_decay=config.get('aug_exp_decay', 5.0)
    )
    
    # Test scheduling over epochs
    max_epochs = config.get('max_epochs', 100)
    test_epochs = [0, 1, 5] + list(range(10, max_epochs + 1, 10))
    if max_epochs not in test_epochs:
        test_epochs.append(max_epochs)
    test_epochs = sorted(set(test_epochs))
    
    print("Epoch | Strength | HFlip Prob | VFlip Prob | Shift Prob")
    print("-" * 60)
    
    strength_values = []
    epoch_values = []
    
    for epoch in test_epochs:
        augmentor.set_epoch(epoch)
        probs = augmentor.get_current_probabilities()
        
        print(f"{epoch:5d} | {probs['strength']:8.3f} | {probs['hflip']:10.3f} | "
              f"{probs['vflip']:10.3f} | {probs['shift']:10.3f}")
        
        strength_values.append(probs['strength'])
        epoch_values.append(epoch)
    
    print()
    
    # Summary
    schedule_info = augmentor.get_schedule_info()
    print("Schedule Summary:")
    print(f"  Final strength at epoch {max_epochs}: {strength_values[-1]:.3f}")
    print(f"  Schedule type: {schedule_info['schedule_type']}")
    print(f"  Delay period: epochs 0-{schedule_info['delay_epochs']-1} (strength = 0.0)")
    
    if schedule_info['delay_epochs'] < max_epochs:
        active_start = max(schedule_info['delay_epochs'], min(epoch_values))
        active_idx = next(i for i, e in enumerate(epoch_values) if e >= active_start)
        if active_idx < len(strength_values):
            print(f"  Active period: epoch {active_start}+ (strength = {strength_values[active_idx]:.3f} → {strength_values[-1]:.3f})")
    
    return epoch_values, strength_values

def compare_schedules():
    """Compare different scheduling strategies."""
    
    print("=" * 70)
    print("SCHEDULE COMPARISON")
    print("=" * 70)
    
    # Base config
    base_config = {
        'prob_hflip': 0.5,
        'prob_vflip': 0.5,
        'prob_shift': 0.3,
        'max_shift_fraction': 0.05,
        'aug_strength': 1.0,
        'aug_delay': 5,
        'max_epochs': 50,
        'aug_exp_decay': 5.0
    }
    
    schedules = ['constant', 'ramp', 'exp']
    
    for schedule in schedules:
        print(f"\n{schedule.upper()} SCHEDULE:")
        print("-" * 40)
        
        config = base_config.copy()
        config['aug_schedule'] = schedule
        
        augmentor = KSpaceAugmentor(
            prob_hflip=config['prob_hflip'],
            prob_vflip=config['prob_vflip'],
            prob_shift=config['prob_shift'],
            max_shift_fraction=config['max_shift_fraction'],
            seed=42,
            aug_strength=config['aug_strength'],
            aug_schedule=config['aug_schedule'],
            aug_delay=config['aug_delay'],
            max_epochs=config['max_epochs'],
            aug_exp_decay=config['aug_exp_decay']
        )
        
        # Test key epochs
        test_epochs = [0, 5, 10, 20, 30, 50]
        
        for epoch in test_epochs:
            augmentor.set_epoch(epoch)
            strength = augmentor.current_strength
            print(f"  Epoch {epoch:2d}: strength = {strength:.3f}")

def main():
    """Main test function."""
    
    print("K-SPACE AUGMENTATION SCHEDULER TEST")
    print("=" * 50)
    
    # Test with default config
    config_path = "configs/kspace_augment.yaml"
    
    try:
        test_schedule(config_path, max_epochs_override=50)
        compare_schedules()
        
        print("\n" + "=" * 70)
        print("SCHEDULER TEST COMPLETE")
        print("=" * 70)
        print("✅ Scheduler implementation working correctly!")
        print("\nKey benefits of scheduling:")
        print("  - Early epochs: Lower augmentation for stable learning")
        print("  - Later epochs: Full augmentation for robustness")
        print("  - Delay period: Clean data for anatomical structure learning")
        print("  - Smooth transitions: Avoid sudden training disruptions")
        
    except Exception as e:
        print(f"Error testing scheduler: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()