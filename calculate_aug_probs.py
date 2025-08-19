#!/usr/bin/env python3
"""Calculate exact augmentation probabilities for 20 epochs"""

import math

# Configuration values
base_prob_hflip = 0.4
base_prob_vflip = 0.4
base_prob_randmask = 0.15
base_prob_equimask = 0.15
base_prob_magicmask = 0.2
base_prob_uniquemask = 0.2

aug_strength = 1.0
aug_schedule = 'exp'
aug_delay = 5
max_epochs = 40  # Original config
aug_exp_decay = 5.0

def calculate_strength(epoch, D, T, p_max, c):
    """Calculate augmentation strength for given epoch"""
    if epoch < D:
        return 0.0
    elif epoch >= T:
        return p_max
    else:
        # Exponential ramp-up formula
        decay_coeff = c / (T - D)
        numerator = p_max * (1 - math.exp(-(epoch - D) * decay_coeff))
        denominator = 1 - math.exp(-(T - D) * decay_coeff)
        return numerator / denominator

print("Exact Augmentation Probabilities for 20 Epochs")
print("=" * 60)
print("Configuration:")
print(f"- Delay: {aug_delay} epochs")
print(f"- Schedule: {aug_schedule}")
print(f"- Decay coefficient: {aug_exp_decay}")
print(f"- Max epochs (config): {max_epochs}")
print()

# Calculate for 20 epochs
training_epochs = 20
print(f"Training for {training_epochs} epochs:")
print()
print("Epoch | Strength | H-Flip | V-Flip | RandMask | EquiMask | MagicMask | UniqueMask")
print("-" * 80)

for epoch in range(1, training_epochs + 1):
    strength = calculate_strength(epoch, aug_delay, max_epochs, aug_strength, aug_exp_decay)
    
    prob_hflip = base_prob_hflip * strength
    prob_vflip = base_prob_vflip * strength
    prob_randmask = base_prob_randmask * strength
    prob_equimask = base_prob_equimask * strength
    prob_magicmask = base_prob_magicmask * strength
    prob_uniquemask = base_prob_uniquemask * strength
    
    print(f"{epoch:5d} | {strength:8.4f} | {prob_hflip:6.4f} | {prob_vflip:6.4f} | "
          f"{prob_randmask:8.4f} | {prob_equimask:8.4f} | {prob_magicmask:9.4f} | {prob_uniquemask:10.4f}")