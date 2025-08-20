#!/bin/bash

# Test k-space augmentation verification
# This script runs visual verification of the augmentation system

echo "Running k-space augmentation verification..."

# Create verification output directory
mkdir -p verification_output

# Run verification on training data (adjust path as needed)
python verify_augmentation.py \
    --data-path /Data/train/ \
    --config configs/kspace_augment.yaml \
    --output-dir verification_output \
    --num-samples 5 \
    --seed 42

echo ""
echo "Verification complete!"
echo "Check the verification_output/ directory for:"
echo "  - Visual comparison plots (verification_*.png)"  
echo "  - Detailed text report (verification_report.txt)"
echo ""
echo "Key things to verify:"
echo "  1. Mathematical consistency: IFFT(augmented_kspace) ≈ augmented_target"
echo "  2. Visual coherence: Augmented images look realistic"
echo "  3. Transform correctness: Applied transforms match expectations"