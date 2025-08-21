import torch
import numpy as np
from typing import Tuple, Optional
import logging
from math import exp
import random

# k-space masking utilities
from utils.augment.subsample import RandomMaskFunc, EquispacedMaskFractionFunc, MagicMaskFunc, UniqueMaskFunc

logger = logging.getLogger(__name__)
"""
K-space domain augmentation for MRI reconstruction.

This module implements augmentation operations directly in k-space domain
to maintain mathematical consistency between k-space and image domains.
"""

class KSpaceAugmentor:
    """
    K-space domain augmentation that maintains mathematical consistency.
    
    Implements:
    - Horizontal flip in k-space domain
    - Vertical flip in k-space domain
    - Custom k-space mask augmentation
    """
    
    def __init__(self, 
                 prob_hflip: float = 0.5,
                 prob_vflip: float = 0.5,
                 seed: Optional[int] = None,
                 # Scheduling parameters
                 aug_strength: float = 1.0,
                 aug_schedule: str = 'constant',
                 aug_delay: int = 0,
                 max_epochs: int = 100,
                 aug_exp_decay: float = 5.0,
                 # Mask generation parameters
                 base_prob_randmask: float = 0.25,
                 base_prob_equimask: float = 0.25,
                 base_prob_magicmask: float = 0.25,
                 base_prob_uniquemask: float = 0.25
                 ):
        """
        Initialize k-space augmentor with scheduling support.
        """
        self.base_prob_hflip = prob_hflip
        self.base_prob_vflip = prob_vflip
        # Mask probabilities
        self.base_prob_randmask = base_prob_randmask
        self.base_prob_equimask = base_prob_equimask
        self.base_prob_magic = base_prob_magicmask
        self.base_prob_uniquemask = base_prob_uniquemask

        # Initialize k-space mask functions
        self.rand_mask_func = RandomMaskFunc(center_fractions=[0.04, 0.08],
                                             accelerations=[4, 8],
                                             allow_any_combination=True)
        self.equispaced_mask_func = EquispacedMaskFractionFunc(center_fractions=[0.04, 0.08],
                                                                 accelerations=[4, 8],
                                                                 allow_any_combination=True)
        self.magic_mask_func = MagicMaskFunc(center_fractions=[0.04, 0.08],
                                               accelerations=[4, 8], allow_any_combination=True)
        self.unique_mask_func = UniqueMaskFunc(center_fractions=[0.04, 0.08],
                                                 accelerations=[9, 10],
                                                 allow_any_combination=True)

        # Scheduling parameters
        self.aug_strength = aug_strength
        self.aug_schedule = aug_schedule
        self.aug_delay = aug_delay
        self.max_epochs = max_epochs
        self.aug_exp_decay = aug_exp_decay
        self.current_epoch = 0
        self.current_strength = 0.0
        
        # Initialize random number generator
        self.rng = np.random.RandomState(seed)
        
        # Update strength for initial epoch
        self._update_strength()

    def augment_kspace(self, clean_kspace: np.ndarray,
                      kspace_tensor: torch.Tensor,
                      target_tensor: torch.Tensor,
                      fname: str, slice_num: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply k-space domain augmentation with corrected logic.
        The correct order is: Flips -> Masking.
        """
        # Convert inputs to complex tensors for easier manipulation
        clean_kspace_complex = torch.from_numpy(clean_kspace).to(kspace_tensor.device)
        original_kspace_complex = torch.complex(kspace_tensor[..., 0], kspace_tensor[..., 1])

        # Track which transforms are applied for debugging
        transforms_applied = []
        
        # Get current scheduled probabilities
        current_prob_hflip = self.base_prob_hflip * self.current_strength
        current_prob_vflip = self.base_prob_vflip * self.current_strength

        # ====================================================================
        # STEP 1: Apply flips FIRST for mathematical consistency.
        # Flips are applied to the target image, the fully-sampled k-space (for custom masks),
        # and the original undersampled k-space (if no custom mask is applied).
        # ====================================================================
        if self.rng.rand() < current_prob_hflip:
            clean_kspace_complex = self._hflip_kspace(clean_kspace_complex)
            original_kspace_complex = self._hflip_kspace(original_kspace_complex)
            target_tensor = self._hflip_target(target_tensor)
            transforms_applied.append('hflip')
            
        if self.rng.rand() < current_prob_vflip:
            clean_kspace_complex = self._vflip_kspace(clean_kspace_complex)
            original_kspace_complex = self._vflip_kspace(original_kspace_complex)
            target_tensor = self._vflip_target(target_tensor)
            transforms_applied.append('vflip')

        # ====================================================================
        # STEP 2: Apply a new custom mask OR use the original mask.
        # ====================================================================
        current_prob_randmask = self.base_prob_randmask * self.current_strength
        current_prob_equimask = self.base_prob_equimask * self.current_strength
        current_prob_magic = self.base_prob_magic * self.current_strength
        current_prob_uniquemask = self.base_prob_uniquemask * self.current_strength
        
        total_custom_mask_prob = current_prob_randmask + current_prob_equimask + current_prob_magic + current_prob_uniquemask
        
        mask_rand = self.rng.rand()
        
        if mask_rand < total_custom_mask_prob:
            # Apply a NEW custom mask to the (potentially flipped) fully-sampled k-space
            H, W = clean_kspace_complex.shape[-2:]
            mask_seed = self.rng.randint(2**32 - 1)
            
            # --- BUG FIX: Call the CORRECT mask function for each case ---
            if mask_rand < current_prob_randmask:
                mask, _ = self.rand_mask_func((1, W, 1), seed=mask_seed)
                transforms_applied.append('randmask')
            elif mask_rand < current_prob_randmask + current_prob_equimask:
                mask, _ = self.equispaced_mask_func((1, W, 1), seed=mask_seed)
                transforms_applied.append('equimask')
            elif mask_rand < current_prob_randmask + current_prob_equimask + current_prob_magic:
                mask, _ = self.magic_mask_func((1, W, 1), seed=mask_seed)
                transforms_applied.append('magicmask')
            else: # The last case is uniquemask
                mask, _ = self.unique_mask_func((1, W, 1), seed=mask_seed)
                transforms_applied.append('uniquemask')

            mask = mask.to(clean_kspace_complex.device)
            mask = mask.transpose(-1, -2).expand_as(clean_kspace_complex)
        
            final_kspace_complex = clean_kspace_complex * mask
        else:
            # No custom mask applied, use the original (and potentially flipped) k-space
            final_kspace_complex = original_kspace_complex
            transforms_applied.append('original_mask')

        # Convert the final complex k-space back to the [..., H, W, 2] format
        kspace_result = torch.stack([final_kspace_complex.real, final_kspace_complex.imag], dim=-1)

        # Debug log
        if transforms_applied:
            logger.debug(f"Applied k-space transforms to {fname}, slice {slice_num}: {transforms_applied} "
                        f"(strength={self.current_strength:.3f}, epoch={self.current_epoch})")
    
        return kspace_result, target_tensor

    def _hflip_kspace(self, kspace: torch.Tensor) -> torch.Tensor:
        """Horizontal flip in k-space domain."""
        # Flip along the kx dimension (width, dim=-1) and take complex conjugate
        # Assuming kx corresponds to the last dimension
        return torch.flip(kspace, dims=[-1]).conj()
    
    def _vflip_kspace(self, kspace: torch.Tensor) -> torch.Tensor:
        """Vertical flip in k-space domain."""
        # Flip along the ky dimension (height, dim=-2) and take complex conjugate
        # Assuming ky corresponds to the second to last dimension
        return torch.flip(kspace, dims=[-2]).conj()

    def _hflip_target(self, target: torch.Tensor) -> torch.Tensor:
        """Horizontal flip in target image domain."""
        return torch.flip(target, dims=[-1])
    
    def _vflip_target(self, target: torch.Tensor) -> torch.Tensor:
        """Vertical flip in target image domain."""
        return torch.flip(target, dims=[-2])
    
    def set_epoch(self, epoch: int):
        """Update epoch and recalculate augmentation strength based on schedule."""
        self.current_epoch = epoch
        self._update_strength()
        logger.debug(f"K-space augmentation epoch {epoch}: strength={self.current_strength:.3f}")
    
    def _update_strength(self):
        """Calculate current augmentation strength based on epoch and schedule."""
        D = self.aug_delay
        T = self.max_epochs
        t = self.current_epoch
        p_max = self.aug_strength
        
        if t < D:
            self.current_strength = 0.0
        else:
            if self.aug_schedule == 'constant':
                self.current_strength = p_max
            elif self.aug_schedule == 'ramp':
                if T - D > 0:
                    self.current_strength = min((t - D) / (T - D), 1.0) * p_max
                else:
                    self.current_strength = p_max # Avoid division by zero
            elif self.aug_schedule == 'exp':
                if T - D > 0:
                    c = self.aug_exp_decay / (T - D)
                    self.current_strength = p_max * (1 - exp(-(t - D) * c)) / (1 - exp(-(T - D) * c))
                else:
                    self.current_strength = p_max # Avoid division by zero
            else:
                logger.warning(f"Unknown schedule '{self.aug_schedule}', using constant")
                self.current_strength = p_max
        # Ensure strength does not exceed max_strength
        self.current_strength = min(self.current_strength, self.aug_strength)

    def get_current_probabilities(self) -> dict:
        """Get current effective probabilities after scheduling."""
        return {
            'hflip': self.base_prob_hflip * self.current_strength,
            'vflip': self.base_prob_vflip * self.current_strength,
            'randmask' : self.base_prob_randmask * self.current_strength,
            'equimask': self.base_prob_equimask * self.current_strength,
            'magicmask': self.base_prob_magic * self.current_strength,
            'uniquemask': self.base_prob_uniquemask * self.current_strength,
            'strength': self.current_strength,
            'epoch': self.current_epoch,
        }
    
    def get_schedule_info(self) -> dict:
        """Get scheduling configuration and current state."""
        return {
            'schedule_type': self.aug_schedule,
            'max_strength': self.aug_strength,
            'delay_epochs': self.aug_delay,
            'max_epochs': self.max_epochs,
            'exp_decay': self.aug_exp_decay,
            'current_epoch': self.current_epoch,
            'current_strength': self.current_strength,
            'base_probabilities': {
                'hflip': self.base_prob_hflip,
                'vflip': self.base_prob_vflip,
                'randmask': self.base_prob_randmask,
                'equimask': self.base_prob_equimask,
                'magicmask': self.base_prob_magic,
                'uniquemask': self.base_prob_uniquemask,
            },
            'current_probabilities': self.get_current_probabilities()
        }