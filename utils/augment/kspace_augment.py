# """
# K-space domain augmentation for MRI reconstruction.

# This module implements augmentation operations directly in k-space domain
# to maintain mathematical consistency between k-space and image domains.
# """
####
# import torch
# import numpy as np
# from typing import Tuple, Optional
# import logging
# from math import exp

# logger = logging.getLogger(__name__)

# class KSpaceAugmentor:
#     """
#     K-space domain augmentation that maintains mathematical consistency.
    
#     Implements:
#     - Horizontal flip in k-space domain
#     - Vertical flip in k-space domain  
#     - Plane movement (circular shift) in k-space domain
#     """
    
#     def __init__(self, 
#                  prob_hflip: float = 0.5,
#                  prob_vflip: float = 0.5,
#                  prob_shift: float = 0.5,
#                  max_shift_fraction: float = 0.1,
#                  seed: Optional[int] = None,
#                  # Scheduling parameters
#                  aug_strength: float = 1.0,
#                  aug_schedule: str = 'constant',
#                  aug_delay: int = 0,
#                  max_epochs: int = 100,
#                  aug_exp_decay: float = 5.0):
#         """
#         Initialize k-space augmentor with scheduling support.
        
#         Args:
#             prob_hflip: Base probability of horizontal flip
#             prob_vflip: Base probability of vertical flip
#             prob_shift: Base probability of applying circular shift
#             max_shift_fraction: Maximum shift as fraction of image size
#             seed: Random seed for reproducibility
#             aug_strength: Maximum augmentation strength (0.0-1.0)
#             aug_schedule: Schedule type ('constant', 'ramp', 'exp')
#             aug_delay: Number of epochs to delay augmentation start
#             max_epochs: Total number of training epochs
#             aug_exp_decay: Exponential decay coefficient for 'exp' schedule
#         """
#         self.base_prob_hflip = prob_hflip
#         self.base_prob_vflip = prob_vflip
#         self.base_prob_shift = prob_shift
#         self.max_shift_fraction = max_shift_fraction
        
#         # Scheduling parameters
#         self.aug_strength = aug_strength
#         self.aug_schedule = aug_schedule
#         self.aug_delay = aug_delay
#         self.max_epochs = max_epochs
#         self.aug_exp_decay = aug_exp_decay
#         self.current_epoch = 0
#         self.current_strength = 0.0
        
#         # Initialize random number generator
#         self.rng = np.random.RandomState(seed)
        
#         # Update strength for initial epoch
#         self._update_strength()
        
#     def augment_kspace(self, kspace_tensor: torch.Tensor, 
#                       target_tensor: torch.Tensor,
#                       fname: str, slice_num: int) -> Tuple[torch.Tensor, torch.Tensor]:
#         """
#         Apply k-space domain augmentation to k-space data and corresponding target.
        
#         Args:
#             kspace_tensor: K-space data in FastMRI format [..., H, W, 2]
#             target_tensor: Target image [H, W]
#             fname: Filename for debugging
#             slice_num: Slice number for debugging
            
#         Returns:
#             Tuple of (augmented_kspace, augmented_target)
#         """
#         # Convert to complex tensor for easier manipulation
#         kspace_complex = torch.complex(kspace_tensor[..., 0], kspace_tensor[..., 1])
        
#         # Track which transforms are applied for target consistency
#         transforms_applied = []
        
#         # Get current scheduled probabilities
#         current_prob_hflip = self.base_prob_hflip * self.current_strength
#         current_prob_vflip = self.base_prob_vflip * self.current_strength
#         current_prob_shift = self.base_prob_shift * self.current_strength
        
#         # Apply horizontal flip in k-space domain
#         if self.rng.rand() < current_prob_hflip:
#             kspace_complex = self._hflip_kspace(kspace_complex)
#             target_tensor = self._hflip_target(target_tensor)
#             transforms_applied.append('hflip')
            
#         # Apply vertical flip in k-space domain  
#         if self.rng.rand() < current_prob_vflip:
#             kspace_complex = self._vflip_kspace(kspace_complex)
#             target_tensor = self._vflip_target(target_tensor)
#             transforms_applied.append('vflip')
            
#         # Apply circular shift (plane movement) in k-space domain
#         if self.rng.rand() < current_prob_shift:
#             shift_h, shift_w = self._get_random_shift(kspace_complex.shape[-2:])
#             kspace_complex = self._shift_kspace(kspace_complex, shift_h, shift_w)
#             target_tensor = self._shift_target(target_tensor, shift_h, shift_w)
#             transforms_applied.append(f'shift_h{shift_h}_w{shift_w}')
        
#         # Convert back to FastMRI format
#         kspace_result = torch.stack([kspace_complex.real, kspace_complex.imag], dim=-1)
        
#         # Debug log
#         if transforms_applied:
#             logger.debug(f"Applied k-space transforms to {fname}, slice {slice_num}: {transforms_applied} "
#                         f"(strength={self.current_strength:.3f}, epoch={self.current_epoch})")
        
#         return kspace_result, target_tensor
    
#     def _hflip_kspace(self, kspace: torch.Tensor) -> torch.Tensor:
#         """
#         Horizontal flip in k-space domain.
        
#         Mathematical relationship:
#         If I(x,y) -> I(-x,y), then K(kx,ky) -> K(-kx,ky)*
#         Where * denotes complex conjugate.
#         """
#         # Flip along the kx dimension and take complex conjugate
#         return torch.flip(kspace, dims=[-2]).conj()
    
#     def _vflip_kspace(self, kspace: torch.Tensor) -> torch.Tensor:
#         """
#         Vertical flip in k-space domain.
        
#         Mathematical relationship:
#         If I(x,y) -> I(x,-y), then K(kx,ky) -> K(kx,-ky)*
#         Where * denotes complex conjugate.
#         """
#         # Flip along the ky dimension and take complex conjugate
#         return torch.flip(kspace, dims=[-1]).conj()
    
#     def _shift_kspace(self, kspace: torch.Tensor, shift_h: int, shift_w: int) -> torch.Tensor:
#         """
#         Circular shift in k-space domain.
        
#         Mathematical relationship:
#         If I(x,y) -> I(x-dx,y-dy), then K(kx,ky) -> K(kx,ky) * exp(-2πi(kx*dx + ky*dy))
        
#         For discrete case with circular shift, we use torch.roll.
#         """
#         # Apply circular shift in k-space domain
#         if shift_h != 0:
#             kspace = torch.roll(kspace, shifts=shift_h, dims=-2)
#         if shift_w != 0:
#             kspace = torch.roll(kspace, shifts=shift_w, dims=-1)
#         return kspace
    
#     def _hflip_target(self, target: torch.Tensor) -> torch.Tensor:
#         """Horizontal flip in target image domain."""
#         return torch.flip(target, dims=[-2])
    
#     def _vflip_target(self, target: torch.Tensor) -> torch.Tensor:
#         """Vertical flip in target image domain."""
#         return torch.flip(target, dims=[-1])
    
#     def _shift_target(self, target: torch.Tensor, shift_h: int, shift_w: int) -> torch.Tensor:
#         """Circular shift in target image domain."""
#         if shift_h != 0:
#             target = torch.roll(target, shifts=shift_h, dims=-2)
#         if shift_w != 0:
#             target = torch.roll(target, shifts=shift_w, dims=-1)
#         return target
    
#     def _get_random_shift(self, shape: Tuple[int, int]) -> Tuple[int, int]:
#         """Generate random shift amounts."""
#         h, w = shape
#         max_shift_h = int(h * self.max_shift_fraction)
#         max_shift_w = int(w * self.max_shift_fraction)
        
#         shift_h = self.rng.randint(-max_shift_h, max_shift_h + 1)
#         shift_w = self.rng.randint(-max_shift_w, max_shift_w + 1)
        
#         return shift_h, shift_w

#     def set_epoch(self, epoch: int):
#         """Update epoch and recalculate augmentation strength based on schedule."""
#         self.current_epoch = epoch
#         self._update_strength()
#         logger.debug(f"K-space augmentation epoch {epoch}: strength={self.current_strength:.3f}")
    
#     def _update_strength(self):
#         """Calculate current augmentation strength based on epoch and schedule."""
#         D = self.aug_delay
#         T = self.max_epochs
#         t = self.current_epoch
#         p_max = self.aug_strength
        
#         if t < D:
#             # Delay period - no augmentation
#             self.current_strength = 0.0
#         else:
#             if self.aug_schedule == 'constant':
#                 # Constant strength after delay
#                 self.current_strength = p_max
#             elif self.aug_schedule == 'ramp':
#                 # Linear ramp from delay epoch to max epochs
#                 self.current_strength = (t - D) / (T - D) * p_max
#             elif self.aug_schedule == 'exp':
#                 # Exponential ramp-up
#                 c = self.aug_exp_decay / (T - D)  # Decay coefficient
#                 self.current_strength = p_max / (1 - exp(-(T - D) * c)) * (1 - exp(-(t - D) * c))
#             else:
#                 logger.warning(f"Unknown schedule '{self.aug_schedule}', using constant")
#                 self.current_strength = p_max
    
#     def get_current_probabilities(self) -> dict:
#         """Get current effective probabilities after scheduling."""
#         return {
#             'hflip': self.base_prob_hflip * self.current_strength,
#             'vflip': self.base_prob_vflip * self.current_strength,
#             'shift': self.base_prob_shift * self.current_strength,
#             'strength': self.current_strength,
#             'epoch': self.current_epoch
#         }
    
#     def get_schedule_info(self) -> dict:
#         """Get scheduling configuration and current state."""
#         return {
#             'schedule_type': self.aug_schedule,
#             'max_strength': self.aug_strength,
#             'delay_epochs': self.aug_delay,
#             'max_epochs': self.max_epochs,
#             'exp_decay': self.aug_exp_decay,
#             'current_epoch': self.current_epoch,
#             'current_strength': self.current_strength,
#             'base_probabilities': {
#                 'hflip': self.base_prob_hflip,
#                 'vflip': self.base_prob_vflip,
#                 'shift': self.base_prob_shift
#             },
#             'current_probabilities': self.get_current_probabilities()
#         }

"""
K-space domain augmentation for MRI reconstruction.

This module implements augmentation operations directly in k-space domain
to maintain mathematical consistency between k-space and image domains.
"""

import torch
import numpy as np
from typing import Tuple, Optional
import logging
from math import exp
import random

# k-space masking utilities
from utils.augment.subsample import RandomMaskFunc, EquispacedMaskFractionFunc, MagicMaskFunc, UniqueMaskFunc

logger = logging.getLogger(__name__)

class KSpaceAugmentor:
    """
    K-space domain augmentation that maintains mathematical consistency.
    
    Implements:
    - Horizontal flip in k-space domain
    - Vertical flip in k-space domain  
    - Plane movement (circular shift) in k-space domain(removed for now)
    """
    
    def __init__(self, 
                 prob_hflip: float = 0.5,
                 prob_vflip: float = 0.5,
                 prob_shift: float = 0.5,
                 max_shift_fraction: float = 0.1,
                 seed: Optional[int] = None,
                 # Scheduling parameters
                 aug_strength: float = 1.0,
                 aug_schedule: str = 'constant',
                 aug_delay: int = 0,
                 max_epochs: int = 100,
                 aug_exp_decay: float = 5.0,
                 #mask generation parameters
                 base_prob_randmask: float = 0.5,
                 base_prob_equimask: float = 0.5,
                 base_prob_magicmask: float = 0.5,
                 base_prob_uniquemask: float = 0.5,
                 aug_mask = None                 
                 ):
        """
        Initialize k-space augmentor with scheduling support.
        
        Args:
            prob_hflip: Base probability of horizontal flip
            prob_vflip: Base probability of vertical flip
            prob_shift: Base probability of applying circular shift
            max_shift_fraction: Maximum shift as fraction of image size
            seed: Random seed for reproducibility
            aug_strength: Maximum augmentation strength (0.0-1.0)
            aug_schedule: Schedule type ('constant', 'ramp', 'exp')
            aug_delay: Number of epochs to delay augmentation start
            max_epochs: Total number of training epochs
            aug_exp_decay: Exponential decay coefficient for 'exp' schedule
        """
        self.base_prob_hflip = prob_hflip
        self.base_prob_vflip = prob_vflip
        self.base_prob_shift = prob_shift
        self.max_shift_fraction = max_shift_fraction
        #mask initialization
        self.base_prob_randmask = base_prob_randmask
        self.base_prob_equimask = base_prob_equimask
        self.base_prob_magic = base_prob_magicmask
        self.base_prob_uniquemask = base_prob_uniquemask
        self.aug_mask = aug_mask

        # Initialize k-space mask functions
        self.rand_mask_func = RandomMaskFunc(center_fractions=[0.04,0.08],
                                        accelerations=[4,8],
                                        allow_any_combination=True)
        self.equispaced_mask_func = EquispacedMaskFractionFunc(center_fractions=[0.04,0.08],
                                                                accelerations=[4,8],
                                                                allow_any_combination=True)
        self.magic_mask_func = MagicMaskFunc(center_fractions=[0.04,0.08],
                                             accelerations=[4,8], allow_any_combination=True)   
        
        #if center_fraction and acceleration이 changes, mask is changed. 
        self.unique_mask_func = UniqueMaskFunc(center_fractions=[0.04,0.08],
                                        accelerations=[9,10],
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
        Apply k-space domain augmentation to k-space data and corresponding target.
        
        Args:
            kspace_tensor: K-space data in FastMRI format [..., H, W, 2]
            target_tensor: Target image [H, W]
            fname: Filename for debugging
            slice_num: Slice number for debugging
            
        Returns:
            Tuple of (augmented_kspace, augmented_target)
        """
        # Convert to complex tensor for easier manipulation
        kspace_complex = torch.complex(kspace_tensor[..., 0], kspace_tensor[..., 1])

        clean_kspace_tensor = torch.from_numpy(clean_kspace).to(kspace_tensor.device)
        clean_kspace_complex = torch.stack((clean_kspace_tensor.real, clean_kspace_tensor.imag), dim=-1)

        # Track which transforms are applied for target consistency
        transforms_applied = []
        
        # Get current scheduled probabilities
        current_prob_hflip = self.base_prob_hflip * self.current_strength
        current_prob_vflip = self.base_prob_vflip * self.current_strength
        # current_prob_shift = self.base_prob_shift * self.current_strength
        current_prob_randmask = self.base_prob_randmask * self.current_strength
        current_prob_equimask = self.base_prob_equimask * self.current_strength
        current_prob_magic = self.base_prob_magic * self.current_strength
        current_prob_uniquemask = self.base_prob_uniquemask * self.current_strength

        # Apply horizontal flip in k-space domain
        if self.rng.rand() < current_prob_hflip:
            kspace_complex = self._hflip_kspace(kspace_complex)
            target_tensor = self._hflip_target(target_tensor)
            transforms_applied.append('hflip')
            self.aug_mask = torch.zeros_like(target_tensor)
            
        # Apply vertical flip in k-space domain  
        if self.rng.rand() < current_prob_vflip:
            kspace_complex = self._vflip_kspace(kspace_complex)
            target_tensor = self._vflip_target(target_tensor)
            transforms_applied.append('vflip')
            self.aug_mask = torch.zeros_like(target_tensor)
            
        # Apply circular shift (plane movement) in k-space domain
        # if self.rng.rand() < current_prob_shift:
        #     shift_h, shift_w = self._get_random_shift(kspace_complex.shape[-2:])
        #     kspace_complex = self._shift_kspace(kspace_complex, shift_h, shift_w)
        #     target_tensor = self._shift_target(target_tensor, shift_h, shift_w)
        #     transforms_applied.append(f'shift_h{shift_h}_w{shift_w}')

        # Apply new mask augmentation

        if self.rng.rand() < current_prob_randmask:
            # Apply new mask augmentation (always applied, no probability)
            # Generate mask from subsample.py
            H, W = clean_kspace_complex.shape[-2:]   # H=768, W=396
            print(H, W)
            
            mask, _ = self.equispaced_mask_func((1, W, 1), seed=None)  # (1, W, 1)
            mask = mask.to(clean_kspace_complex.device)
            # print("raw mask:", mask.shape)                # (1, W, 1)
            
            mask = mask.transpose(-1, -2).expand(1, H, W)
            # print("after expand:", mask.shape)            # (1, H, W)
        
            mask = mask.to(clean_kspace_complex.device)

            # Apply mask augmentation        
            clean_kspace_complex = clean_kspace_complex * mask
            
            #save mask
            self.aug_mask = mask
            transforms_applied.append('randmask')
            # print("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
            kspace_result = torch.stack([clean_kspace_complex.real, clean_kspace_complex.imag], dim=-1)

        
        elif self.rng.rand() < current_prob_randmask + current_prob_equimask:
            H, W = clean_kspace_complex.shape[-2:]   # H=768, W=396
            mask, _ = self.equispaced_mask_func((1, W, 1), seed=None)  # (1, W, 1)
            mask = mask.to(clean_kspace_complex.device)
            # print("raw mask:", mask.shape)                # (1, W, 1)
            mask = mask.transpose(-1, -2).expand(1, H, W)
            # print("after expand:", mask.shape)            # (1, H, W)
            clean_kspace_complex = clean_kspace_complex * mask
            self.aug_mask = mask
            transforms_applied.append('equimask')
            # print("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
            kspace_result = torch.stack([clean_kspace_complex.real, clean_kspace_complex.imag], dim=-1)

        elif self.rng.rand() < current_prob_randmask + current_prob_equimask + current_prob_magic:
            H, W = clean_kspace_complex.shape[-2:]   # H=768, W=396
            mask, _ = self.equispaced_mask_func((1, W, 1), seed=None)  # (1, W, 1)
            mask = mask.to(clean_kspace_complex.device)
            # print("raw mask:", mask.shape)                # (1, W, 1)
            mask = mask.transpose(-1, -2).expand(1, H, W)
            # print("after expand:", mask.shape)            # (1, H, W)
            clean_kspace_complex = clean_kspace_complex* mask
            self.aug_mask = mask
            transforms_applied.append('magicmask')
            # print("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
            kspace_result = torch.stack([clean_kspace_complex.real, clean_kspace_complex.imag], dim=-1)

        elif self.rng.rand() < current_prob_randmask + current_prob_equimask + current_prob_magic + current_prob_uniquemask:
            H, W = clean_kspace_complex.shape[-2:]   # H=768, W=396
            mask, _ = self.unique_mask_func((1, W, 1), seed=None)  # (1, W, 1)
            mask = mask.to(clean_kspace_complex.device)
            # print("raw mask:", mask.shape)                # (1, W, 1)
            mask = mask.transpose(-1, -2).expand(1, H, W)
            # print("after expand:", mask.shape)            # (1, H, W)
            clean_kspace_complex = clean_kspace_complex* mask
            self.aug_mask = mask
            transforms_applied.append('uniquemask')
            # print("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
            kspace_result = torch.stack([clean_kspace_complex.real, clean_kspace_complex.imag], dim=-1)

        else:
            kspace_result = torch.stack([kspace_complex.real, kspace_complex.imag], dim=-1)

        
        # Debug log
        if transforms_applied:
            logger.debug(f"Applied k-space transforms to {fname}, slice {slice_num}: {transforms_applied} "
                        f"(strength={self.current_strength:.3f}, epoch={self.current_epoch})")
        
        #if you want to check the verify,
        # return kspace_result, target_tensor, self.aug_mask
    
        return kspace_result, target_tensor


        
    
    def _hflip_kspace(self, kspace: torch.Tensor) -> torch.Tensor:
        """
        Horizontal flip in k-space domain.
        
        Mathematical relationship:
        If I(x,y) -> I(-x,y), then K(kx,ky) -> K(-kx,ky)*
        Where * denotes complex conjugate.
        """
        # Flip along the kx dimension and take complex conjugate
        return torch.flip(kspace, dims=[-2]).conj()
    
    def _vflip_kspace(self, kspace: torch.Tensor) -> torch.Tensor:
        """
        Vertical flip in k-space domain.
        
        Mathematical relationship:
        If I(x,y) -> I(x,-y), then K(kx,ky) -> K(kx,-ky)*
        Where * denotes complex conjugate.
        """
        # Flip along the ky dimension and take complex conjugate
        return torch.flip(kspace, dims=[-1]).conj()
    
    # def _shift_kspace(self, kspace: torch.Tensor, shift_h: int, shift_w: int) -> torch.Tensor:
    #     """
    #     Circular shift in k-space domain.
        
    #     Mathematical relationship:
    #     If I(x,y) -> I(x-dx,y-dy), then K(kx,ky) -> K(kx,ky) * exp(-2πi(kx*dx + ky*dy))
        
    #     For discrete case with circular shift, we use torch.roll.
    #     """
    #     # Apply circular shift in k-space domain
    #     if shift_h != 0:
    #         kspace = torch.roll(kspace, shifts=shift_h, dims=-2)
    #     if shift_w != 0:
    #         kspace = torch.roll(kspace, shifts=shift_w, dims=-1)
    #     return kspace




    def _hflip_target(self, target: torch.Tensor) -> torch.Tensor:
        """Horizontal flip in target image domain."""
        return torch.flip(target, dims=[-2])
    
    def _vflip_target(self, target: torch.Tensor) -> torch.Tensor:
        """Vertical flip in target image domain."""
        return torch.flip(target, dims=[-1])
    
    # def _shift_target(self, target: torch.Tensor, shift_h: int, shift_w: int) -> torch.Tensor:
    #     """Circular shift in target image domain."""
    #     if shift_h != 0:
    #         target = torch.roll(target, shifts=shift_h, dims=-2)
    #     if shift_w != 0:
    #         target = torch.roll(target, shifts=shift_w, dims=-1)
    #     return target
    
    # not used
    def _get_random_shift(self, shape: Tuple[int, int]) -> Tuple[int, int]:
        """Generate random shift amounts."""
        h, w = shape
        max_shift_h = int(h * self.max_shift_fraction)
        max_shift_w = int(w * self.max_shift_fraction)
        
        shift_h = self.rng.randint(-max_shift_h, max_shift_h + 1)
        shift_w = self.rng.randint(-max_shift_w, max_shift_w + 1)
        
        return shift_h, shift_w

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
            # Delay period - no augmentation
            self.current_strength = 0.0
        else:
            if self.aug_schedule == 'constant':
                # Constant strength after delay
                self.current_strength = p_max
            elif self.aug_schedule == 'ramp':
                # Linear ramp from delay epoch to max epochs
                self.current_strength = (t - D) / (T - D) * p_max
            elif self.aug_schedule == 'exp':
                # Exponential ramp-up
                c = self.aug_exp_decay / (T - D)  # Decay coefficient
                self.current_strength = p_max / (1 - exp(-(T - D) * c)) * (1 - exp(-(t - D) * c))
            else:
                logger.warning(f"Unknown schedule '{self.aug_schedule}', using constant")
                self.current_strength = p_max
    
    def get_current_probabilities(self) -> dict:
        """Get current effective probabilities after scheduling."""
        return {
            'hflip': self.base_prob_hflip * self.current_strength,
            'vflip': self.base_prob_vflip * self.current_strength,
            # 'shift': self.base_prob_shift * self.current_strength,
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
                # 'shift': self.base_prob_shift
                'randmask': self.base_prob_randmask,
                'equimask': self.base_prob_equimask,
                'magicmask': self.base_prob_magic,
                'uniquemask': self.base_prob_uniquemask,
            },
            'current_probabilities': self.get_current_probabilities()
        }