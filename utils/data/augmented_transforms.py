"""
Augmented data transforms for FastMRI with MRAugment integration.
Provides backward-compatible data transforms with optional physics-aware augmentation.
"""
import numpy as np
import torch
import logging
from typing import Optional, Tuple, Any

from .transforms import DataTransform, to_tensor
from utils.augment.kspace_augment import KSpaceAugmentor
import yaml

logger = logging.getLogger(__name__)


class AugmentedDataTransform(DataTransform):
    """
    K-space domain augmented data transform.
    Maintains mathematical consistency between k-space and image domains.
    """
    
    def __init__(self, isforward: bool, max_key: str, augment_config_path: Optional[str] = None):
        """
        Initialize k-space augmented data transform.
        
        Args:
            isforward: Whether this is forward-only mode (no augmentation)
            max_key: Key for maximum value in attributes
            augment_config_path: Path to k-space augmentation YAML config
        """
        super().__init__(isforward, max_key)
        
        self.use_augmentation = augment_config_path is not None and not isforward
        self.training = True  # Training mode flag
        
        if self.use_augmentation:
            try:
                # Load k-space augmentation configuration
                with open(augment_config_path, 'r') as f:
                    config = yaml.safe_load(f)
                
                # Create KSpaceAugmentor with configuration including scheduling
                self.augmentor = KSpaceAugmentor(
                    prob_hflip=config.get('prob_hflip', 0.5),
                    prob_vflip=config.get('prob_vflip', 0.5),
                    prob_shift=config.get('prob_shift', 0.0),
                    max_shift_fraction=config.get('max_shift_fraction', 0.05),
                    seed=config.get('seed', None),
                    # Scheduling parameters
                    aug_strength=config.get('aug_strength', 1.0),
                    aug_schedule=config.get('aug_schedule', 'constant'),
                    aug_delay=config.get('aug_delay', 0),
                    max_epochs=config.get('max_epochs', 100),
                    aug_exp_decay=config.get('aug_exp_decay', 5.0),
                    # Mask function
                    base_prob_randmask=config.get('base_prob_randmask', 0.5),
                    base_prob_equimask=config.get('base_prob_equimask', 0.5),
                    base_prob_magicmask=config.get('base_prob_magicmask', 0.5),
                    base_prob_uniquemask=config.get('base_prob_uniquemask', 0.5),
                )
                logger.info(f"K-space augmentation enabled from config: {augment_config_path}")
                logger.info(f"Augmentation schedule: {config.get('aug_schedule', 'constant')} "
                          f"(strength={config.get('aug_strength', 1.0)}, "
                          f"delay={config.get('aug_delay', 0)} epochs)")
                
            except Exception as e:
                logger.error(f"Failed to initialize k-space augmentation: {e}")
                logger.warning("Continuing without augmentation")
                self.use_augmentation = False
                self.augmentor = None
        else:
            self.augmentor = None
    
    def set_training(self, training: bool):
        """Set training mode to enable/disable augmentation"""
        self.training = training
        if not training and self.use_augmentation:
            logger.debug("K-space augmentation disabled for evaluation")
    
    def set_epoch(self, epoch: int):
        """Set current epoch for augmentation scheduling"""
        self.current_epoch = epoch
        if self.use_augmentation and self.augmentor:
            self.augmentor.set_epoch(epoch)
            logger.debug(f"K-space augmentation epoch updated: {epoch}")
    
    def __call__(self, mask, input, target, attrs, fname, slice_num):
        """
        Apply data transform with optional augmentation.
        
        Args:
            mask: Sampling mask
            input: Input k-space data
            target: Target image (if available)
            attrs: Metadata attributes
            fname: Filename
            slice_num: Slice number
            
        Returns:
            Tuple of processed data: (mask, kspace, target, maximum, fname, slice_num)
        """
        # Apply standard transform first
        mask_tensor, kspace_tensor, target_tensor, maximum, fname, slice_num = super().__call__(
            mask, input, target, attrs, fname, slice_num
        )

        clean_kspace = input
        
        # Apply k-space augmentation if enabled and in training mode
        if self.use_augmentation and self.training and self.augmentor is not None:
            try:
                # Apply k-space domain augmentation
                
                kspace_tensor, target_tensor = self.augmentor.augment_kspace(
                    clean_kspace, kspace_tensor, target_tensor, fname, slice_num
                )
                logger.debug(f"Applied k-space augmentation to {fname}, slice {slice_num}")
                
            except Exception as e:
                logger.warning(f"K-space augmentation failed for {fname}, slice {slice_num}: {e}")
                logger.debug("Using non-augmented data")
                # Continue with non-augmented data
        
        return mask_tensor, kspace_tensor, target_tensor, maximum, fname, slice_num
    
    def get_augmentation_info(self) -> dict:
        """Get current augmentation status and parameters"""
        if not self.use_augmentation or self.augmentor is None:
            return {"enabled": False}
        
        # Get detailed scheduling info from augmentor
        schedule_info = self.augmentor.get_schedule_info()
        
        info = {
            "enabled": True,
            "training": self.training,
            "epoch": getattr(self, 'current_epoch', 0),
            "schedule_info": schedule_info
        }
        
        return info



def create_augmented_transform(isforward: bool, max_key: str, 
                             augment_config_path: Optional[str] = None) -> DataTransform:
    """
    Factory function to create appropriate data transform.
    
    Args:
        isforward: Forward mode flag
        max_key: Maximum value key
        augment_config_path: Path to k-space augmentation YAML config
        
    Returns:
        DataTransform instance (augmented or standard)
    """
    if augment_config_path is not None:
        return AugmentedDataTransform(isforward, max_key, augment_config_path)
    else:
        return DataTransform(isforward, max_key)