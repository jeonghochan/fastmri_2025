"""
Configuration management for MRAugment integration with FastMRI.
Handles YAML configuration loading, validation, and parameter management.
"""
import os
import yaml
from typing import Dict, List, Optional, Tuple, Any
import logging

logger = logging.getLogger(__name__)


class AugmentationError(Exception):
    """Base exception for augmentation errors"""
    pass


class ConfigurationError(AugmentationError):
    """Invalid configuration parameters"""
    pass


class ValidationError(AugmentationError):
    """Configuration validation failure"""
    pass


class AugmentConfig:
    """
    Configuration manager for MRAugment integration.
    Handles parameter loading, validation, and conversion to MRAugment format.
    """
    
    def __init__(self, **kwargs):
        """Initialize configuration with parameters"""
        # Core augmentation settings
        self.aug_on = kwargs.get('aug_on', True)
        self.aug_strength = kwargs.get('aug_strength', 0.6)
        self.aug_schedule = kwargs.get('aug_schedule', 'exp')
        self.aug_delay = kwargs.get('aug_delay', 0)
        self.aug_exp_decay = kwargs.get('aug_exp_decay', 5.0)
        
        # Transform weights
        self.aug_weight_translation = kwargs.get('aug_weight_translation', 1.0)
        self.aug_weight_rotation = kwargs.get('aug_weight_rotation', 0.5)
        self.aug_weight_scaling = kwargs.get('aug_weight_scaling', 1.0)
        self.aug_weight_shearing = kwargs.get('aug_weight_shearing', 0.0)
        self.aug_weight_rot90 = kwargs.get('aug_weight_rot90', 0.5)
        self.aug_weight_fliph = kwargs.get('aug_weight_fliph', 0.5)
        self.aug_weight_flipv = kwargs.get('aug_weight_flipv', 0.0)
        
        # Transform parameters
        self.aug_max_translation_x = kwargs.get('aug_max_translation_x', 0.125)
        self.aug_max_translation_y = kwargs.get('aug_max_translation_y', 0.08)
        self.aug_max_rotation = kwargs.get('aug_max_rotation', 15.0)
        self.aug_max_scaling = kwargs.get('aug_max_scaling', 0.15)
        self.aug_max_shearing_x = kwargs.get('aug_max_shearing_x', 0.0)
        self.aug_max_shearing_y = kwargs.get('aug_max_shearing_y', 0.0)
        
        # Processing options
        self.aug_upsample = kwargs.get('aug_upsample', True)
        self.aug_upsample_factor = kwargs.get('aug_upsample_factor', 2)
        self.aug_upsample_order = kwargs.get('aug_upsample_order', 1)
        self.aug_interpolation_order = kwargs.get('aug_interpolation_order', 1)
        
        # Validate configuration
        self._validate()
    
    def _validate(self):
        """Validate configuration parameters"""
        errors = []
        
        # Validate strength
        if not 0.0 <= self.aug_strength <= 1.0:
            errors.append(f"aug_strength must be between 0.0 and 1.0, got {self.aug_strength}")
        
        # Validate schedule
        if self.aug_schedule not in ['constant', 'exp', 'linear']:
            errors.append(f"aug_schedule must be 'constant', 'exp', or 'linear', got {self.aug_schedule}")
        
        # Validate weights
        for attr in ['aug_weight_translation', 'aug_weight_rotation', 'aug_weight_scaling',
                     'aug_weight_shearing', 'aug_weight_rot90', 'aug_weight_fliph', 'aug_weight_flipv']:
            value = getattr(self, attr)
            if not 0.0 <= value <= 1.0:
                errors.append(f"{attr} must be between 0.0 and 1.0, got {value}")
        
        # Validate translation parameters
        if not 0.0 <= self.aug_max_translation_x <= 1.0:
            errors.append(f"aug_max_translation_x must be between 0.0 and 1.0, got {self.aug_max_translation_x}")
        if not 0.0 <= self.aug_max_translation_y <= 1.0:
            errors.append(f"aug_max_translation_y must be between 0.0 and 1.0, got {self.aug_max_translation_y}")
        
        # Validate rotation
        if not 0.0 <= self.aug_max_rotation <= 360.0:
            errors.append(f"aug_max_rotation must be between 0.0 and 360.0, got {self.aug_max_rotation}")
        
        # Validate scaling
        if not 0.0 <= self.aug_max_scaling <= 1.0:
            errors.append(f"aug_max_scaling must be between 0.0 and 1.0, got {self.aug_max_scaling}")
        
        # Validate upsample factor
        if not isinstance(self.aug_upsample_factor, int) or not 1 <= self.aug_upsample_factor <= 4:
            errors.append(f"aug_upsample_factor must be integer between 1 and 4, got {self.aug_upsample_factor}")
        
        if errors:
            raise ValidationError(f"Configuration validation failed: {'; '.join(errors)}")
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'AugmentConfig':
        """Load configuration from YAML file"""
        if not os.path.exists(yaml_path):
            raise FileNotFoundError(f"Configuration file not found: {yaml_path}")
        
        try:
            with open(yaml_path, 'r') as f:
                config_dict = yaml.safe_load(f)
            
            if config_dict is None:
                config_dict = {}
            
            return cls(**config_dict)
        
        except yaml.YAMLError as e:
            raise ConfigurationError(f"Failed to parse YAML file {yaml_path}: {e}")
        except Exception as e:
            raise ConfigurationError(f"Failed to load configuration from {yaml_path}: {e}")
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'AugmentConfig':
        """Create configuration from dictionary"""
        return cls(**config_dict)
    
    @classmethod
    def from_args(cls, args) -> 'AugmentConfig':
        """Create configuration from argparse namespace"""
        config_dict = {}
        
        # Map argparse arguments to config parameters
        if hasattr(args, 'augment_strength') and args.augment_strength is not None:
            config_dict['aug_strength'] = args.augment_strength
        if hasattr(args, 'augment_schedule') and args.augment_schedule is not None:
            config_dict['aug_schedule'] = args.augment_schedule
        
        return cls(**config_dict)
    
    @classmethod
    def default(cls) -> 'AugmentConfig':
        """Create default configuration"""
        return cls(
            aug_on=True,
            aug_strength=0.6,
            aug_schedule='exp',
            aug_weight_translation=1.0,
            aug_weight_rotation=0.5,
            aug_weight_scaling=1.0,
            aug_weight_fliph=0.5,
            aug_upsample=True,
            aug_upsample_factor=2
        )
    
    def merge(self, other: 'AugmentConfig') -> 'AugmentConfig':
        """Merge with another configuration (other takes precedence)"""
        merged_dict = self.to_dict()
        merged_dict.update(other.to_dict())
        return AugmentConfig.from_dict(merged_dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}
    
    def to_hparams(self):
        """Convert to MRAugment hyperparameters format"""
        # Create a simple namespace object for compatibility with MRAugment
        class HParams:
            pass
        
        hparams = HParams()
        for key, value in self.to_dict().items():
            setattr(hparams, key, value)
        
        return hparams
    
    def override_from_args(self, args):
        """Override parameters from command line arguments"""
        if hasattr(args, 'augment_strength') and args.augment_strength is not None:
            self.aug_strength = args.augment_strength
            logger.info(f"Overriding augmentation strength: {self.aug_strength}")
        
        if hasattr(args, 'augment_schedule') and args.augment_schedule is not None:
            self.aug_schedule = args.augment_schedule
            logger.info(f"Overriding augmentation schedule: {self.aug_schedule}")
        
        if hasattr(args, 'no_augment') and args.no_augment:
            self.aug_on = False
            logger.info("Augmentation disabled via --no-augment flag")
        
        # Re-validate after override
        self._validate()
    
    def get_preset_path(self, preset_name: str) -> str:
        """Get path to preset configuration file"""
        current_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        return os.path.join(current_dir, 'configs', f'augment_{preset_name}.yaml')
    
    @classmethod
    def from_preset(cls, preset_name: str) -> 'AugmentConfig':
        """Load from preset configuration"""
        if preset_name not in ['minimal', 'basic', 'aggressive']:
            raise ValueError(f"Unknown preset: {preset_name}. Choose from: minimal, basic, aggressive")
        
        config = cls()
        preset_path = config.get_preset_path(preset_name)
        
        if os.path.exists(preset_path):
            return cls.from_yaml(preset_path)
        else:
            logger.warning(f"Preset file not found: {preset_path}, using defaults")
            return cls.default()
    
    def __str__(self) -> str:
        """String representation"""
        return f"AugmentConfig(strength={self.aug_strength}, schedule={self.aug_schedule}, enabled={self.aug_on})"