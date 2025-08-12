"""
MRAugment data augmentation pipeline adapted for FastMRI.
Physics-aware data augmentation for MRI reconstruction.

Adapted from the original MRAugment repository:
https://github.com/z-fabian/MRAugment
"""
import numpy as np
from math import exp
import torch
import torchvision.transforms.functional as TF
from .helpers import complex_crop_if_needed, crop_if_needed, complex_channel_first, complex_channel_last

# Use fastmri transforms 
try:
    from utils.model.fastmri.fftc import fft2c, ifft2c
    from utils.model.fastmri.coil_combine import rss_complex
    from utils.model.fastmri.math import complex_abs
    from utils.model.fastmri.data import transforms as T
    fastmri_available = True
except ImportError:
    # Fallback implementations
    def fft2c(x):
        """2D FFT with proper centering"""
        if torch.is_complex(x):
            return torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(x, dim=(-2, -1)), norm='ortho'), dim=(-2, -1))
        else:
            # Handle [..., 2] format
            complex_x = torch.complex(x[..., 0], x[..., 1])
            result = torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(complex_x, dim=(-2, -1)), norm='ortho'), dim=(-2, -1))
            return torch.stack([result.real, result.imag], dim=-1)
    
    def ifft2c(x):
        """2D IFFT with proper centering"""
        if torch.is_complex(x):
            return torch.fft.fftshift(torch.fft.ifft2(torch.fft.ifftshift(x, dim=(-2, -1)), norm='ortho'), dim=(-2, -1))
        else:
            # Handle [..., 2] format
            complex_x = torch.complex(x[..., 0], x[..., 1])
            result = torch.fft.fftshift(torch.fft.ifft2(torch.fft.ifftshift(complex_x, dim=(-2, -1)), norm='ortho'), dim=(-2, -1))
            return torch.stack([result.real, result.imag], dim=-1)
    
    def complex_abs(x):
        """Compute absolute value of complex tensor"""
        if torch.is_complex(x):
            return torch.abs(x)
        else:
            return torch.sqrt(x[..., 0]**2 + x[..., 1]**2)
    
    def rss_complex(x):
        """Root sum of squares for complex data"""
        if torch.is_complex(x):
            return torch.sqrt(torch.sum(torch.abs(x)**2, dim=0))
        else:
            return torch.sqrt(torch.sum(x[..., 0]**2 + x[..., 1]**2, dim=0))
    
    class T:
        @staticmethod
        def complex_center_crop(x, shape):
            return x  # Placeholder
    
    fastmri_available = False


class AugmentationPipeline:
    """
    Describes the transformations applied to MRI data and handles
    augmentation probabilities including generating random parameters for 
    each augmentation.
    """
    
    def __init__(self, hparams):
        self.hparams = hparams
        self.weight_dict = {
            'translation': hparams.aug_weight_translation,
            'rotation': hparams.aug_weight_rotation,
            'scaling': hparams.aug_weight_scaling,
            'shearing': hparams.aug_weight_shearing,
            'rot90': hparams.aug_weight_rot90,
            'fliph': hparams.aug_weight_fliph,
            'flipv': hparams.aug_weight_flipv
        }
        self.upsample_augment = hparams.aug_upsample
        self.upsample_factor = hparams.aug_upsample_factor
        self.upsample_order = hparams.aug_upsample_order
        self.transform_order = hparams.aug_interpolation_order
        self.augmentation_strength = 0.0
        self.rng = np.random.RandomState()

    def augment_image(self, im, max_output_size=None):
        """Apply augmentation to image"""
        # Trailing dims must be image height and width (for torchvision) 
        im = complex_channel_first(im)
        
        # ---------------------------  
        # pixel preserving transforms
        # ---------------------------  
        # Horizontal flip
        if self.random_apply('fliph'):
            im = TF.hflip(im)

        # Vertical flip 
        if self.random_apply('flipv'):
            im = TF.vflip(im)

        # Rotation by multiples of 90 deg 
        if self.random_apply('rot90'):
            k = self.rng.randint(1, 4)  
            im = torch.rot90(im, k, dims=[-2, -1])

        # Translation by integer number of pixels
        if self.random_apply('translation'):
            h, w = im.shape[-2:]
            t_x = self.rng.uniform(-self.hparams.aug_max_translation_x, self.hparams.aug_max_translation_x)
            t_x = int(t_x * h)
            t_y = self.rng.uniform(-self.hparams.aug_max_translation_y, self.hparams.aug_max_translation_y)
            t_y = int(t_y * w)
            
            pad, top, left = self._get_translate_padding_and_crop(im, (t_x, t_y))
            im = TF.pad(im, padding=pad, padding_mode='reflect')
            im = TF.crop(im, top, left, h, w)

        # ------------------------       
        # interpolating transforms
        # ------------------------  
        interp = False 

        # Rotation
        if self.random_apply('rotation'):
            interp = True
            rot = self.rng.uniform(-self.hparams.aug_max_rotation, self.hparams.aug_max_rotation)
        else:
            rot = 0.

        # Shearing
        if self.random_apply('shearing'):
            interp = True
            shear_x = self.rng.uniform(-self.hparams.aug_max_shearing_x, self.hparams.aug_max_shearing_x)
            shear_y = self.rng.uniform(-self.hparams.aug_max_shearing_y, self.hparams.aug_max_shearing_y)
        else:
            shear_x, shear_y = 0., 0.

        # Scaling
        if self.random_apply('scaling'):
            interp = True
            scale = self.rng.uniform(1-self.hparams.aug_max_scaling, 1 + self.hparams.aug_max_scaling)
        else:
            scale = 1.

        # Upsample if needed
        upsample = interp and self.upsample_augment
        if upsample:
            upsampled_shape = [im.shape[-2] * self.upsample_factor, im.shape[-1] * self.upsample_factor]
            original_shape = im.shape[-2:]
            interpolation = TF.InterpolationMode.BICUBIC if self.upsample_order == 3 else TF.InterpolationMode.BILINEAR
            im = TF.resize(im, size=upsampled_shape, interpolation=interpolation)

        # Apply interpolating transformations 
        if interp:
            h, w = im.shape[-2:]
            pad = self._get_affine_padding_size(im, rot, scale, (shear_x, shear_y))
            im = TF.pad(im, padding=pad, padding_mode='reflect')
            im = TF.affine(im,
                           angle=rot,
                           scale=scale,
                           shear=(shear_x, shear_y),
                           translate=[0, 0],
                           interpolation=TF.InterpolationMode.BILINEAR)
            im = TF.center_crop(im, (h, w))
        
        # Downsampling
        if upsample:
            im = TF.resize(im, size=original_shape, interpolation=interpolation)
        
        # Final cropping if augmented image is too large
        if max_output_size is not None:
            im = crop_if_needed(im, max_output_size)
            
        # Reset original channel ordering
        im = complex_channel_last(im)
        
        return im
    
    def augment_from_kspace(self, kspace, target_size, max_train_size=None):       
        """Augment k-space data"""
        # Convert PyTorch complex tensor to FastMRI format for processing
        if torch.is_complex(kspace):
            # Convert complex tensor to [..., 2] format
            im = ifft2c(kspace) 
            if torch.is_complex(im):
                im = torch.stack([im.real, im.imag], dim=-1)
        else:
            # Already in FastMRI format
            im = ifft2c(kspace)
            
        im = self.augment_image(im, max_output_size=max_train_size)
        target = self.im_to_target(im, target_size)
        
        # Convert back to complex for fft2c
        if im.shape[-1] == 2:
            im_complex = torch.complex(im[..., 0], im[..., 1])
        else:
            im_complex = im
            
        kspace_result = fft2c(im_complex)
        
        # Return in complex format for consistency
        if not torch.is_complex(kspace_result) and kspace_result.shape[-1] == 2:
            kspace_result = torch.complex(kspace_result[..., 0], kspace_result[..., 1])
            
        return kspace_result, target
    
    def im_to_target(self, im, target_size):     
        """Convert image to target format - always return exact target_size"""
        
        if im.shape[-1] == 2:  # FastMRI format [..., 2]
            if len(im.shape) == 3:  # Single-coil [H, W, 2]
                # Compute magnitude
                target = complex_abs(im)
            else:  # Multi-coil [C, H, W, 2]
                # Compute root sum of squares
                target = rss_complex(im)
        else:
            # Complex tensor format
            target = complex_abs(im) if torch.is_complex(im) else torch.abs(im)
        
        # Always crop/pad to exact target_size to maintain consistency
        current_h, current_w = target.shape[-2], target.shape[-1]
        target_h, target_w = target_size[0], target_size[1]
        
        # Center crop or pad to match exact target size
        if current_h != target_h or current_w != target_w:
            # Calculate crop/pad parameters
            h_diff = current_h - target_h
            w_diff = current_w - target_w
            
            if h_diff >= 0 and w_diff >= 0:
                # Need to crop
                h_start = h_diff // 2
                w_start = w_diff // 2
                target = target[..., h_start:h_start+target_h, w_start:w_start+target_w]
            elif h_diff <= 0 and w_diff <= 0:
                # Need to pad
                pad_h = (-h_diff + 1) // 2
                pad_w = (-w_diff + 1) // 2
                target = torch.nn.functional.pad(target, (pad_w, -h_diff-pad_h, pad_h, -w_diff-pad_w))
                # Crop to exact size in case of odd padding
                target = target[..., :target_h, :target_w]
            else:
                # Mix of crop and pad - handle separately
                if h_diff > 0:  # Crop height
                    h_start = h_diff // 2
                    target = target[..., h_start:h_start+target_h, :]
                else:  # Pad height
                    pad_h = (-h_diff + 1) // 2
                    target = torch.nn.functional.pad(target, (0, 0, pad_h, -h_diff-pad_h))
                    target = target[..., :target_h, :]
                    
                if w_diff > 0:  # Crop width
                    w_start = w_diff // 2
                    target = target[..., w_start:w_start+target_w]
                else:  # Pad width
                    pad_w = (-w_diff + 1) // 2
                    target = torch.nn.functional.pad(target, (pad_w, -w_diff-pad_w))
                    target = target[..., :target_w]
        
        return target  
            
    def random_apply(self, transform_name):
        """Check if transform should be applied based on probability"""
        if self.rng.uniform() < self.weight_dict[transform_name] * self.augmentation_strength:
            return True
        else: 
            return False
        
    def set_augmentation_strength(self, p):
        """Set augmentation strength"""
        self.augmentation_strength = p

    @staticmethod
    def _get_affine_padding_size(im, angle, scale, shear):
        """
        Calculates the necessary padding size before applying the 
        general affine transformation.
        """
        h, w = im.shape[-2:]
        corners = [
            [-h/2, -w/2, 1.],
            [-h/2, w/2, 1.], 
            [h/2, w/2, 1.], 
            [h/2, -w/2, 1.]
        ]
        mx = torch.tensor(TF._get_inverse_affine_matrix([0.0, 0.0], -angle, [0, 0], scale, [-s for s in shear])).reshape(2,3)
        corners = torch.cat([torch.tensor(c).reshape(3,1) for c in corners], dim=1)
        tr_corners = torch.matmul(mx, corners)
        all_corners = torch.cat([tr_corners, corners[:2, :]], dim=1)
        bounding_box = all_corners.amax(dim=1) - all_corners.amin(dim=1)
        px = torch.clip(torch.floor((bounding_box[0] - h) / 2), min=0.0, max=h-1) 
        py = torch.clip(torch.floor((bounding_box[1] - w) / 2),  min=0.0, max=w-1)
        return int(py.item()), int(px.item())

    @staticmethod
    def _get_translate_padding_and_crop(im, translation):
        """Calculate padding and crop parameters for translation"""
        t_x, t_y = translation
        h, w = im.shape[-2:]
        pad = [0, 0, 0, 0]
        if t_x >= 0:
            pad[3] = min(t_x, h - 1) # pad bottom
            top = pad[3]
        else:
            pad[1] = min(-t_x, h - 1) # pad top
            top = 0
        if t_y >= 0:
            pad[0] = min(t_y, w - 1) # pad left
            left = 0
        else:
            pad[2] = min(-t_y, w - 1) # pad right
            left = pad[2]
        return pad, top, left

            
class DataAugmentor:
    """
    High-level class encompassing the augmentation pipeline and augmentation
    probability scheduling. Simplified for FastMRI integration.
    """
        
    def __init__(self, hparams, current_epoch_fn=None):
        """
        Args:
            hparams: Configuration object with augmentation parameters
            current_epoch_fn: Function that returns current epoch (optional)
        """
        self.current_epoch_fn = current_epoch_fn if current_epoch_fn else lambda: 0
        self.hparams = hparams
        self.aug_on = hparams.aug_on
        if self.aug_on:
            self.augmentation_pipeline = AugmentationPipeline(hparams)
        
        # Set default max train resolution if not provided
        if hasattr(hparams, 'max_train_resolution'):
            self.max_train_resolution = hparams.max_train_resolution
        else:
            self.max_train_resolution = None
        
    def __call__(self, kspace, target_size=None):
        """
        Generates augmented kspace and corresponding augmented target pair.
        
        Args:
            kspace: torch tensor of shape [C, H, W, 2] (multi-coil) or [H, W, 2]
            target_size: [H, W] shape of the generated augmented target
        
        Returns:
            Tuple of (augmented_kspace, augmented_target)
        """
        # Set augmentation probability based on schedule
        if self.aug_on:
            current_epoch = self.current_epoch_fn()
            strength = self._get_scheduled_strength(current_epoch)
            self.augmentation_pipeline.set_augmentation_strength(strength)
            
            # Use default target size if not provided
            if target_size is None:
                if len(kspace.shape) == 3:  # Single-coil
                    target_size = kspace.shape[-3:-1]  # [H, W]
                else:  # Multi-coil
                    target_size = kspace.shape[-3:-1]  # [H, W]
            
            return self.augmentation_pipeline.augment_from_kspace(
                kspace, target_size, self.max_train_resolution
            )
        else:
            # No augmentation, return original data
            if target_size is None:
                if len(kspace.shape) == 3:  # Single-coil
                    target_size = kspace.shape[-3:-1]
                else:  # Multi-coil  
                    target_size = kspace.shape[-3:-1]
            
            # Create target from original kspace
            im = ifft2c(kspace)
            target = self.augmentation_pipeline.im_to_target(im, target_size) if hasattr(self, 'augmentation_pipeline') else None
            return kspace, target
    
    def _get_scheduled_strength(self, epoch):
        """Calculate augmentation strength based on schedule"""
        if not hasattr(self.hparams, 'aug_schedule'):
            return self.hparams.aug_strength
            
        if self.hparams.aug_schedule == 'constant':
            return self.hparams.aug_strength
        elif self.hparams.aug_schedule == 'exp':
            # Exponential decay schedule
            decay_rate = getattr(self.hparams, 'aug_exp_decay', 5.0)
            delay = getattr(self.hparams, 'aug_delay', 0)
            if epoch < delay:
                return 0.0
            else:
                decay_factor = exp(-(epoch - delay) / decay_rate)
                return self.hparams.aug_strength * decay_factor
        elif self.hparams.aug_schedule == 'linear':
            # Linear decay schedule  
            delay = getattr(self.hparams, 'aug_delay', 0)
            if epoch < delay:
                return 0.0
            else:
                # Linear decay over remaining epochs
                max_epochs = getattr(self.hparams, 'max_epochs', 100)
                remaining_ratio = max(0, (max_epochs - epoch) / (max_epochs - delay))
                return self.hparams.aug_strength * remaining_ratio
        else:
            return self.hparams.aug_strength