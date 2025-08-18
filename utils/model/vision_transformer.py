# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import logging
import math

from os.path import join as pjoin

import torch
import torch.nn as nn
import numpy as np

from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm
from torch.nn.modules.utils import _pair
from scipy import ndimage
from .swin_transformer_unet_skip_expand_decoder_sys import SwinTransformerSys

logger = logging.getLogger(__name__)

class SwinUnet(nn.Module):
    # STEP 1: Add 'img_size=None' to the definition to fix the TypeError.
    def __init__(self, config, img_size=None, num_classes=21843, zero_head=False, vis=False):
        super(SwinUnet, self).__init__()
        self.num_classes = num_classes
        self.zero_head = zero_head
        self.config = config

        # STEP 2: Use None for img_size to let model determine size from first input
        if img_size is None:
            # Use None to allow model to determine size from first forward pass
            final_img_size = None
        else:
            # Use the size provided by the calling code (e.g., from varnet.py).
            final_img_size = img_size

        # Now, initialize the underlying model
        self.swin_unet = SwinTransformerSys(
                                img_size=final_img_size,
                                patch_size=config.MODEL.SWIN.PATCH_SIZE,
                                in_chans=config.MODEL.SWIN.IN_CHANS,
                                num_classes=self.num_classes,
                                embed_dim=config.MODEL.SWIN.EMBED_DIM,
                                depths=config.MODEL.SWIN.DEPTHS,
                                num_heads=config.MODEL.SWIN.NUM_HEADS,
                                window_size=config.MODEL.SWIN.WINDOW_SIZE,
                                mlp_ratio=config.MODEL.SWIN.MLP_RATIO,
                                qkv_bias=config.MODEL.SWIN.QKV_BIAS,
                                qk_scale=config.MODEL.SWIN.QK_SCALE,
                                drop_rate=config.MODEL.DROP_RATE,
                                drop_path_rate=config.MODEL.DROP_PATH_RATE,
                                ape=config.MODEL.SWIN.APE,
                                patch_norm=config.MODEL.SWIN.PATCH_NORM,
                                use_checkpoint=config.TRAIN.USE_CHECKPOINT)

    def forward(self, x):
        # The input `x` should already be padded before calling this
        # print(f"SwinUNet input shape: {x.shape}")
        
        # --- FIX: REMOVE the entire re-initialization block ---
        
        if x.size()[1] == 1:
            x = x.repeat(1,3,1,1)
            
        logits = self.swin_unet(x)
        # You will need to crop the logits back to original size outside this model
        return logits

    def pad_if_needed(self, x):
        B, C, H, W = x.shape
        
        # The network requires input dimensions to be a multiple of the total downsampling factor
        # Factor = patch_size * 2^(num_layers - 1)
        # Example: patch_size=4, num_layers=4 -> Factor = 4 * 2^3 = 32
        patch_size = self.swin_unet.patch_embed.patch_size[0]
        num_layers = len(self.config.MODEL.SWIN.DEPTHS)
        factor = patch_size * (2 ** (num_layers - 1))

        pad_h = (factor - H % factor) % factor
        pad_w = (factor - W % factor) % factor

        if pad_h > 0 or pad_w > 0:
            # Pad format is (pad_left, pad_right, pad_top, pad_bottom)
            return F.pad(x, (0, pad_w, 0, pad_h))
        return x

    # A corrected and safer load_from method
def load_from(self, config):
    pretrained_path = config.MODEL.PRETRAIN_CKPT
    if pretrained_path is not None:
        print(f"Loading pretrained weights from: {pretrained_path}")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        pretrained_dict = torch.load(pretrained_path, map_location=device)

        # Handle different checkpoint formats
        if "model" in pretrained_dict:
            pretrained_dict = pretrained_dict['model']

        # --- FIX: Load pretrained weights into the ENCODER part of our model ---
        encoder_dict = {}
        for k, v in pretrained_dict.items():
            # Select weights that belong to the swin transformer encoder part
            # This typically includes 'patch_embed', 'layers', 'absolute_pos_embed', etc.
            if k in self.swin_unet.state_dict() and not k.startswith('layers_up') and not k.startswith('up.'):
                encoder_dict[k] = v

        # Filter out weights with mismatched shapes (e.g., relative_position_bias_table)
        model_dict = self.swin_unet.state_dict()
        encoder_dict = {k: v for k, v in encoder_dict.items() if k in model_dict and v.shape == model_dict[k].shape}
        
        msg = self.swin_unet.load_state_dict(encoder_dict, strict=False)
        print("Pretrained weight loading message:", msg)
    else:
        print("No pretrained checkpoint specified.")
 