import numpy as np
import torch

from collections import defaultdict
from utils.common.utils import save_reconstructions
from utils.data.load_data import create_data_loaders
from utils.model.varnet import VarNet

def test(args, model, data_loader):
    model.eval()
    reconstructions = defaultdict(dict)
    
    with torch.no_grad():
        for (mask, kspace, _, _, fnames, slices) in data_loader:
            kspace = kspace.cuda(non_blocking=True)
            mask = mask.cuda(non_blocking=True)
            output = model(kspace, mask)

            for i in range(output.shape[0]):
                reconstructions[fnames[i]][int(slices[i])] = output[i].cpu().numpy()

    for fname in reconstructions:
        reconstructions[fname] = np.stack(
            [out for _, out in sorted(reconstructions[fname].items())]
        )
    return reconstructions, None


def forward(args):

    device = torch.device(f'cuda:{args.GPU_NUM}' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(device)
    print ('Current cuda device ', torch.cuda.current_device())

    # Load checkpoint first to get training configuration
    checkpoint_file = args.checkpoint if hasattr(args, 'checkpoint') and args.checkpoint else 'best_model.pt'
    checkpoint = torch.load(args.exp_dir / checkpoint_file, map_location='cpu', weights_only=False)
    print(f"Loaded model from epoch {checkpoint['epoch']}, best val loss: {checkpoint['best_val_loss'].item()}")
    
    # Robust auto-detection: Check actual model keys in checkpoint
    model_keys = list(checkpoint['model'].keys())
    
    # Check if SwinUNet keys are present (transformer-specific keys)
    swin_keys = ['swin_unet.patch_embed', 'swin_unet.layers.0.blocks.0.attn']
    unet_keys = ['down_sample_layers.0.layers.0', 'up_conv.0.layers.0']
    
    has_swin_keys = any(any(swin_key in key for key in model_keys) for swin_key in swin_keys)
    has_unet_keys = any(any(unet_key in key for key in model_keys) for unet_key in unet_keys)
    
    # Determine architecture from actual model structure
    if has_swin_keys:
        use_transformer = True
        architecture = "SwinUNet (Transformer)"
    elif has_unet_keys:
        use_transformer = False
        architecture = "CNN UNet"
    else:
        # Fallback to args if available
        saved_args = checkpoint['args']
        use_transformer = getattr(saved_args, 'use_transformer', False)
        architecture = "SwinUNet (Transformer)" if use_transformer else "CNN UNet (Fallback)"
    
    print(f"Auto-detected architecture: {architecture}")
    print(f"Model contains SwinUNet keys: {has_swin_keys}")
    print(f"Model contains UNet keys: {has_unet_keys}")
    
    # Create model with detected architecture
    model = VarNet(num_cascades=args.cascade, 
                   chans=args.chans, 
                   sens_chans=args.sens_chans,
                   use_transformer=use_transformer)  # Use detected value, not args
    model.to(device=device)
    model.load_state_dict(checkpoint['model'])
    
    forward_loader = create_data_loaders(data_path = args.data_path, args = args, isforward = True)
    reconstructions, inputs = test(args, model, forward_loader)
    save_reconstructions(reconstructions, args.forward_dir, inputs=inputs)