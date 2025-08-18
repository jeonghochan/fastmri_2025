import shutil
import numpy as np
import torch
import torch.nn as nn
import time
from pathlib import Path
import copy

from collections import defaultdict
from utils.data.load_data import create_data_loaders
from utils.common.utils import save_reconstructions, ssim_loss
from utils.common.loss_function import SSIMLoss
from utils.model.varnet import VarNet

import os

def train_epoch(args, epoch, model, data_loader, optimizer, loss_type):
    model.train()
    start_epoch = start_iter = time.perf_counter()
    len_loader = len(data_loader)
    total_loss = 0.

    for iter, data in enumerate(data_loader):
        mask, kspace, target, maximum, _, _ = data
        mask = mask.cuda(non_blocking=True)
        kspace = kspace.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        maximum = maximum.cuda(non_blocking=True)
        
        # Ensure FP32 calculation (mask stays boolean for torch.where)
        kspace = kspace.float()
        target = target.float()
        maximum = maximum.float()

        output = model(kspace, mask)
        loss = loss_type(output, target, maximum)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        if iter % args.report_interval == 0:
            print(
                f'Epoch = [{epoch:3d}/{args.num_epochs:3d}] '
                f'Iter = [{iter:4d}/{len(data_loader):4d}] '
                f'Loss = {loss.item():.4g} '
                f'Time = {time.perf_counter() - start_iter:.4f}s',
            )
            start_iter = time.perf_counter()
    total_loss = total_loss / len_loader
    return total_loss, time.perf_counter() - start_epoch


def validate(args, model, data_loader):
    model.eval()
    reconstructions = defaultdict(dict)
    targets = defaultdict(dict)
    start = time.perf_counter()

    with torch.no_grad():
        for iter, data in enumerate(data_loader):
            mask, kspace, target, _, fnames, slices = data
            kspace = kspace.cuda(non_blocking=True)
            mask = mask.cuda(non_blocking=True)
            
            # Ensure FP32 calculation (mask stays boolean for torch.where)
            kspace = kspace.float()
            
            output = model(kspace, mask)

            for i in range(output.shape[0]):
                reconstructions[fnames[i]][int(slices[i])] = output[i].cpu().numpy()
                targets[fnames[i]][int(slices[i])] = target[i].numpy()

    for fname in reconstructions:
        reconstructions[fname] = np.stack(
            [out for _, out in sorted(reconstructions[fname].items())]
        )
    for fname in targets:
        targets[fname] = np.stack(
            [out for _, out in sorted(targets[fname].items())]
        )
    metric_loss = sum([ssim_loss(targets[fname], reconstructions[fname]) for fname in reconstructions])
    num_subjects = len(reconstructions)
    return metric_loss, num_subjects, reconstructions, targets, None, time.perf_counter() - start


def save_model(args, exp_dir, epoch, model, optimizer, best_val_loss, is_new_best, top3_models):
    torch.save({
            'epoch': epoch,
            'best_val_loss': best_val_loss,
            'args': args,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'exp_dir': exp_dir 
        },
        f=exp_dir / 'model.pt')

    if is_new_best:
        shutil.copyfile(exp_dir / 'model.pt', exp_dir / 'best_model.pt')

    current_loss = best_val_loss.item() if hasattr(best_val_loss, 'item') else best_val_loss

    # Add current model to top3_models list
    top3_models.append({
        'epoch': epoch,
        'val_loss': current_loss,
        'model_path': exp_dir / f'model_epoch_{epoch}.pt'
    })

    # Sort by validation loss and keep only top 5
    top3_models.sort(key=lambda x: x['val_loss'])

    # Remove old models if we have more than 5
    while len(top3_models) > 5:
        removed_model = top3_models.pop()
        if removed_model['model_path'].exists():
            removed_model['model_path'].unlink()

    # Only save epoch files for models in top 5
    for model_info in top3_models:
        if model_info['epoch'] == epoch and not model_info['model_path'].exists():
            shutil.copyfile(exp_dir / 'model.pt', model_info['model_path'])

    return top3_models

        
def train(args, kspace_augment_config_path=None):
    device = torch.device(f'cuda:{args.GPU_NUM}' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(device)
    print('Current cuda device: ', torch.cuda.current_device())
    
    # Force FP32 for GTX 1080 compatibility
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    model = VarNet(num_cascades=args.cascade, 
                   chans=args.chans, 
                   sens_chans=args.sens_chans,
                   use_transformer=args.use_transformer)
    model.to(device=device)

    loss_type = SSIMLoss().to(device=device)
    optimizer = torch.optim.Adam(model.parameters(), args.lr)

    best_val_loss = 1.
    start_epoch = 0
    top3_models = []  # List to track top 3 models

    # Resume training from checkpoint if specified
    if hasattr(args, 'resume') and args.resume:
        if os.path.isfile(args.resume):
            print(f"=> Loading checkpoint '{args.resume}'")
            try:
                # Always assume full checkpoint dict
                checkpoint = torch.load(args.resume, map_location='cpu', weights_only=False)
                model.load_state_dict(checkpoint['model'])

            except Exception as e:
                print(f"=> Failed to load checkpoint: {e}")
                print("=> Starting training from scratch")
        else:
            print(f"=> No checkpoint found at '{args.resume}'")


    # Create data loaders with k-space augmentation support
    train_loader = create_data_loaders(data_path=args.data_path_train, args=args, shuffle=True, augment_config_path=kspace_augment_config_path)
    val_loader = create_data_loaders(data_path=args.data_path_val, args=args, augment_config_path=None)  # No augmentation for validation
    
    # Set up epoch tracking for k-space augmentation
    if kspace_augment_config_path is not None:
        train_transform = train_loader.dataset.transform
        if hasattr(train_transform, 'set_epoch'):
            print("K-space augmentation scheduling enabled")
    
    val_loss_log = np.empty((0, 2))
    for epoch in range(start_epoch, args.num_epochs):
        print(f'Epoch #{epoch:2d} ............... {args.net_name} ...............')
        
        # Update epoch for k-space augmentation scheduling
        if kspace_augment_config_path is not None:
            train_transform = train_loader.dataset.transform
            if hasattr(train_transform, 'set_epoch'):
                train_transform.set_epoch(epoch)
        
        train_loss, train_time = train_epoch(args, epoch, model, train_loader, optimizer, loss_type)
        val_loss, num_subjects, reconstructions, targets, inputs, val_time = validate(args, model, val_loader)
        
        val_loss_log = np.append(val_loss_log, np.array([[epoch, val_loss]]), axis=0)
        file_path = os.path.join(args.val_loss_dir, "val_loss_log")
        np.save(file_path, val_loss_log)
        print(f"loss file saved! {file_path}")

        train_loss = torch.tensor(train_loss).cuda(non_blocking=True)
        val_loss = torch.tensor(val_loss).cuda(non_blocking=True)
        num_subjects = torch.tensor(num_subjects).cuda(non_blocking=True)

        val_loss = val_loss / num_subjects

        is_new_best = val_loss < best_val_loss
        best_val_loss = min(best_val_loss, val_loss)

        top3_models = save_model(args, args.exp_dir, epoch + 1, model, optimizer, best_val_loss, is_new_best, top3_models)
        print(
            f'Epoch = [{epoch:4d}/{args.num_epochs:4d}] TrainLoss = {train_loss:.4g} '
            f'ValLoss = {val_loss:.4g} TrainTime = {train_time:.4f}s ValTime = {val_time:.4f}s',
        )

        if is_new_best:
            print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@NewRecord@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
            start = time.perf_counter()
            save_reconstructions(reconstructions, args.val_dir, targets=targets, inputs=inputs)
            print(
                f'ForwardTime = {time.perf_counter() - start:.4f}s',
            )
