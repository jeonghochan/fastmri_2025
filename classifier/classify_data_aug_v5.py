#!/usr/bin/env python3
"""
train_brain_knee_cnn.py  – FastMRI k-space Brain·Knee 분류용 간단 CNN 모델 학습 (Fixed-slice sampling without I/O caching)
"""
import os
import glob
import random
import argparse

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time

from torch.utils.data import Dataset, DataLoader, random_split
# from torch.cuda.amp import GradScaler, autocast
from torch.amp import GradScaler, autocast

# k-space masking utilities
from subsample import RandomMaskFunc, EquispacedMaskFractionFunc, MagicMaskFunc

# ─── reproducibility ─────────────────────────────────────────
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# ─── label helper ─────────────────────────────────────────────
def label_from_name(fname):
    low = fname.lower()
    if "brain" in low:
        return 1
    elif "knee" in low:
        return 0
    else:
        raise ValueError(f"Cannot extract label from {fname}")

class KSpaceDataset(Dataset):
    def __init__(self, filepaths, ds='kspace', num_slices=4,
                 noise_std=0.01, flip_prob=0.5,
                 center_fractions=[0.08], accelerations=[4,8],
                 p_random=0.60, p_equi=0.35, p_magic=0.05,
                 plain_use = False):
        # 4× dataset augmentation by variant
        self.orig_paths = filepaths
        self.paths = filepaths * 4
        self.ds = ds
        self.num_slices = num_slices
        self.noise_std = noise_std
        self.flip_prob = flip_prob
        self.plain_use = plain_use
        
        # MaskFunc pool
        self.mask_pool = [
            (RandomMaskFunc(
                center_fractions=[0.04, 0.08],
                accelerations=accelerations,
                allow_any_combination=True), p_random),
            (EquispacedMaskFractionFunc(
                center_fractions=center_fractions,
                accelerations=accelerations,
                allow_any_combination=True), p_equi),
            (MagicMaskFunc(
                center_fractions=center_fractions,
                accelerations=accelerations,
                allow_any_combination=True), p_magic),
        ]
        cum = 0
        self.mask_pool = [(mf, (cum := cum + prob)) for mf, prob in self.mask_pool]
        
        # fixed padding size calc
        self.max_h = self.max_w = 0
        for path in self.paths:
            with h5py.File(path, 'r') as f:
                arr = f[self.ds][()]
            if arr.shape[-1] <= 32:
                arr = np.moveaxis(arr, -1, 1)
            _, _, Ky, Kx = arr.shape
            self.max_h = max(self.max_h, Ky)
            self.max_w = max(self.max_w, Kx)

    def __len__(self):
        return len(self.paths)
    
    def _sample_mask(self, shape):
        r = random.random()
        for mf, thr in self.mask_pool:
            if r <= thr:
                return mf(shape)
        return self.mask_pool[-1][0](shape)

    def __getitem__(self, idx):

        # if plain_use is True, return original k-space without augmentation
        if self.plain_use:
            path = self.orig_paths[idx // 4]
            with h5py.File(path, 'r') as f:
                arr = f[self.ds][()]
                maskk = f['mask'][()]
            if arr.shape[-1] <= 32:
                arr = np.moveaxis(arr, -1, 1)
                maskk = np.moveaxis(maskk, -1, 1)
            k2 = torch.from_numpy(arr * maskk)
            S, C, Ky, Kx = k2.shape
            idxs = [0, S//4, S//2, S-1]
            k_slices = k2[idxs]
            mag = torch.abs(k_slices)
            rss = torch.sqrt(torch.sum(mag**2, dim=1))
            rss = (rss - rss.mean()) / (rss.std() + 1e-8)
            pad_h = self.max_h - Ky
            pad_w = self.max_w - Kx
            x = F.pad(rss, (0, pad_w, 0, pad_h), value=0.0)
            y = label_from_name(os.path.basename(path))
            return x.float(), torch.tensor(y, dtype=torch.long)
        
        # select file and variant
        base_idx = idx // 4
        variant = idx % 4
        path = self.orig_paths[base_idx]
        
        # load k-space and file mask
        with h5py.File(path, 'r') as f:
            arr = f[self.ds][()]
            maskk = f['mask'][()]
        if arr.shape[-1] <= 32:
            arr = np.moveaxis(arr, -1, 1)
            maskk = np.moveaxis(maskk, -1, 1)
        k1 = torch.from_numpy(arr)
        k2 = torch.from_numpy(arr * maskk)
        
        # slice indices
        S, C, Ky, Kx = k2.shape
        idxs = [0, S//4, S//2, S-1]
        
        # new mask sampling
        mask_vol, _ = self._sample_mask(k1.shape)
        k1_slices = (k1 * mask_vol.to(k1.dtype))[idxs]
        
        # file-mask variants
        if variant == 0:
            final = k1_slices
        else:
            k_tmp = k2.clone()
            if variant == 1:
                k_tmp = torch.flip(k_tmp, dims=[-1])
                k_tmp += torch.randn_like(k_tmp) * self.noise_std * torch.std(torch.abs(k_tmp))
            elif variant == 2:
                k_tmp = torch.flip(k_tmp, dims=[-2])
                k_tmp += torch.randn_like(k_tmp) * self.noise_std * torch.std(torch.abs(k_tmp))
            # variant 3: original k2
            final = k_tmp[idxs]
        k_slices = final

        # RSS and normalize
        mag = torch.abs(k_slices)
        rss = torch.sqrt(torch.sum(mag**2, dim=1))
        rss = (rss - rss.mean()) / (rss.std() + 1e-8)

        # padding
        pad_h = self.max_h - Ky
        pad_w = self.max_w - Kx
        x = F.pad(rss, (0, pad_w, 0, pad_h), value=0.0)
        y = label_from_name(os.path.basename(path))
        return x.float(), torch.tensor(y, dtype=torch.long)

# ─── model definitions ───────────────────────────────────────
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.skip = nn.Identity()
        if stride!=1 or in_channels!=out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        out = self.conv_block(x)
        return self.relu(out + self.skip(x))

class ResSimpleCNN(nn.Module):
    def __init__(self, in_channels=4, num_classes=2):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, padding=1, bias=False),
            nn.BatchNorm2d(16), nn.ReLU(inplace=True)
        )
        self.layer1 = nn.Sequential(ResidualBlock(16,16,1), nn.MaxPool2d(2))
        self.layer2 = ResidualBlock(16,32,2)
        self.layer3 = ResidualBlock(32,64,2)
        self.pool = nn.AdaptiveAvgPool2d((1,1))
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(64,64), nn.ReLU(inplace=True),
            nn.Dropout(0.5), nn.Linear(64,num_classes)
        )
    def forward(self, x):
        x = self.stem(x); x = self.layer1(x)
        x = self.layer2(x); x = self.layer3(x)
        x = self.pool(x)
        return self.classifier(x)

# ─── training & validation ─────────────────────────────────
def train_one_epoch(model, loader, criterion, optimizer, device, scaler):
    model.train()
    running_loss = correct = total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        # with torch.cuda.amp.autocast():
        with autocast():
            out = model(x)
            loss = criterion(out, y)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        running_loss += loss.item() * y.size(0)
        preds = out.argmax(1)
        correct += (preds==y).sum().item()
        total += y.size(0)
    return running_loss/total, correct/total

def validate(model, loader, criterion, device):
    model.eval()
    running_loss = correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out, y)
            running_loss += loss.item() * y.size(0)
            preds = out.argmax(1)
            correct += (preds==y).sum().item()
            total += y.size(0)
    return running_loss/total, correct/total

# ─── main ───────────────────────────────────────────────────
def main(args):
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"▶ Using device: {device}")

    train_paths = sorted(glob.glob(os.path.join(args.data, '*.h5')))
    print(f"▶ Found {len(train_paths)} train files")
    

    dataset = KSpaceDataset(train_paths)
    val_size = int(len(dataset)*0.2)
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])
    print(f"▶ Train size: {len(train_ds)}, Val size: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True)

    model     = ResSimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    # scaler = GradScaler()
    scaler    = GradScaler(device_type = 'cuda')
    start = time.time()

    best_val_loss = float('inf')
    for epoch in range(1, args.epochs+1):
        epoch_start = time.time()
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        elapsed = time.time() - epoch_start
        print(f"[Epoch {epoch}/{args.epochs}] {elapsed:.1f}s "
              f"Train Loss: {tr_loss:.6f}, Acc: {tr_acc:.6f} "
              f"Val Loss: {val_loss:.6f}, Acc: {val_acc:.6f}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), args.out)
            print(f"→ Saved best model (Val Loss: {val_loss:.6f})")

    print(f"▶ Training done in {((time.time()-start)/60):.6f} min")

    if args.test_data:
        test_paths = sorted(glob.glob(os.path.join(args.test_data, '*.h5')))
        test_ds = KSpaceDataset(test_paths, plain_use=True)  # Use plain k-space without augmentation
        test_loader = DataLoader(test_ds, batch_size=args.batch_size,
                                 shuffle=False, num_workers=args.num_workers, pin_memory=True)
        model.load_state_dict(torch.load(args.out, map_location=device))
        test_loss, test_acc = validate(model, test_loader, criterion, device)
        print(f"▶ Test Acc: {test_acc:.6f}, Loss: {test_loss:.6f}")

        model.eval()
        correct, total = 0, 0

        # 2) 배치별로 추론 → 예측/정답 비교
        with torch.no_grad():
            for x, y in test_loader:            # (x: [B,4,H,W], y: [B])
                x, y = x.to(device), y.to(device)
                out  = model(x)                 # logits
                preds = out.argmax(dim=1)       # 예측 클래스
                correct += (preds == y).sum().item()
                total   += y.size(0)

        # 3) 결과 출력
        print(f"Test with augmented => Accuracy: {correct}/{total} = {correct/total:.4f}")


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True)
    parser.add_argument('--test-data', required=True)
    parser.add_argument('--out', default='model_v5.pth')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_workers', type=int, default=4)
    args = parser.parse_args()
    main(args)
