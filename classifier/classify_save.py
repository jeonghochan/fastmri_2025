#!/usr/bin/env python3
"""
save_classified_kspace.py — k-space 파일을 분류하여 별도 폴더에 저장

사용법 예시:
  python save_classified_kspace.py \
    --model-path model.pth \
    --data-dir Data/leaderboard/acc4/kspace \
    --out-dir classified_kspace \
    --batch-size 8 \
    --num-workers 4
"""
import os
import glob
import argparse
import shutil
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import h5py
import numpy as np
from pathlib import Path

# ─── CNN 모델 정의 ──────────────────────────────────────────────────────────────
class ResidualBlock(torch.nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.skip = torch.nn.Identity()
        if stride != 1 or in_ch != out_ch:
            self.skip = torch.nn.Sequential(
                torch.nn.Conv2d(in_ch, out_ch, 1, stride, bias=False),
                torch.nn.BatchNorm2d(out_ch),
            )
        self.conv_block = torch.nn.Sequential(
            torch.nn.Conv2d(in_ch, out_ch, 3, stride, padding=1, bias=False),
            torch.nn.BatchNorm2d(out_ch),
            torch.nn.ReLU(True),
            torch.nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            torch.nn.BatchNorm2d(out_ch),
        )
        self.relu = torch.nn.ReLU(True)
    def forward(self, x):
        return self.relu(self.conv_block(x) + self.skip(x))

class ResSimpleCNN(torch.nn.Module):
    def __init__(self, in_ch=4, num_classes=2):
        super().__init__()
        self.stem = torch.nn.Sequential(
            torch.nn.Conv2d(in_ch, 16, 3, padding=1, bias=False),
            torch.nn.BatchNorm2d(16),
            torch.nn.ReLU(True),
        )
        self.layer1 = torch.nn.Sequential(ResidualBlock(16,16,1), torch.nn.MaxPool2d(2))
        self.layer2 = ResidualBlock(16,32,2)
        self.layer3 = ResidualBlock(32,64,2)
        self.pool   = torch.nn.AdaptiveAvgPool2d((1,1))
        self.classifier = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(64,64),
            torch.nn.ReLU(True),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(64, num_classes),
        )
    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.pool(x)
        return self.classifier(x)

class KSpaceDataset(Dataset):
    def __init__(self, filepaths, ds='kspace', num_slices=4):
        self.paths = filepaths
        self.ds = ds
        self.num_slices = num_slices
        self.max_h = 0
        self.max_w = 0
        # Padding 크기 산출
        for p in self.paths:
            with h5py.File(p, 'r') as f:
                arr = f[self.ds][()]
            if arr.ndim == 3:
                arr = np.expand_dims(arr, axis=1)
            if arr.shape[-1] <= 32:
                arr = np.moveaxis(arr, -1, 1)
            _, _, Ky, Kx = arr.shape
            self.max_h = max(self.max_h, Ky)
            self.max_w = max(self.max_w, Kx)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        with h5py.File(path, 'r') as f:
            k = f[self.ds][()]
            mask = f['mask'][()] if 'mask' in f else None
        if k.ndim == 3:
            k = np.expand_dims(k, axis=1)
        if k.shape[-1] <= 32:
            k = np.moveaxis(k, -1, 1)
            if mask is not None and mask.ndim == 3:
                mask = np.moveaxis(mask, -1, 0)
        if mask is not None:
            k = k * mask
        t = torch.from_numpy(np.abs(k))
        # Slices 선택 및 RSS 계산
        S, C, Ky, Kx = t.shape
        idxs = [0, S//4, S//2, S-1][:self.num_slices]
        slices = t[idxs]
        rss = torch.sqrt((slices**2).sum(dim=1))
        # 정규화
        rss = (rss - rss.mean()) / (rss.std() + 1e-8)
        # 패딩
        pad_h = self.max_h - Ky
        pad_w = self.max_w - Kx
        x = F.pad(rss, (0, pad_w, 0, pad_h), 'constant', 0.0)
        # 채널 차원으로 사용
        return x, Path(path).name


def main():
    parser = argparse.ArgumentParser(description='Classify and save k-space files')
    parser.add_argument('--model-path', required=True)
    parser.add_argument('--data-dir',   required=True)
    parser.add_argument('--out-dir',    required=True)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--acc', type =str, required=True, help='acc4 or acc8')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ResSimpleCNN().to(device)
    state = torch.load(args.model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    paths = sorted(glob.glob(os.path.join(args.data_dir, '*.h5')))
    ds = KSpaceDataset(paths)
    loader = DataLoader(ds, batch_size=args.batch_size,
                        shuffle=False, num_workers=args.num_workers)

    acc = args.acc
    root = Path(args.out_dir)
    brain_dir = root / 'brain' / acc/ 'kspace'
    knee_dir  = root / 'knee' / acc / 'kspace'
    brain_dir.mkdir(parents=True, exist_ok=True)
    knee_dir.mkdir(parents=True, exist_ok=True)

    softmax = torch.nn.Softmax(dim=1)
    with torch.no_grad():
        for batch_x, batch_names in loader:
            batch_x = batch_x.to(device)  # shape: [B, C=4, H, W]
            logits = model(batch_x)
            probs = softmax(logits)[:,1].cpu().numpy()
            for name, p in zip(batch_names, probs):
                src = Path(args.data_dir) / name
                dst = brain_dir if p >= 0.5 else knee_dir
                shutil.copy(src, dst / name)
                print(f"{name}: {'brain' if p>=0.5 else 'knee'} ({p:.3f})")

if __name__ == '__main__':
    main()

