#!/usr/bin/env python3
"""
eval_classify_model_kspace.py — 학습된 ResSimpleCNN으로 다른 .h5 k-space 데이터셋 평가
사용법 예시:
  python eval_classify_model_kspace.py \
    --model-path model.pth \
    --data-dir Data/leaderboard/acc4/kspace \
    --batch-size 8 \
    --num-workers 4
"""
import os
import glob
import argparse

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tabulate import tabulate
import sys

# ─── CNN 모델 정의 (train 스크립트에서 가져오기) ─────────────────
class ResidualBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.skip = torch.nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.skip = torch.nn.Sequential(
                torch.nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                torch.nn.BatchNorm2d(out_channels),
            )
        self.conv_block = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels, 3, stride, padding=1, bias=False),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(True),
            torch.nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            torch.nn.BatchNorm2d(out_channels),
        )
        self.relu = torch.nn.ReLU(True)
    def forward(self, x):
        return self.relu(self.conv_block(x) + self.skip(x))

class ResSimpleCNN(torch.nn.Module):
    def __init__(self, in_channels=4, num_classes=2):
        super().__init__()
        self.stem = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, 16, 3, padding=1, bias=False),
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

# ─── KSpaceDataset (train 때와 동일한 pipeline) ──────────────────
class KSpaceDataset(Dataset):
    def __init__(self, filepaths, ds='kspace', num_slices=3):
        import numpy as _np
        self.paths      = filepaths
        self.ds         = ds
        self.num_slices = num_slices
        # padding 계산
        self.max_h = self.max_w = 0
        for p in self.paths:
            with h5py.File(p,'r') as f:
                arr = f[self.ds][()]
            if arr.shape[-1] <= 32:
                arr = _np.moveaxis(arr, -1, 1)
            _,_,Ky,Kx = arr.shape
            self.max_h = max(self.max_h, Ky)
            self.max_w = max(self.max_w, Kx)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        import numpy as _np
        path = self.paths[idx]
        # load complex k-space volume
        with h5py.File(path, 'r') as f:
            k = f[self.ds][()]
            maskk = f['mask'][()]   

        if k.shape[-1] <= 32:
            k = _np.moveaxis(k, -1, 1)
            maskk = _np.moveaxis(maskk, -1, 0)  # (Kx,) → (Ky, Kx)

        # to PyTorch complex tensor

        t = torch.from_numpy(k*maskk)  # complex64 tensor

        # fixed slice indices
        S, C, Ky, Kx = t.shape
        idxs = [0, S//4, S//2 , S-1]
        k_slices = t[idxs] 

        # RSS: magnitude → sum of squares → root
        mag = torch.abs(k_slices)  # |z|
        rss = torch.sqrt(torch.sum(mag**2, dim=1)) 
        rss = (rss - rss.mean()) / (rss.std() + 1e-8)

        # pad to max_h, max_w
        pad_h = self.max_h - Ky
        pad_w = self.max_w - Kx
        x = F.pad(rss, (0, pad_w, 0, pad_h), 'constant', 0.0)

        # batch dim + dummy label
        return x.unsqueeze(0), torch.tensor(0)

# ─── 평가 스크립트 본문 ────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', required=True,  help='학습된 .pth 모델 파일')
    parser.add_argument('--data-dir',   required=True,  help='평가할 .h5가 있는 디렉터리')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--num-workers',type=int, default=4)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 모델 로드
    model = ResSimpleCNN(in_channels=4, num_classes=2).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device, weights_only=True))
    model.eval()

    # 데이터셋/로더
    paths = sorted(glob.glob(os.path.join(args.data_dir, '*.h5')))
    ds    = KSpaceDataset(paths)
    loader = DataLoader(ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    # 예측 및 출력
    rows = []
    softmax = torch.nn.Softmax(dim=1)
    with torch.no_grad():
        for batch_idx, (x, _) in enumerate(loader):
            start = batch_idx * args.batch_size
            end   = start + x.size(0)
            batch_files = paths[start:end]
            x = x.to(device)
            logits = model(x.squeeze(1))
            probs  = softmax(logits)[:,1].cpu().detach().numpy()
            for fname, prob in zip(batch_files, probs):
                label = 'brain' if prob>=0.5 else 'knee'
                rows.append([os.path.basename(fname), label, f"{prob:.3f}"])
    print(tabulate(rows, headers=["file","pred","P(brain)"], tablefmt="github"))
    # if label and filename is same, count as correct
    correct = sum(1 for fname, label, _ in rows if 'brain' in fname and label == 'brain' or 'knee' in fname and label == 'knee')
    total = len(rows)
    print(f"Accuracy: {correct}/{total} = {correct/total:.4f}")

if __name__=='__main__':
    #try to save the log to a file
    sys.stdout = open('model_25_v3_val.log', 'w')
    main()
