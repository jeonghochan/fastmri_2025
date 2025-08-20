#!/usr/bin/env python3
"""
infer.py — 단일 .h5 파일의 k-space & mask 구조 확인 및 시각화
사용법:
  python infer.py --file path/to/brain_test1.h5
"""

import argparse
import h5py
import numpy as np
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', required=True, help='검토할 .h5 파일 경로')
    args = parser.parse_args()

    # 1) HDF5 키·형태 출력
    with h5py.File(args.file, 'r') as f:
        print("=== HDF5 keys and shapes ===")
        for k in f.keys():
            print(f"  {k}: {f[k].shape}")
        k = f['kspace'][()]    # 복소수 배열, shape (S, C, Ky, Kx)
        m = f['mask'][()]      # 언더샘플링 마스크, shape (Kx,)

    # 2) 차원 정리
    import numpy as _np
    if k.shape[-1] <= 32:
        k = _np.moveaxis(k, -1, 1)
    S, C, Ky, Kx = k.shape
    print(f"\nvolume shape: S={S}, C={C}, Ky={Ky}, Kx={Kx}")
    print(f"mask shape: {m.shape}")

    # 3) 중간 슬라이스 하나 선택
    idx = S // 2
    k_slice = k[idx]        # (C, Ky, Kx)

    # 4) magnitude 및 RSS
    mag      = np.abs(k_slice)               # (C, Ky, Kx)
    masked   = mag * m[np.newaxis, np.newaxis, :]  # 브로드캐스트로 (C, Ky, Kx)
    rss_full = np.sqrt((mag**2).sum(axis=0))       # (Ky, Kx)
    rss_mask = np.sqrt((masked**2).sum(axis=0))

    # 5) 마스크를 2D 형태로 만들기 (Ky×Kx)
    mask2d = np.tile(m, (Ky, 1))                  # (Ky, Kx)

    # 6) 시각화
    fig, axes = plt.subplots(2, 2, figsize=(8,8))
    axes[0,0].imshow(rss_full, cmap='gray')
    axes[0,0].set_title('RSS full k-space')
    axes[0,1].imshow(mask2d, cmap='gray', aspect='auto')
    axes[0,1].set_title('mask (tiled 2D)')
    axes[1,0].imshow(rss_mask, cmap='gray')
    axes[1,0].set_title('RSS masked k-space')
    diff = np.abs(rss_full - rss_mask)
    axes[1,1].imshow(diff, cmap='hot')
    axes[1,1].set_title('difference')
    for ax in axes.ravel():
        ax.axis('off')
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()
