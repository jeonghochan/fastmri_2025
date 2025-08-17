#!/usr/bin/env python3
"""
Slice-wise MASK augmentation verifier (scheduler-free, presence-only).
- kspace: (S, C, H, W)
- mask  : (W,) if exists in .h5
- 판단: 원본 mask(1D) vs 증강 mask(1D)가 다르면 augmentation APPLIED
"""

import argparse
from pathlib import Path
import os, sys
import h5py
import torch

# Add project utils to path
if os.getcwd() + '/utils/model/' not in sys.path:
    sys.path.insert(1, os.getcwd() + '/utils/model/')
if os.getcwd() + '/utils/augment/' not in sys.path:
    sys.path.insert(1, os.getcwd() + '/utils/augment/')

from utils.augment.kspace_augment import KSpaceAugmentor

# ----------------- helpers -----------------

def to_ri(x_complex: torch.Tensor) -> torch.Tensor:
    return torch.view_as_real(x_complex)  # [C,H,W,2]

def to_complex(x_ri: torch.Tensor) -> torch.Tensor:
    return torch.view_as_complex(x_ri)    # [C,H,W]

def infer_mask1d_from_kspace(ks_ri: torch.Tensor, eps: float = 1e-10) -> torch.Tensor:
    """
    ks_ri: [C,H,W,2]
    열(=readout, W축)별 평균 에너지를 보고 1D 마스크 추론: >eps → 1, else 0
    """
    kc = to_complex(ks_ri)                         # [C,H,W]
    col_energy = torch.abs(kc).mean(dim=(0, 1))    # [W]
    return (col_energy > eps).float()              # [W]

def mask2d_to_1d(m2d: torch.Tensor, W: int) -> torch.Tensor:
    """
    2D 마스크(H,W)를 열 방향으로 reduce(any) → 1D(W,)
    """
    m2d = m2d.float()
    if m2d.ndim != 2:
        raise ValueError(f"Expected 2D mask, got {tuple(m2d.shape)}")
    if m2d.shape[1] != W and m2d.shape[1] == 1:
        m2d = m2d.expand(m2d.shape[0], W)
    return (m2d > 0.5).any(dim=0).float()          # [W]

def normalize_aug_mask_to_1d(aug_mask: torch.Tensor, H: int, W: int) -> torch.Tensor:
    """
    aug_mask를 어떤 모양이든 1D(W,)로 정규화:
    허용: (W,), (H,), (H,1), (1,W), (H,W), (W,H), (C,H,W)
    """
    m = aug_mask.float()
    while m.ndim > 2 and 1 in m.shape:
        m = m.squeeze()
    if m.ndim == 1:
        if m.shape[0] == W:
            return m
        raise ValueError(f"1D aug_mask length {m.shape[0]} != W={W}")
    if m.ndim == 2:
        Hm, Wm = m.shape
        if (Hm, Wm) == (H, W):
            return mask2d_to_1d(m, W)
        if (Hm, Wm) == (W, H):
            return mask2d_to_1d(m.t(), W)
        if (Hm, Wm) == (H, 1):
            return mask2d_to_1d(m.expand(H, W), W)
        if (Hm, Wm) == (1, W):
            return mask2d_to_1d(m.expand(H, W), W)
        if Wm == W:
            return mask2d_to_1d(m, W)
        if Hm == W:
            return mask2d_to_1d(m.t(), W)
        raise ValueError(f"Unexpected 2D aug_mask shape {tuple(m.shape)}")
    if m.ndim == 3:
        if m.shape[-1] != W:
            raise ValueError(f"3D aug_mask last dim {m.shape[-1]} != W={W}")
        return (m > 0.5).any(dim=(0, 1)).float()
    raise ValueError(f"Unexpected aug_mask ndim={m.ndim}, shape={tuple(m.shape)}")

# ----------------- data iterator -----------------

def iter_slices_from_file(file_path: Path, max_slices: int = None):
    """
    Yields:
      'filename', 'slice_idx', 'kspace_ri'([C,H,W,2]), 'orig_mask1d'([W] or None), 'H','W'
    """
    with h5py.File(str(file_path), 'r') as f:
        ks_np = f['kspace'][...]            # (S,C,H,W)
        S, C, H, W = ks_np.shape
        mask_np = f['mask'][...] if 'mask' in f else None  # (W,)

        # 파일 단위 shape 출력
        print(f"[FILE] {file_path.name} | kspace: {ks_np.shape} (S,C,H,W) "
              f"| mask: {None if mask_np is None else mask_np.shape}")
        print("---------------------------------------------------------")

        n = S if max_slices is None else min(S, max_slices)
        for s in range(n):
            ks_c = torch.from_numpy(ks_np[s]).to(torch.complex64)  # [C,H,W]
            ks_ri = to_ri(ks_c)                                    # [C,H,W,2]
            orig_mask1d = torch.from_numpy(mask_np).float() if mask_np is not None else None

            # 슬라이스 단위 입력 shape 출력
            print(f"  [SLICE {s:03d} INPUT] ks_c: {tuple(ks_c.shape)} (C,H,W) "
                  f"| ks_ri: {tuple(ks_ri.shape)} (C,H,W,2) "
                  f"| orig_mask1d: {None if orig_mask1d is None else tuple(orig_mask1d.shape)}")

            yield {
                "filename": file_path.name,
                "slice_idx": s,
                "kspace_ri": ks_ri,
                "orig_mask1d": orig_mask1d,  # [W] or None
                "H": H, "W": W,
                "ks_c" : ks_c,
            }

# ----------------- main -----------------

def main():
    ap = argparse.ArgumentParser(description="Verify MASK augmentation presence (scheduler-free)")
    ap.add_argument("--data-path", type=Path, required=True, help="Folder containing *.h5 kspace files")
    ap.add_argument("--config", type=str, default="configs/kspace_mask_augment.yaml", help="Augmentor config yaml")
    ap.add_argument("--mode", type=str, default="rand", choices=["rand", "equi", "magic", "auto"],
                    help="Force a specific mask aug or use YAML base probs.")
    ap.add_argument("--max-files", type=int, default=1)
    ap.add_argument("--max-slices", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--visualize", action="store_true", help="Show slice-wise original vs augmented k-space")

    args = ap.parse_args()
    print("Loading MASK augmentation verification tool (NO scheduler)…")

    # YAML: base probs만 사용 (스케줄 파라미터는 무시)
    import yaml
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f) or {}

    # augmentor 생성 (flip/shift 비활성화, 스케줄 인자 전달 안함)
    base_kwargs = dict(
        prob_hflip=0.0, prob_vflip=0.0, prob_shift=0.0, max_shift_fraction=0.0,
        seed=args.seed,
        base_prob_randmask = cfg.get('base_prob_randmask', cfg.get('prob_randmask', 0.0)),
        base_prob_equimask = cfg.get('base_prob_equimask', cfg.get('prob_equimask', 0.0)),
        base_prob_magicmask = cfg.get('base_prob_magicmask', cfg.get('prob_magicmask', 0.0)),
    )
    augmentor = KSpaceAugmentor(**base_kwargs)

    # 내부 스케줄 멤버가 있더라도 강제로 무력화
    for attr, val in [
        ("aug_schedule", "constant"),
        ("aug_delay", 0),
        ("max_epochs", 1),
        ("current_epoch", 0),
        ("current_strength", 1.0),
    ]:
        try: setattr(augmentor, attr, val)
        except Exception: pass
    if hasattr(augmentor, "set_epoch"):
        try: augmentor.set_epoch(0)
        except Exception: pass

    # --mode로 하나만 100% 강제
    if args.mode != "auto":
        augmentor.base_prob_randmask  = 1.0 if args.mode == "rand"  else 0.0
        augmentor.base_prob_equimask  = 1.0 if args.mode == "equi"  else 0.0
        if hasattr(augmentor, "base_prob_magicmask"):
            augmentor.base_prob_magicmask = 1.0 if args.mode == "magic" else 0.0
        elif hasattr(augmentor, "base_prob_magic"):
            augmentor.base_prob_magic = 1.0 if args.mode == "magic" else 0.0

    magic_show = getattr(augmentor, "base_prob_magicmask",
                    getattr(augmentor, "base_prob_magic", 0.0))
    print(f"Mask probs (mode={args.mode}): rand={augmentor.base_prob_randmask}, "
          f"equi={augmentor.base_prob_equimask}, magic={magic_show}")

    #file number check
    files = sorted(list(args.data_path.glob("*.h5")))[:args.max_files]
    print(f"Found {len(files)} files.")

    total_slices = 0
    applied_slices = 0

    for fpath in files:
        print(f"\n== File: {fpath.name} ==")
        for sample in iter_slices_from_file(fpath, args.max_slices):
            total_slices += 1
            fname = sample["filename"]
            sidx  = sample["slice_idx"]
            ks_ri = sample["kspace_ri"]      # [C,H,W,2]
            orig_m1d = sample["orig_mask1d"] # [W] or None
            H, W = sample["H"], sample["W"]
            ks_c = sample["ks_c"]            # [C,H,W]

            target = torch.zeros(H, W)  # [H,W]
            print(f"  [SLICE {sidx:03d} INPUT] target(img): {tuple(target.shape)} (H,W)")

            ret = augmentor.augment_kspace(ks_ri, target, fname, sidx)
            if isinstance(ret, tuple) and len(ret) == 3:
                aug_ks_ri, _, aug_mask = ret
            elif isinstance(ret, tuple) and len(ret) == 2:
                aug_ks_ri, _ = ret
                aug_mask = getattr(augmentor, "aug_mask", None)
            else:
                raise RuntimeError("augment_kspace must return 2-tuple or 3-tuple")

            #visualization
            if orig_m1d is not None:
                mask2d = orig_m1d[None, None, :].expand_as(ks_c)  # [C,H,W]
                ks_orig = ks_c * mask2d
            else:
                ks_orig = ks_c

            # 증강된 k-space
            ks_aug = to_complex(aug_ks_ri)  # [C,H,W]


            print(f"  [SLICE {sidx:03d} AUG ] aug_kspace: {tuple(aug_ks_ri.shape)} (C,H,W,2) "
                  f"| raw aug_mask: {None if aug_mask is None else tuple(aug_mask.shape)}")

            if aug_mask is not None:
                try:
                    aug_m1d = normalize_aug_mask_to_1d(aug_mask, H, W)  # [W]
                    print(f"  [SLICE {sidx:03d} AUG ] aug_mask1d(norm): {tuple(aug_m1d.shape)} (W)")
                except Exception as e:
                    print(f"  [SLICE {sidx:03d} AUG ] aug_mask normalize failed ({e}), infer from kspace")
                    aug_m1d = infer_mask1d_from_kspace(aug_ks_ri)
                    print(f"  [SLICE {sidx:03d} AUG ] aug_mask1d(inferred): {tuple(aug_m1d.shape)} (W)")
            else:
                aug_m1d = infer_mask1d_from_kspace(aug_ks_ri)
                print(f"  [SLICE {sidx:03d} AUG ] aug_mask1d(inferred): {tuple(aug_m1d.shape)} (W)")

            # 원본 1D 마스크가 없으면 원본 k-space로부터 추론
            if orig_m1d is None:
                orig_m1d = infer_mask1d_from_kspace(ks_ri)
                print(f"  [SLICE {sidx:03d} INPUT] orig_mask1d(inferred): {tuple(orig_m1d.shape)} (W)")

            if args.visualize:
                # show_kspace_comparison(ks_orig.cpu(), ks_aug.cpu(), sidx)
                show_kspace_comparison(
                                        ks_orig, ks_aug, sidx,
                                        aug_m1d=aug_m1d,      # normalize_aug_mask_to_1d(...) 결과
                                        orig_m1d=orig_m1d,    # 원본 1D 마스크 (없으면 None)
                                        alpha_aug=0.30,
                                        alpha_orig=0.50
                                    )

            
            # 적용 여부 판단: 1D 마스크가 다르면 APPLIED
            diff = torch.abs(orig_m1d.float() - aug_m1d.float())
            changed = bool(torch.any(diff > 1e-5).item())
            changed_count = int(diff.sum().item())

            if changed:
                applied_slices += 1
                print(f"  [SLICE {sidx:03d} RES ] APPLIED  (changed_cols={changed_count})")
            else:
                print(f"  [SLICE {sidx:03d} RES ] NOT APPLIED")

    print("\n==============================")
    print("MASK AUGMENTATION PRESENCE")
    print("==============================")
    print(f"Total slices checked : {total_slices}")
    print(f"Slices APPLIED       : {applied_slices}/{total_slices}")


import matplotlib.pyplot as plt
import numpy as np

# def show_kspace_comparison(ks_orig, ks_aug, sidx):
#     """
#     ks_orig, ks_aug: torch.complex64 [C,H,W] or numpy
#     """
#     if isinstance(ks_orig, torch.Tensor):
#         ks_orig = ks_orig.numpy()
#     if isinstance(ks_aug, torch.Tensor):
#         ks_aug = ks_aug.numpy()

#     # coil combine (RSS in k-space magnitude)
#     img_orig = np.sqrt(np.sum(np.abs(ks_orig)**2, axis=0))
#     img_aug  = np.sqrt(np.sum(np.abs(ks_aug )**2, axis=0))

#     # log scale for better visibility
#     img_orig = np.log1p(img_orig)
#     img_aug  = np.log1p(img_aug)

#     fig, axes = plt.subplots(1, 2, figsize=(10,5))
#     axes[0].imshow(img_orig, cmap="gray")
#     axes[0].set_title(f"Original Masked k-space (slice {sidx})")
#     axes[1].imshow(img_aug, cmap="gray")
#     axes[1].set_title(f"Augmented Masked k-space (slice {sidx})")
#     plt.show()


# # mask 차이를 보는 function
# import numpy as np
# import matplotlib.pyplot as plt

# def show_kspace_comparison(
#     ks_orig, ks_aug, sidx,
#     aug_m1d=None,          # [W] 증강 마스크(0/1) - 권장: normalize_aug_mask_to_1d 결과 전달
#     orig_m1d=None,         # [W] 원본 마스크(0/1) - 주면 '변경 열'을 빨간색으로 표시
#     alpha_aug=0.45,        # 파란 오버레이 투명도
#     alpha_orig=0.45        # 빨간 오버레이 투명도
# ):
#     """
#     ks_orig, ks_aug: complex k-space [C,H,W] (torch.Tensor or np.ndarray)
#     """
#     # ---- to numpy & RSS + log ----
#     if hasattr(ks_orig, "detach"): ks_orig = ks_orig.detach().cpu().numpy()
#     if hasattr(ks_aug,  "detach"): ks_aug  = ks_aug.detach().cpu().numpy()
#     img_orig = np.sqrt(np.sum(np.abs(ks_orig)**2, axis=0))
#     img_aug  = np.sqrt(np.sum(np.abs(ks_aug )**2, axis=0))
#     img_orig = np.log1p(img_orig)
#     img_aug  = np.log1p(img_aug)

#     H, W = img_orig.shape

#     # ---- 오버레이용 마스크 준비 ----
#     def _to_np(m):
#         if m is None: return None
#         if hasattr(m, "detach"): m = m.detach().cpu().numpy()
#         return np.asarray(m).astype(np.float32)

#     aug_m1d  = _to_np(aug_m1d)
#     orig_m1d = _to_np(orig_m1d)

#     # 증강 마스크가 없으면 ks_aug에서 추론(간단히 열 평균 에너지 > 0)
#     if aug_m1d is None:
#         col_energy = np.abs(ks_aug).mean(axis=(0,1))  # [W]
#         aug_m1d = (col_energy > 1e-10).astype(np.float32)

#     # 변경 열 (orig 제공 시)
#     changed_cols = None
#     if orig_m1d is not None:
#         changed_cols = (aug_m1d.astype(np.int32) ^ orig_m1d.astype(np.int32)).astype(bool)  # [W]

#     # ---- 그리기 ----
#     fig, axes = plt.subplots(1, 2, figsize=(10, 5))

#     axes[0].imshow(img_orig, cmap="gray", aspect="auto")
#     axes[0].set_title(f"Original Masked k-space (slice {sidx})")
#     axes[0].axis("off")

#     axes[1].imshow(img_aug, cmap="gray", aspect="auto")
#     axes[1].set_title("Augmented Masked k-space (overlay)")
#     axes[1].axis("off")

#     # 파란색: aug mask == 1 인 열
#     overlay_aug = np.zeros((H, W, 4), dtype=np.float32)  # RGBA
#     overlay_aug[:, aug_m1d > 0.5, :] = (0.12, 0.56, 1.00, alpha_aug)  # 파랑
#     axes[1].imshow(overlay_aug, aspect="auto")

#     # 빨간색: 원본과 달라진 열(있으면)
#     # if changed_cols is not None and changed_cols.any():
#     #     overlay_diff = np.zeros((H, W, 4), dtype=np.float32)
#     #     overlay_diff[:, changed_cols, :] = (1.00, 0.20, 0.20, alpha_diff)  # 빨강
#     #     axes[1].imshow(overlay_diff, aspect="auto")
#     overlay_orig = np.zeros((H, W, 4), dtype=np.float32)  # RGBA
#     overlay_orig[:, orig_m1d > 0.5, :] = (1.00, 0.56, 0.12, alpha_orig)  # 파랑
#     axes[0].imshow(overlay_orig, aspect = "auto")
#     plt.tight_layout()
#     plt.show()

import numpy as np
import matplotlib.pyplot as plt

def show_kspace_comparison(
    ks_orig, ks_aug, sidx,
    aug_m1d=None,          # [W] 증강 마스크(0/1)
    orig_m1d=None,         # [W] 원본 마스크(0/1)
    alpha_aug=0.45,        # 파란 오버레이(오른쪽)
    alpha_orig=0.45,       # 빨간 오버레이(왼쪽)
    alpha_diff=0.60        # 차이 오버레이(가운데/오른쪽 새 패널)
):
    # ---- to numpy & RSS + log ----
    if hasattr(ks_orig, "detach"): ks_orig = ks_orig.detach().cpu().numpy()
    if hasattr(ks_aug,  "detach"): ks_aug  = ks_aug.detach().cpu().numpy()
    img_orig = np.sqrt(np.sum(np.abs(ks_orig)**2, axis=0))
    img_aug  = np.sqrt(np.sum(np.abs(ks_aug )**2, axis=0))
    img_orig = np.log1p(img_orig)
    img_aug  = np.log1p(img_aug)
    H, W = img_orig.shape

    # 두 패널 밝기 범위를 통일
    vmin, vmax = np.percentile(np.stack([img_orig, img_aug], 0), [1, 99])

    # ---- 마스크 numpy화 ----
    def _to_np(m):
        if m is None: return None
        if hasattr(m, "detach"): m = m.detach().cpu().numpy()
        return np.asarray(m).astype(np.float32)

    aug_m1d  = _to_np(aug_m1d)
    orig_m1d = _to_np(orig_m1d)

    # 차이(XOR)
    # changed_cols = (orig_m1d.astype(np.int32) ^ orig_m1d.astype(np.int32)).astype(bool)
    changed_cols = (aug_m1d.astype(np.int32) ^ orig_m1d.astype(np.int32)).astype(bool)

    # ---- 그리기: 1x3 패널 ----
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # (0) 원본 + 빨강(원본 마스크)
    axes[0].imshow(img_orig, cmap="gray", vmin=vmin, vmax=vmax, aspect="auto", interpolation="nearest")
    overlay_orig = np.zeros((H, W, 4), dtype=np.float32)
    overlay_orig[:, orig_m1d > 0.5, :] = (1.00, 0.20, 0.20, alpha_orig)  # 빨강
    axes[0].imshow(overlay_orig, aspect="auto")
    axes[0].set_title(f"Original masked k-space (slice {sidx})")
    axes[0].axis("off")

    # (1) 증강 + 파랑(증강 마스크)
    axes[1].imshow(img_aug, cmap="gray", vmin=vmin, vmax=vmax, aspect="auto", interpolation="nearest")
    overlay_aug = np.zeros((H, W, 4), dtype=np.float32)
    overlay_aug[:, aug_m1d > 0.5, :] = (0.12, 0.56, 1.00, alpha_aug)      # 파랑
    axes[1].imshow(overlay_aug, aspect="auto")
    axes[1].set_title("Augmented masked k-space (overlay)")
    axes[1].axis("off")

    # (2) 차이만 강조(노랑) — 베이스는 증강 영상 위에 표시
    axes[2].imshow(img_aug, cmap="gray", vmin=vmin, vmax=vmax, aspect="auto", interpolation="nearest")
    overlay_diff = np.zeros((H, W, 4), dtype=np.float32)
    overlay_diff[:, changed_cols, :] = (1.00, 1.00, 0.00, alpha_diff)     # 노랑(변경 열만)
    axes[2].imshow(overlay_diff, aspect="auto")
    axes[2].set_title("Mask difference only (orig vs aug)")
    axes[2].axis("off")

    plt.tight_layout()
    plt.show()



if __name__ == "__main__":
    main()
