#!/usr/bin/env python3
"""
Repartition val/train by moving all val/{kspace,image} → train/{kspace,image},
then (per class group) select every Nth pair (default step=11) and move them back to val.

Grouping rule:
  basename = "<class>_<index>", e.g., "brain_acc4_17" → class="brain_acc4", index=17

Dry-run prints detailed verification (per-group selections and predicted final counts).
"""

import argparse, shutil, sys, re
from pathlib import Path
from collections import defaultdict

# -------- helpers --------

def require_subdirs(root: Path):
    ks = root / "kspace"
    im = root / "image"
    if not ks.exists() or not im.exists():
        print(f"[ERROR] {root} must contain both kspace/ and image/ subfolders.")
        sys.exit(1)
    return ks, im

def natural_key(s: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', s)]

def split_class_and_index(stem: str):
    """
    Split "brain_acc4_17" -> ("brain_acc4", 17)
    Returns (class_name, index_int) or (None, None) if it doesn't match.
    """
    m = re.match(r"^(.*)_(\d+)$", stem)
    if not m:
        return None, None
    return m.group(1), int(m.group(2))

def move_all(src_dir: Path, dst_dir: Path, apply: bool):
    dst_dir.mkdir(parents=True, exist_ok=True)
    moved = 0
    for f in sorted(src_dir.glob("*")):
        if not f.is_file():
            continue
        target = dst_dir / f.name
        if apply:
            if target.exists():
                raise FileExistsError(f"[STOP] File already exists: {target}")
            shutil.move(str(f), str(target))
        moved += 1
    return moved

# -------- main --------

def main(train_root: Path, val_root: Path, step: int, apply: bool):
    train_k, train_i = require_subdirs(train_root)
    val_k,   val_i   = require_subdirs(val_root)

    print("== Settings ==")
    print(f"train root: {train_root}")
    print(f"val root  : {val_root}")
    print(f"step      : {step} (per-class indices: 0, {step}, {2*step}, ...)")
    print(f"mode      : {'APPLY' if apply else 'DRY-RUN'}\n")

    # 0) current counts
    cur_train_k = len(list(train_k.glob("*")))
    cur_train_i = len(list(train_i.glob("*")))
    cur_val_k   = len(list(val_k.glob("*")))
    cur_val_i   = len(list(val_i.glob("*")))

    # 1) move val → train (or simulate on dry-run)
    print("[1/4] Moving val → train")
    n_k = move_all(val_k, train_k, apply)
    n_i = move_all(val_i, train_i, apply)
    print(f" - would move kspace: {n_k}")
    print(f" - would move image : {n_i}")

    # 2) collect maps after the (possibly simulated) merge
    print("\n[2/4] Collecting common basenames in train")
    k_map = {p.stem: p for p in train_k.glob("*") if p.is_file()}
    i_map = {p.stem: p for p in train_i.glob("*") if p.is_file()}
    common_stems = sorted(set(k_map.keys()) & set(i_map.keys()), key=natural_key)
    if not common_stems:
        print("[ERROR] No common basenames found.")
        sys.exit(1)

    # 3) group by class and pick every Nth within each group
    print("\n[3/4] Grouping by class and selecting every Nth within each class")
    groups = defaultdict(list)  # class_name -> [stem,...]
    skipped_bad = []
    for stem in common_stems:
        cls, _ = split_class_and_index(stem)
        if cls is None:
            skipped_bad.append(stem)
            continue
        groups[cls].append(stem)

    if skipped_bad:
        print(f" - WARNING: {len(skipped_bad)} names didn't match <class>_<index> and were ignored (e.g., {skipped_bad[:5]})")

    picks = []
    per_group_report = []
    for cls, stems in groups.items():
        stems_sorted = sorted(stems, key=natural_key)
        cls_picks = [stems_sorted[i] for i in range(0, len(stems_sorted), step)]
        picks.extend(cls_picks)
        per_group_report.append((cls, len(stems_sorted), len(cls_picks), cls_picks[:5]))

    # report per-group selection
    total_pairs = sum(len(stems) for stems in groups.values())
    print(f" - total train pairs (common): {total_pairs}")
    print(f" - total selected pairs      : {len(picks)}")
    for cls, total_in_cls, picked_in_cls, preview in sorted(per_group_report, key=lambda x: x[0]):
        print(f"   * {cls}: total={total_in_cls}, selected={picked_in_cls}, preview={preview}")

    # 4) move selected pairs train → val
    print("\n[4/4] Moving selected pairs train → val")
    moved_pairs = 0
    for stem in picks:
        src_k = k_map[stem]
        src_i = i_map[stem]
        dst_k = val_k / src_k.name
        dst_i = val_i / src_i.name

        if apply:
            # After apply=True, val is empty because we moved it to train already.
            # Perform existence check and move.
            if dst_k.exists() or dst_i.exists():
                raise FileExistsError(f"[STOP] Already exists in val: {dst_k} or {dst_i}")
            shutil.move(str(src_k), str(dst_k))
            shutil.move(str(src_i), str(dst_i))
        moved_pairs += 1

    # predicted final counts
    predicted_val_pairs = len(picks)
    predicted_train_pairs = total_pairs - predicted_val_pairs

    print(f"\n - moved back pairs: {moved_pairs}")
    print("\n= Predicted final counts (after full procedure) =")
    print(f"  train/kspace: {predicted_train_pairs}")
    print(f"  train/image : {predicted_train_pairs}")
    print(f"  val/kspace  : {predicted_val_pairs}")
    print(f"  val/image   : {predicted_val_pairs}")

    if not apply:
        print("\nDRY-RUN: No files were changed. Use --apply to execute the moves.")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Per-class every-Nth repartition of MRI pairs.")
    ap.add_argument("--train", type=Path, required=True, help="train root (must contain kspace/ and image/)")
    ap.add_argument("--val",   type=Path, required=True, help="val root (must contain kspace/ and image/)")
    ap.add_argument("--step",  type=int, default=11, help="interval per class (default: 11)")
    ap.add_argument("--apply", action="store_true", help="actually move files (default: dry-run)")
    args = ap.parse_args()
    main(args.train, args.val, args.step, args.apply)
