# test_uniquemask.py
import numpy as np
import torch

from subsample import UniqueMaskFunc  # 프로젝트 경로에 맞게 import

ALLOWED_ACCS = (3, 5, 6, 7, 9)

def _center_indices(W: int, num_low: int):
    # fastMRI 스타일: left_pad = (W - num_low + 1)//2
    left = (W - num_low + 1) // 2
    right = left + num_low
    return left, right

def test_unique_mask_once(func, W=640, seed=None, msg=""):
    # UniqueMaskFunc는 RandomMaskFunc와 동일하게 (mask, num_low) 반환
    mask_1d, num_low = func((1, W, 1), seed=seed)

    # --- 형상/타입 체크 ---
    assert isinstance(mask_1d, torch.Tensor), f"{msg} not torch.Tensor"
    assert mask_1d.ndim >= 3 and mask_1d.shape[-2] == W and mask_1d.shape[-1] == 1, f"{msg} shape {mask_1d.shape}"
    assert mask_1d.dtype == torch.float32, f"{msg} dtype {mask_1d.dtype}"

    # 1D로 펼치기: (1, W, 1) → (W,)
    m = mask_1d.detach().cpu().numpy().reshape(-1)
    assert m.ndim == 1 and m.shape[0] == W, f"{msg} vec shape {m.shape}"
    assert np.all((m == 0.0) | (m == 1.0)), f"{msg} non-binary values"

    # --- 센터(ACS) 보장 검증 ---
    # num_low는 round(W * cf)여야 하며, 중앙 연속 num_low개가 1
    assert 0 <= num_low <= W, f"{msg} num_low out of range: {num_low}"
    left, right = _center_indices(W, int(num_low))
    center_ok = np.all(m[left:right] == 1.0)
    assert center_ok, f"{msg} center block not fully 1s: [{left}:{right}]"

    # --- 총 1의 개수: 기대값은 ~ round(W/acc) (확률 샘플링이므로 허용오차) ---
    ones = int(m.sum())
    assert 1 <= ones <= W, f"{msg} ones out of range: {ones}"

    # 허용 타깃 집합(각 acc에 대한 이상적 총 샘플 수)과의 근접성 검사
    targets = [int(round(W / a)) for a in ALLOWED_ACCS]  # 예: W=640 → [213,128,107,91,71]
    # 확률 샘플링 편차 허용(대략적인 표준편차 ~ 몇 줄 수준) → 타깃 대비 ±10% + 3줄 버퍼
    ok = any(abs(ones - t) <= max(3, int(0.10 * t)) for t in targets)
    assert ok, f"{msg} ones={ones} not close to any target {targets}"

    # --- 재현성(같은 seed → 같은 마스크) ---
    if seed is not None:
        mask_1d_2, _ = func((1, W, 1), seed=seed)
        assert torch.equal(mask_1d, mask_1d_2), f"{msg} not reproducible with same seed"

    return ones, num_low

def run_smoke():
    W = 640

    # 기본 설정: cf∈{0.04,0.08}, acc∈{3,5,6,7,9}
    func = UniqueMaskFunc(seed=123)

    # 단일 호출 검증
    ones, num_low = test_unique_mask_once(func, W=W, seed=2025, msg="[single]")
    print(f"[single] ones={ones}, num_low={num_low}")

    # 여러 번 호출하여 분포/범위 확인
    counts = []
    lows = []
    for s in range(10):
        o, nl = test_unique_mask_once(func, W=W, seed=1000 + s, msg=f"[loop{s}]")
        counts.append(o)
        lows.append(nl)

    implied_accs = [W / c for c in counts]
    # 가속도 근사치가 허용 범위(3~9) 안인지(라운딩/확률 변동 고려, 여유 있게 2.5~9.5)
    assert min(implied_accs) >= 2.5 and max(implied_accs) <= 9.5, f"implied_accs out of range: {implied_accs}"

    # 센터 폭도 {round(0.04*W), round(0.08*W)} 중 하나여야 함(라운딩 오차 ±1 허용)
    cf_targets = [int(round(0.04 * W)), int(round(0.08 * W))]
    assert all(any(abs(nl - ct) <= 1 for ct in cf_targets) for nl in lows), f"num_low not in {cf_targets}: {lows}"

    print("All unique-mask smoke tests PASSED.",
          f"(examples) ones={ones}, implied_accs~{[round(v,2) for v in implied_accs[:3]]}, num_low~{lows[:3]}")

if __name__ == "__main__":
    run_smoke()
