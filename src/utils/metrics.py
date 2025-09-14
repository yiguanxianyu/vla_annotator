from __future__ import annotations

from typing import List, Tuple


def mae_boundaries(pred_starts: List[int], pred_ends: List[int], gt_starts: List[int], gt_ends: List[int]) -> Tuple[float, float]:
    n = min(len(pred_starts), len(gt_starts))
    if n == 0:
        return 0.0, 0.0
    s_mae = sum(abs(int(ps) - int(gs)) for ps, gs in zip(pred_starts[:n], gt_starts[:n])) / n
    e_mae = sum(abs(int(pe) - int(ge)) for pe, ge in zip(pred_ends[:n], gt_ends[:n])) / n
    return s_mae, e_mae


def tiou(interval_a: Tuple[int, int], interval_b: Tuple[int, int]) -> float:
    a0, a1 = interval_a
    b0, b1 = interval_b
    inter = max(0, min(a1, b1) - max(a0, b0))
    union = max(a1, b1) - min(a0, b0)
    if union <= 0:
        return 0.0
    return inter / union


def tiou_at_thresholds(pred: List[Tuple[int, int]], gt: List[Tuple[int, int]], thresholds=(0.3, 0.5)) -> dict:
    res = {float(t): 0.0 for t in thresholds}
    if not pred or not gt:
        return res
    n = min(len(pred), len(gt))
    for t in thresholds:
        hits = 0
        for i in range(n):
            if tiou(pred[i], gt[i]) >= t:
                hits += 1
        res[float(t)] = hits / n
    return res

