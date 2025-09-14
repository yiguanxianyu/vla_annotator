from __future__ import annotations

from typing import List, Tuple

import numpy as np


def gaussian_soft_labels(T: int, indices: List[int], sigma: float) -> np.ndarray:
    if T <= 0:
        return np.zeros((0,), dtype=np.float32)
    x = np.arange(T, dtype=np.float32)
    y = np.zeros((T,), dtype=np.float32)
    for idx in indices:
        if idx < 0 or idx >= T:
            continue
        y = np.maximum(y, np.exp(-((x - idx) ** 2) / (2 * sigma * sigma)))
    return y


def build_start_end_soft_labels(
    T: int, segments: List[Tuple[int, int]], sigma: float
) -> Tuple[np.ndarray, np.ndarray]:
    starts = [int(s) for s, _ in segments]
    ends = [int(e) for _, e in segments]
    return gaussian_soft_labels(T, starts, sigma), gaussian_soft_labels(T, ends, sigma)

