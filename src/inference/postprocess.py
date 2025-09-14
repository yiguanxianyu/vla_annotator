from __future__ import annotations

import json
from typing import List, Sequence, Tuple

import numpy as np


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))


def local_peaks(prob: np.ndarray, window: int = 3) -> List[int]:
    T = len(prob)
    w = window // 2
    peaks = []
    for t in range(T):
        l = max(0, t - w)
        r = min(T, t + w + 1)
        if prob[t] == prob[l:r].max():
            peaks.append(t)
    return peaks


def topk_indices(values: np.ndarray, k: int) -> List[int]:
    if k <= 0:
        return []
    k = min(k, len(values))
    idx = np.argpartition(values, -k)[-k:]
    idx = idx[np.argsort(values[idx])][::-1]
    return idx.tolist()


def pair_segments(p_start: np.ndarray, p_end: np.ndarray, K: int, min_len: int = 4) -> List[Tuple[int, int]]:
    T = len(p_start)
    peaks = local_peaks(p_start, window=5)
    # Select top-K start peaks
    start_idx = topk_indices(p_start[peaks], K)
    starts = sorted([peaks[i] for i in start_idx])
    segs: List[Tuple[int, int]] = []
    prev_end = -1
    for s in starts:
        s = max(s, prev_end)
        if s >= T:
            s = T - 1
        right = p_end[s + 1 :]
        if len(right) == 0:
            e = min(T - 1, s + min_len)
        else:
            e = s + 1 + int(np.argmax(right))
            if e < s + min_len:
                e = min(T - 1, s + min_len)
        e = max(e, s)
        segs.append((int(s), int(e)))
        prev_end = e
    return segs


def fill_frames_in_json(json_text: str, segments: Sequence[Tuple[int, int]]) -> str:
    # Replace first two occurrences of <FRAME> per segment (start then end)
    out = json_text
    for s, e in segments:
        out = out.replace("<FRAME>", str(int(s)), 1)
        out = out.replace("<FRAME>", str(int(e)), 1)
    return out

