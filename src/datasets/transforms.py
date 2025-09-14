from __future__ import annotations

import os
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image


def load_image(path: str) -> Image.Image:
    with Image.open(path) as im:
        return im.convert("RGB")


def load_frames(paths: List[str], max_frames: Optional[int] = None) -> List[Image.Image]:
    frames = [load_image(p) for p in paths]
    if max_frames is not None and len(frames) > max_frames:
        # uniform subsample
        idx = np.linspace(0, len(frames) - 1, max_frames).astype(int).tolist()
        frames = [frames[i] for i in idx]
    return frames

