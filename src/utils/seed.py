import os
import random
from typing import Optional

import numpy as np


def set_seed(seed: int = 42, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True  # type: ignore[attr-defined]
            torch.backends.cudnn.benchmark = False  # type: ignore[attr-defined]
    except Exception:
        pass

