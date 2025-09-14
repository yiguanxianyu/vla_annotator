from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def bce_with_logits_masked(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    pos_weight: Optional[float] = None,
    focal_gamma: float = 0.0,
) -> torch.Tensor:
    # logits, targets, mask: [B, T]
    if focal_gamma and focal_gamma > 0:
        p = torch.sigmoid(logits)
        ce = F.binary_cross_entropy(p, targets, reduction="none")
        pt = torch.where(targets > 0, p, 1 - p)
        loss = ((1 - pt) ** focal_gamma) * ce
    else:
        if pos_weight is not None:
            pw = torch.as_tensor(pos_weight, dtype=logits.dtype, device=logits.device)
        else:
            pw = None
        loss = F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pw, reduction="none")
    if mask is not None:
        loss = loss * mask
        denom = mask.sum().clamp(min=1.0)
    else:
        denom = torch.tensor(loss.numel(), dtype=loss.dtype, device=loss.device)
    return loss.sum() / denom


def soft_argmax_1d(prob: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    # prob: [B,T] in [0,1]
    if mask is not None:
        prob = prob * mask
    T = prob.shape[1]
    idx = torch.arange(T, device=prob.device, dtype=prob.dtype).unsqueeze(0)
    denom = prob.sum(dim=1, keepdim=True).clamp(min=1e-6)
    return (prob * idx).sum(dim=1) / denom.squeeze(1)


def length_prior_loss(start_logits: torch.Tensor, end_logits: torch.Tensor, min_len: float, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    ps = torch.sigmoid(start_logits)
    pe = torch.sigmoid(end_logits)
    s_pos = soft_argmax_1d(ps, mask)
    e_pos = soft_argmax_1d(pe, mask)
    diff = (min_len - (e_pos - s_pos)).relu()
    return diff.mean()


def build_lm_labels(input_ids: torch.Tensor, frame_token_id: int, ignore_index: int = -100) -> torch.Tensor:
    labels = input_ids.clone()
    if frame_token_id >= 0:
        labels[labels == frame_token_id] = ignore_index
    return labels

