from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml

from ..datasets.collate import Sample
from ..datasets.transforms import load_frames
from ..datasets.gaussian_labels import build_start_end_soft_labels
from ..models.heads import KHead, StartEndHead
from ..models.qwen_vl_loader import QwenVLWithHeads
from ..training.losses import bce_with_logits_masked, length_prior_loss, build_lm_labels
from ..utils.logging import setup_logger
from ..utils.seed import set_seed


logger = setup_logger(__name__)


@dataclass
class EpisodeRow:
    episode_id: int
    frames_paths: List[str]
    task_name: str
    init_scene_text: str
    action_config: List[Dict[str, Any]]


def read_jsonl(path: str) -> List[EpisodeRow]:
    rows: List[EpisodeRow] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            rows.append(
                EpisodeRow(
                    episode_id=int(obj["episode_id"]),
                    frames_paths=list(obj["frames_paths"]),
                    task_name=str(obj.get("task_name", "")),
                    init_scene_text=str(obj.get("init_scene_text", "")),
                    action_config=list(obj.get("label_info", {}).get("action_config", obj.get("action_config", []))),
                )
            )
    return rows


def iter_batches(rows: List[EpisodeRow], batch_size: int) -> Iterable[List[EpisodeRow]]:
    for i in range(0, len(rows), batch_size):
        yield rows[i : i + batch_size]


def run_train(cfg_path: str) -> None:
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    set_seed(42)

    data_path = cfg["train"]["data"]
    rows = read_jsonl(data_path)
    logger.info("Loaded %d episodes", len(rows))

    base = cfg["model"]["base"]
    add_tokens = cfg["model"]["add_tokens"]
    k_max = int(cfg["model"].get("k_max", 12))

    qlora = cfg["qlora"]
    lora = cfg.get("lora", {})
    use_4bit = bool(qlora.get("load_in_4bit", True))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = QwenVLWithHeads(base, add_tokens, use_4bit=use_4bit, lora_cfg=lora).to(device)

    # Attach heads
    vis_dim = model.vision_backbone.out_dim
    k_head = KHead(in_dim=vis_dim, k_max=k_max).to(device)
    se_head = StartEndHead(in_dim=vis_dim).to(device)
    # For stability, split LR: heads a bit higher
    base_lr = float(cfg["optimizer"]["lr"])
    opt = optim.AdamW(
        [
            {"params": model.parameters(), "lr": base_lr},
            {"params": list(k_head.parameters()) + list(se_head.parameters()), "lr": base_lr * 1.5},
        ],
        betas=tuple(cfg["optimizer"]["betas"]),
        weight_decay=float(cfg["optimizer"]["weight_decay"]),
    )

    epochs = int(cfg["train"]["epochs"])
    batch_size = int(cfg["train"]["batch_size"])
    grad_accum = int(cfg["train"]["grad_accum"])
    max_frames = int(cfg["model"]["max_frames"])
    curriculum_pct = float(cfg["train"].get("curriculum_pct", 0.2))
    heatmap_cfg = cfg["heatmap"]
    sigma = float(heatmap_cfg["sigma"])
    pos_weight = float(heatmap_cfg.get("pos_weight", 0.0)) if float(heatmap_cfg.get("focal_gamma", 0.0)) <= 0.0 else None
    focal_gamma = float(heatmap_cfg.get("focal_gamma", 0.0))
    min_len = int(heatmap_cfg.get("min_len", 4))

    lw = cfg["loss_weights"]
    w_txt = float(lw.get("txt", 1.0))
    w_k = float(lw.get("k", 1.0))
    w_h = float(lw.get("heatmap", 1.0))
    w_len = float(lw.get("len_prior", 0.1))

    total_steps = epochs * math.ceil(len(rows) / batch_size)
    curriculum_steps = int(curriculum_pct * total_steps)

    model.train()
    k_head.train()
    se_head.train()

    step = 0
    for epoch in range(1, epochs + 1):
        for batch_rows in iter_batches(rows, batch_size):
            # Load frames and build inputs
            frames_list = []  # list of list of PIL images
            valid_masks = []
            K_gt = []
            segs_list = []
            lm_texts = []
            for r in batch_rows:
                frames = load_frames(r.frames_paths, max_frames=max_frames)
                frames_list.append(frames)
                valid_masks.append([1.0] * len(frames))
                k = min(len(r.action_config), k_max)
                K_gt.append(k)
                segs = [(int(x["start_frame"]), int(x["end_frame"])) for x in r.action_config]
                segs_list.append(segs)
                # System prompt only — target JSON is ground-truth rendered with <FRAME>
                # For curriculum: LM learns structure but ignores <FRAME> token
                obj = {
                    "episode_id": r.episode_id,
                    "task_name": r.task_name,
                    "init_scene_text": r.init_scene_text,
                    "label_info": {
                        "action_config": [
                            {
                                "start_frame": "<FRAME>",
                                "end_frame": "<FRAME>",
                                "action_text": s.get("action_text", ""),
                                "skill": s.get("skill", ""),
                            }
                            for s in r.action_config[:k]
                        ]
                    },
                }
                lm_texts.append("<JSON>" + json.dumps(obj, ensure_ascii=False))

            # Temporal features and K/heatmap heads
            max_T = max(len(fr) for fr in frames_list) if frames_list else 0
            # Pad mask to tensor
            valid_mask = torch.zeros(len(frames_list), max_T, device=device)
            for i, fr in enumerate(frames_list):
                valid_mask[i, : len(fr)] = 1.0

            # Fallback encoder processes per sample sequentially (small batch)
            frame_feats_list = []
            pooled_list = []
            for fr in frames_list:
                ff, pp = model.get_temporal_features(fr)
                frame_feats_list.append(ff)
                pooled_list.append(pp)
            # pad to [B,T,D]
            D = frame_feats_list[0].shape[-1] if frame_feats_list else model.vision_backbone.out_dim
            frame_feats = torch.zeros(len(frames_list), max_T, D, device=device)
            for i, ff in enumerate(frame_feats_list):
                T_i = ff.shape[1]
                frame_feats[i, :T_i] = ff[0]
            pooled = torch.cat(pooled_list, dim=0) if pooled_list else torch.zeros(0, D, device=device)

            k_logits = k_head(pooled)
            start_logits, end_logits = se_head(frame_feats, valid_mask=valid_mask)

            # Heatmap targets
            y_start = torch.zeros_like(start_logits)
            y_end = torch.zeros_like(end_logits)
            for i, segs in enumerate(segs_list):
                ys, ye = build_start_end_soft_labels(T=int(valid_mask[i].sum().item()), segments=segs, sigma=sigma)
                y_start[i, : ys.shape[0]] = torch.from_numpy(ys).to(device)
                y_end[i, : ye.shape[0]] = torch.from_numpy(ye).to(device)

            # Losses
            K_gt_t = torch.tensor(K_gt, device=device, dtype=torch.long)
            k_loss = nn.CrossEntropyLoss()(k_logits, K_gt_t)
            heat_loss = bce_with_logits_masked(start_logits, y_start, mask=valid_mask, pos_weight=pos_weight, focal_gamma=focal_gamma) \
                + bce_with_logits_masked(end_logits, y_end, mask=valid_mask, pos_weight=pos_weight, focal_gamma=focal_gamma)
            len_loss = length_prior_loss(start_logits, end_logits, float(min_len), mask=valid_mask)

            # LM loss (text-only training for structure with <FRAME> masked)
            tok = model.processor.tokenizer
            enc = tok(lm_texts, return_tensors="pt", padding=True)
            input_ids = enc["input_ids"].to(device)
            attn_mask = enc["attention_mask"].to(device)
            labels = build_lm_labels(input_ids, frame_token_id=model.special_ids.frame, ignore_index=-100)
            out = model.model(input_ids=input_ids, attention_mask=attn_mask, labels=labels)
            lm_loss = out.loss

            # Curriculum schedule
            if step < curriculum_steps:
                loss = w_txt * lm_loss + w_k * k_loss
            else:
                loss = w_txt * lm_loss + w_k * k_loss + w_h * heat_loss + w_len * len_loss

            loss = loss / grad_accum
            loss.backward()

            if (step + 1) % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(k_head.parameters()) + list(se_head.parameters()), 1.0)
                opt.step()
                opt.zero_grad(set_to_none=True)

            if step % int(cfg["train"].get("log_interval", 50)) == 0:
                logger.info(
                    "ep %d step %d | lm %.4f | k %.4f | heat %.4f | len %.4f | total %.4f",
                    epoch,
                    step,
                    float(lm_loss.detach().cpu()),
                    float(k_loss.detach().cpu()),
                    float(heat_loss.detach().cpu()),
                    float(len_loss.detach().cpu()),
                    float((loss * grad_accum).detach().cpu()),
                )

            step += 1

    out_dir = cfg["train"]["output_dir"]
    os.makedirs(out_dir, exist_ok=True)
    # Save LoRA adapter if any
    try:
        if hasattr(model.model, "save_pretrained"):
            adir = os.path.join(out_dir, "adapter")
            model.model.save_pretrained(adir)
            logger.info("Saved adapter to %s", adir)
    except Exception as e:
        logger.warning("Failed to save adapter: %s", e)
    try:
        torch.save({"k_head": k_head.state_dict(), "se_head": se_head.state_dict()}, os.path.join(out_dir, "heads.pt"))
        logger.info("Saved heads to %s", os.path.join(out_dir, "heads.pt"))
    except Exception as e:
        logger.warning("Failed to save heads: %s", e)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/train.yaml")
    args = ap.parse_args()
    run_train(args.config)
