from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np


SPECIAL_FRAME_TOKEN = "<FRAME>"


@dataclass
class Sample:
    episode_id: int
    frames_paths: List[str]
    task_name: str
    init_scene_text: str
    action_config: List[Dict[str, Any]]  # {start_frame, end_frame, action_text, skill}


def make_lm_target_text(sample: Sample, k: int) -> str:
    # JSON with <FRAME> placeholders
    segs = []
    for i in range(k):
        item = sample.action_config[i] if i < len(sample.action_config) else {
            "action_text": sample.action_config[0]["action_text"] if sample.action_config else "",
            "skill": sample.action_config[0]["skill"] if sample.action_config else "",
        }
        segs.append(
            {
                "start_frame": SPECIAL_FRAME_TOKEN,
                "end_frame": SPECIAL_FRAME_TOKEN,
                "action_text": item.get("action_text", ""),
                "skill": item.get("skill", ""),
            }
        )
    obj = {
        "episode_id": sample.episode_id,
        "task_name": sample.task_name,
        "init_scene_text": sample.init_scene_text,
        "label_info": {"action_config": segs},
    }
    return "<JSON>" + json.dumps(obj, ensure_ascii=False)


def collate_fn(samples: List[Sample], max_frames: int, k_max: int) -> Dict[str, Any]:
    batch = {}
    B = len(samples)
    # Frames are loaded later by processor; here we "pass through" frame paths
    frames_paths = [s.frames_paths[:max_frames] for s in samples]
    T = max(len(p) for p in frames_paths) if frames_paths else 0
    valid_mask = np.zeros((B, T), dtype=np.float32)
    K_gt = []
    start_indices = []
    end_indices = []
    lm_target_texts = []
    for b, s in enumerate(samples):
        t = len(frames_paths[b])
        valid_mask[b, :t] = 1.0
        k = min(len(s.action_config), k_max)
        K_gt.append(k)
        starts = [int(x["start_frame"]) for x in s.action_config]
        ends = [int(x["end_frame"]) for x in s.action_config]
        start_indices.append(starts)
        end_indices.append(ends)
        lm_target_texts.append(make_lm_target_text(s, k))

    batch.update(
        {
            "frames_paths": frames_paths,
            "valid_mask": valid_mask,
            "K_gt": np.array(K_gt, dtype=np.int64),
            "start_indices": start_indices,
            "end_indices": end_indices,
            "lm_target_texts": lm_target_texts,
        }
    )
    return batch

