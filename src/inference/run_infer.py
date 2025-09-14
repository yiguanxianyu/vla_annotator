from __future__ import annotations

import argparse
import json
import os
from typing import List

import numpy as np
import torch
import yaml

from ..datasets.transforms import load_frames
from ..models.constrained_decoder import ConstrainedJSONGenerator
from ..models.heads import KHead, StartEndHead
from ..models.qwen_vl_loader import QwenVLWithHeads
from ..inference.postprocess import fill_frames_in_json, pair_segments, sigmoid
from ..utils.json_schema import validate_json_str
from ..utils.logging import setup_logger


logger = setup_logger(__name__)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/infer.yaml")
    ap.add_argument("--frames", type=str, nargs="*", default=None)
    ap.add_argument("--episode-id", type=int, default=None)
    ap.add_argument("--adapter-path", type=str, default=None)
    ap.add_argument("--heads-path", type=str, default=None)
    ap.add_argument("--save", type=str, default=None)
    return ap.parse_args()


def main():
    args = parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    base = cfg["model"]["base"]
    add_tokens = cfg["model"]["add_tokens"]
    k_max = int(cfg["model"].get("k_max", 12))
    max_frames = int(cfg["model"].get("max_frames", 128))
    min_len = int(cfg["heatmap"].get("min_len", 4))

    qlora = cfg.get("qlora", {"load_in_4bit": False})
    use_4bit = bool(qlora.get("load_in_4bit", False))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = QwenVLWithHeads(base, add_tokens, use_4bit=use_4bit, lora_cfg=None).to(device)

    # Load LoRA adapter if provided
    adapter_path = args.adapter_path or cfg.get("lora", {}).get("adapter_path", None)
    if adapter_path and os.path.isdir(adapter_path):
        try:
            # For PEFT models, load adapter if method exists
            if hasattr(model.model, "load_adapter"):
                model.model.load_adapter(adapter_path)
                logger.info("Loaded adapter from %s", adapter_path)
            else:
                logger.warning("Model does not support load_adapter(); skipping adapter load")
        except Exception as e:
            logger.warning("Failed to load adapter: %s", e)

    # Load heads
    vis_dim = model.vision_backbone.out_dim
    k_head = KHead(in_dim=vis_dim, k_max=k_max).to(device)
    se_head = StartEndHead(in_dim=vis_dim).to(device)
    heads_path = args.heads_path or os.path.join(os.path.dirname(cfg["infer"].get("save_path", "")) or ".", "heads.pt")
    if heads_path and os.path.isfile(heads_path):
        try:
            sd = torch.load(heads_path, map_location=device)
            k_head.load_state_dict(sd["k_head"])  # type: ignore[index]
            se_head.load_state_dict(sd["se_head"])  # type: ignore[index]
            logger.info("Loaded heads from %s", heads_path)
        except Exception as e:
            logger.warning("Failed to load heads: %s", e)

    # Inputs
    frames = args.frames or cfg.get("infer", {}).get("input_frames", [])
    if not frames:
        raise SystemExit("No input frames provided. Use --frames or infer.input_frames.")
    frames = frames[:max_frames]
    images = load_frames(frames, max_frames=max_frames)

    episode_id = args.episode_id if args.episode_id is not None else int(cfg.get("infer", {}).get("episode_id", 0))
    template = str(cfg.get("infer", {}).get("system_prompt_template", ""))
    system_prompt = template.replace("${EPISODE_ID}", str(episode_id))

    # Features
    with torch.no_grad():
        ff, pp = model.get_temporal_features(images)
        k_logits = k_head(pp)
        K_pred = int(torch.argmax(k_logits, dim=-1).item())
        s_log, e_log = se_head(ff, valid_mask=torch.ones(1, ff.shape[1], device=ff.device))
        p_start = sigmoid(s_log[0].detach().cpu().numpy())
        p_end = sigmoid(e_log[0].detach().cpu().numpy())
        segs = pair_segments(p_start, p_end, K_pred, min_len=min_len)

    # Constrained JSON generation
    generator = ConstrainedJSONGenerator(tokenizer=model.processor.tokenizer)

    def gen_fn(prompt: str, max_new_tokens: int = 512):
        # Build chat with frames and system prompt
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": images},
                    {"type": "text", "text": system_prompt},
                ],
            }
        ]
        text = model.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = model.processor(text=[text], images=images, padding=True, return_tensors="pt")
        return model.generate_text(inputs, max_new_tokens=max_new_tokens)

    json_text = generator.generate(episode_id=episode_id, k=len(segs), hints={}, generate_fn=None)
    json_text = fill_frames_in_json(json_text, segs)
    ok = validate_json_str(json_text)
    if not ok:
        logger.warning("Generated JSON failed validation; attempting minimal repair")

    save_path = args.save or cfg.get("infer", {}).get("save_path", "outputs/pred.json")
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        f.write(json_text)
    logger.info("Saved output to %s", save_path)


if __name__ == "__main__":
    main()

