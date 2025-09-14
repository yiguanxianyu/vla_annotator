Qwen Robot 2 — K/K-Head + Heatmap + Constrained JSON (QLoRA)

Overview

- Implements the three-head design on top of Qwen2.5-VL-3B:
  - Text head (LM): Generates structured JSON; frame numbers replaced by `<FRAME>`.
  - K head: Predicts segment count `K ∈ [0, K_max]`.
  - Heatmap head: Start/End two-channel heatmap over T frames with Gaussian soft labels.
- Adds minimal special tokens to tokenizer and resizes embeddings.
- QLoRA-compatible loader with safe fallbacks if bitsandbytes is unavailable.
- Single-pass inference pipeline: predict K + heatmaps, postprocess to segments, constrained decode JSON, fill frame numbers.

Repo Layout

project/
- configs/
  - train.yaml — Training configuration
  - infer.yaml — Inference configuration
- data/
  - episodes.jsonl — Example placeholder (not tracked); each line is one sample
- src/
  - datasets/
    - collate.py — Variable length batching and masks
    - transforms.py — Frame loading and preprocessing
    - gaussian_labels.py — Gaussian soft label utilities
  - models/
    - qwen_vl_loader.py — Load Qwen2.5-VL-3B with tokenizer extension; QLoRA attach
    - heads.py — K head and Start/End heatmap heads
    - constrained_decoder.py — Minimal FSA/templater to enforce structure
  - training/
    - losses.py — LM CE, BCE/Focal, priors
    - train_loop.py — Curriculum schedule and multi-head loss combine
    - qlora_setup.py — 4-bit load + PEFT config helpers
  - inference/
    - postprocess.py — Peak/NMS/greedy pairing, monotonic/min-len fixes, filling
    - run_infer.py — Orchestrates single-pass inference and output JSON
  - utils/
    - logging.py — Logger setup
    - metrics.py — Basic metrics (tIoU, MAE)
    - seed.py — Reproducibility
    - json_schema.py — Output schema + validator
- scripts/
  - prepare_data.py — Convert raw annotations into `episodes.jsonl` and label texts
  - train.sh — Training entry (Python)
  - infer.sh — Inference entry (Python)

Quick Start

1) Create and activate a Python 3.10+ environment. Install dependencies (see notes below):

   - Core: torch/torchvision, transformers>=4.43, accelerate, peft, einops, safetensors
   - Optional: bitsandbytes (4-bit QLoRA; not well supported on Windows GPU)
   - Utils: numpy, pillow, pyyaml, jsonschema, scipy, scikit-image (optional), opencv-python (optional)

2) Prepare data (make your own converter to `data/episodes.jsonl`):

   `python scripts/prepare_data.py --input /path/to/raw --output data/episodes.jsonl`

   Each line should include: `episode_id`, `frames_paths` (list of image paths), `task_name`, `init_scene_text`, and `action_config` with `start_frame/end_frame/action_text/skill`.

3) Train (curriculum schedule enabled):

   `bash scripts/train.sh`

4) Inference:

   `bash scripts/infer.sh`

Notes & Limitations

- QLoRA: On Windows, bitsandbytes may be unavailable. The loader will safely fall back to non-4bit load and warn.
- Features: Extracting [B,T,D] fused features from Qwen2.5-VL internals varies by version. The wrapper exposes a placeholder that can be adapted to your installed model version to pull time-aligned features. A CPU-friendly fallback uses average pooled embeddings if precise fusion hooks are missing.
- Constrained decode: A pragmatic FSA/templating approach is provided to enforce JSON structure and `<FRAME>` placeholders. You can swap it with a tighter Trie/JsonSchema-driven logits processor for stricter token-level constraints.

