from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=str, required=True, help="Path to raw dataset folder or JSON")
    ap.add_argument("--output", type=str, required=True, help="Output JSONL path")
    return ap.parse_args()


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    # This is a placeholder script. Convert your raw annotations to the JSONL format:
    # {episode_id, frames_paths[], task_name, init_scene_text, action_config[]}
    # where action_config: [{start_frame, end_frame, action_text, skill}, ...]
    # Below we just write an empty file if none exists.
    if not os.path.exists(args.output):
        with open(args.output, "w", encoding="utf-8") as f:
            pass
    print(f"Wrote placeholder to {args.output}. Please implement conversion logic for your dataset.")


if __name__ == "__main__":
    main()

