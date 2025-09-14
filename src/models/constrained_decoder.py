from __future__ import annotations

import json
from typing import Any, Dict, List, Optional


class ConstrainedJSONGenerator:
    """
    Pragmatic constrained generator: builds a JSON template with `<FRAME>` placeholders
    and lets an optional `generate_fn` fill free-text fields. When `generate_fn` is None,
    it uses provided hints or placeholders.
    """

    def __init__(self, tokenizer=None):
        self.tokenizer = tokenizer

    def generate(
        self,
        episode_id: int,
        k: int,
        hints: Optional[Dict[str, Any]] = None,
        generate_fn: Optional[callable] = None,
        max_new_tokens: int = 512,
    ) -> str:
        hints = hints or {}
        obj = {
            "episode_id": episode_id,
            "task_name": hints.get("task_name", ""),
            "init_scene_text": hints.get("init_scene_text", ""),
            "label_info": {
                "action_config": [
                    {
                        "start_frame": "<FRAME>",
                        "end_frame": "<FRAME>",
                        "action_text": hints.get("action_text", ""),
                        "skill": hints.get("skill", ""),
                    }
                    for _ in range(k)
                ]
            },
        }

        if generate_fn is not None:
            # Provide a brief prompt seed; generation is left to caller
            # The caller can decode under additional constraints if supported
            prompt = "<JSON>" + json.dumps(obj, ensure_ascii=False)
            text = generate_fn(prompt, max_new_tokens=max_new_tokens)
            return text[0] if isinstance(text, list) else str(text)
        else:
            return "<JSON>" + json.dumps(obj, ensure_ascii=False)

