from __future__ import annotations

import json
from typing import Any, Dict, List

try:
    from jsonschema import validate, Draft7Validator
except Exception:  # pragma: no cover
    validate = None
    Draft7Validator = None


OUTPUT_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "episode_id": {"type": "integer"},
        "task_name": {"type": "string"},
        "init_scene_text": {"type": "string"},
        "label_info": {
            "type": "object",
            "properties": {
                "action_config": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "start_frame": {"type": "integer", "minimum": 0},
                            "end_frame": {"type": "integer", "minimum": 0},
                            "action_text": {"type": "string"},
                            "skill": {"type": "string"},
                        },
                        "required": [
                            "start_frame",
                            "end_frame",
                            "action_text",
                            "skill",
                        ],
                    },
                }
            },
            "required": ["action_config"],
        },
    },
    "required": ["episode_id", "task_name", "init_scene_text", "label_info"],
}


def validate_json_str(s: str) -> bool:
    try:
        obj = json.loads(s)
    except Exception:
        return False
    if Draft7Validator is None:
        # Best effort if jsonschema not installed
        return all(k in obj for k in ("episode_id", "task_name", "init_scene_text", "label_info"))
    try:
        validate(instance=obj, schema=OUTPUT_SCHEMA)
        # check monotonicity/non-overlap
        cfg = obj.get("label_info", {}).get("action_config", [])
        prev_end = -1
        for item in cfg:
            if not (isinstance(item.get("start_frame"), int) and isinstance(item.get("end_frame"), int)):
                return False
            if item["start_frame"] > item["end_frame"]:
                return False
            if prev_end > item["start_frame"]:
                return False
            prev_end = item["end_frame"]
        return True
    except Exception:
        return False

