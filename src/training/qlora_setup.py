from __future__ import annotations

from typing import Any, Dict

from ..utils.logging import setup_logger

logger = setup_logger(__name__)


def parse_lora_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "r": int(cfg.get("r", 16)),
        "alpha": int(cfg.get("alpha", 16)),
        "dropout": float(cfg.get("dropout", 0.05)),
        "target_modules": cfg.get("target_modules", None),
    }


def parse_qlora_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "load_in_4bit": bool(cfg.get("load_in_4bit", True)),
        "quant_type": str(cfg.get("quant_type", "nf4")),
        "compute_dtype": str(cfg.get("compute_dtype", "bfloat16")),
        "double_quant": bool(cfg.get("double_quant", True)),
    }

