from __future__ import annotations

import contextlib
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import bitsandbytes as bnb
import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoProcessor

from ..utils.logging import setup_logger

logger = setup_logger(__name__)


@dataclass
class SpecialTokenIDs:
    json: int
    thought: int
    seg_start: int
    seg_end: int
    start_frame: int
    end_frame: int
    frame: int


class VisionBackbone(nn.Module):
    """
    Fallback temporal vision encoder to produce [B,T,D] features when direct fused
    features from Qwen2.5-VL are not readily accessible. Uses MobileNetV3-Small and
    a projection to D.
    """

    def __init__(self, out_dim: int = 768):
        super().__init__()
        try:
            import torchvision.models as tvm

            m = tvm.mobilenet_v3_small(weights=tvm.MobileNet_V3_Small_Weights.DEFAULT)
            self.backbone = m.features
            in_ch = 576  # MobileNetV3 small final channels
        except Exception:
            self.backbone = nn.Sequential(
                nn.Conv2d(3, 32, 3, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 64, 3, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d((1, 1)),
            )
            in_ch = 64
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(in_ch, out_dim))
        self.out_dim = out_dim

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        # images: [B*T,3,H,W] in [0,1]
        x = self.backbone(images)
        x = self.pool(x)
        x = self.head(x)
        return x  # [B*T, D]


class QwenVLWithHeads(nn.Module):
    def __init__(
        self,
        base: str,
        add_tokens: List[str],
        use_4bit: bool,
        lora_cfg: Optional[dict] = None,
        compute_dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        self.processor = AutoProcessor.from_pretrained(base, trust_remote_code=True)

        quant_kwargs = {}
        device_map = "auto"
        if use_4bit:
            quant_kwargs = {
                "load_in_4bit": True,
                "bnb_4bit_quant_type": "nf4",
                "bnb_4bit_use_double_quant": True,
                "bnb_4bit_compute_dtype": compute_dtype,
                "device_map": device_map,
                "trust_remote_code": True,
            }
            logger.info("Loading base model in 4-bit (NF4)")
        else:
            quant_kwargs = {
                "torch_dtype": compute_dtype,
                "device_map": device_map,
                "trust_remote_code": True,
            }
            if use_4bit:
                logger.warning("bitsandbytes not available; falling back to non-4bit load")

        self.model = AutoModelForCausalLM.from_pretrained(base, **quant_kwargs)

        # Add tokens
        new_tokens = {
            "additional_special_tokens": add_tokens,
        }
        added = self.processor.tokenizer.add_special_tokens(new_tokens)
        if added > 0:
            self.model.resize_token_embeddings(len(self.processor.tokenizer))
            logger.info("Added %d special tokens and resized embeddings", added)

        # Map tokens to ids
        tok = self.processor.tokenizer
        self.special_ids = SpecialTokenIDs(
            json=tok.convert_tokens_to_ids("<JSON>"),
            thought=tok.convert_tokens_to_ids("<THOUGHT>"),
            seg_start=tok.convert_tokens_to_ids("<SEG_START>"),
            seg_end=tok.convert_tokens_to_ids("<SEG_END>"),
            start_frame=tok.convert_tokens_to_ids("<START_FRAME>"),
            end_frame=tok.convert_tokens_to_ids("<END_FRAME>"),
            frame=tok.convert_tokens_to_ids("<FRAME>"),
        )

        # Optional LoRA
        if lora_cfg:
            if LoraConfig is None:
                logger.warning("PEFT not available, skipping LoRA attach")
            else:
                target_modules = lora_cfg.get("target_modules")
                lora = LoraConfig(
                    r=lora_cfg.get("r", 16),
                    lora_alpha=lora_cfg.get("alpha", 16),
                    lora_dropout=lora_cfg.get("dropout", 0.05),
                    target_modules=target_modules,
                    bias="none",
                    task_type="CAUSAL_LM",
                )
                if use_4bit:
                    self.model = prepare_model_for_kbit_training(self.model)
                self.model = get_peft_model(self.model, lora)
                logger.info("Attached LoRA to modules: %s", target_modules)

        # Fallback vision backbone for [B,T,D] features if fused features aren't exposed
        self.vision_backbone = VisionBackbone(out_dim=768)

    def encode_frames(self, frames: List["PIL.Image.Image"], fps: Optional[float] = None) -> Dict[str, torch.Tensor]:
        # Build messages for Qwen2.5-VL processor; we only prepare inputs here.
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": [f"file://{i}" for i in range(len(frames))]},
                    {"type": "text", "text": ""},
                ],
            }
        ]
        # We won't actually pass file:// images to processor; instead we pass PIL frames directly
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=frames, videos=None, fps=fps, padding=True, return_tensors="pt")
        return inputs

    @torch.no_grad()
    def get_temporal_features(self, frames: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns (frame_feats [1,T,D], pooled [1,D]) using fallback encoder.
        frames: list of PIL images or tensors [3,H,W] in [0,1].
        """
        device = next(self.parameters()).device
        imgs = []
        for im in frames:
            if isinstance(im, torch.Tensor):
                x = im
            else:
                import torchvision.transforms as T

                x = T.Compose([T.Resize(224), T.CenterCrop(224), T.ToTensor()])(im)
            imgs.append(x)
        if not imgs:
            return torch.zeros(1, 0, self.vision_backbone.out_dim, device=device), torch.zeros(
                1, self.vision_backbone.out_dim, device=device
            )
        x = torch.stack(imgs, 0).to(device)
        x = (x.clamp(0, 1) - 0.5) * 2.0
        B = 1
        Tt = x.shape[0]
        feats = self.vision_backbone(x).view(B, Tt, -1)
        pooled = feats.mean(dim=1)
        return feats, pooled

    def generate_text(self, inputs: Dict[str, torch.Tensor], **gen_kwargs) -> List[str]:
        device = next(self.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        out_ids = self.model.generate(**inputs, **gen_kwargs)
        trimmed = [o[len(inp) :] for inp, o in zip(inputs["input_ids"], out_ids)]
        text = self.processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        return text
