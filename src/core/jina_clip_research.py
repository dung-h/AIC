"""Research-only loader for jina-clip-v2.

Jina's current remote modeling code imports ``clip_loss``, removed by
Transformers 5.x. Inject the unchanged historical helper before dynamic-module
loading so this experiment does not downgrade the shared Qwen runtime.
"""
from __future__ import annotations


def load_jina_clip(model_path: str):
    import torch
    import json
    from pathlib import Path
    from transformers import AutoModel
    from transformers.models.clip import modeling_clip

    if not hasattr(modeling_clip, "clip_loss"):
        def clip_loss(similarity: torch.Tensor) -> torch.Tensor:
            caption_loss = modeling_clip.contrastive_loss(similarity)
            image_loss = modeling_clip.contrastive_loss(similarity.t())
            return (caption_loss + image_loss) / 2.0

        modeling_clip.clip_loss = clip_loss

    # Disable optional fused attention paths that force bf16 kernels on this GPU.
    config_path = Path(model_path) / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config.setdefault("text_config", {})["use_flash_attn"] = False
    config["use_text_flash_attn"] = False
    config.setdefault("vision_config", {})["use_xformers"] = False
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model = AutoModel.from_pretrained(
        model_path, trust_remote_code=True, dtype=dtype,
        config=config
    )
    return model.to("cuda" if torch.cuda.is_available() else "cpu", dtype=dtype).eval()
