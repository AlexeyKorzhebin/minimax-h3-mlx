"""Keyframe preprocessing without torch.

`upstream`'s text encoder resolves its image processor through
`transformers.AutoProcessor.from_pretrained`, which constructs the *composite* Qwen3VL processor —
image, video and tokenizer together. The video half imports torchvision, which imports torch, and
this fork installs neither. The failure lands after the 28.22 GB text encoder has loaded.

Only two things are ever read off that processor (`upstream/minimax_h3_mlx/text_encoder.py:193,196`):
the image processor's `__call__` and its `merge_size`. So the composite is unnecessary, and
`mlx_vlm` already ships a torch-free `Qwen3VLImageProcessor` that satisfies both.
"""
from __future__ import annotations

import json
from pathlib import Path

#: Keys describing the composite processor, not the image processor's own parameters.
_COMPOSITE_KEYS = ("processor_class", "image_processor_type", "auto_map")


def load_image_processor(processor_dir: str | Path):
    """Build the image processor from the checkpoint's own preprocessor config."""
    from mlx_vlm.models.qwen3_vl.processing_qwen3_vl import Qwen3VLImageProcessor

    config_path = Path(processor_dir) / "preprocessor_config.json"
    try:
        config = json.loads(config_path.read_text())
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"could not read {config_path}: {exc}") from exc

    kwargs = {k: v for k, v in config.items() if k not in _COMPOSITE_KEYS}
    return Qwen3VLImageProcessor(**kwargs)


class TorchFreeProcessor:
    """Stands in for the composite processor, exposing only what upstream reads.

    Deliberately not a subclass of anything: the surface is two call sites, and a wider stand-in
    would invite code to depend on parts that are not actually torch-free.
    """

    def __init__(self, processor_dir: str | Path):
        self.image_processor = load_image_processor(processor_dir)
