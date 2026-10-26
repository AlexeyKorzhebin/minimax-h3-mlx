#!/usr/bin/env python3
"""Generate reference outputs from the real transformers processor, once.

Run this in a THROWAWAY venv outside the repository — torch must never enter the project
environment. The fixtures it writes are committed, so the equivalence keeps being checked on every
test run afterwards without torch:

    python3 -m venv /tmp/h3-fixture-venv
    /tmp/h3-fixture-venv/bin/pip install -q torch transformers pillow numpy
    /tmp/h3-fixture-venv/bin/python scripts/generate_processor_fixtures.py
    rm -rf /tmp/h3-fixture-venv
"""
import json
from pathlib import Path

import numpy as np
from PIL import Image
from transformers import AutoImageProcessor

PROCESSOR_DIR = Path.home() / "models/h3-converted/processor"
OUT = Path(__file__).resolve().parent.parent / "tests/fixtures/processor"

#: Sizes chosen for what varies in preprocessing, not for variety's sake: a landscape and a
#: portrait (resize rounding differs), one below `min_pixels` and one above `max_pixels` (both
#: bounds), and one not a multiple of patch_size*merge_size (the padding path).
CASES = {
    "landscape": (640, 384),
    "portrait": (384, 640),
    "tiny": (64, 48),
    "huge": (2048, 1536),
}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    # `AutoProcessor.from_pretrained(PROCESSOR_DIR)` (the composite) cannot be built from this
    # checkpoint's `processor/` directory at all: it has no `config.json` or
    # `video_preprocessor_config.json`, so transformers cannot resolve a video sub-processor for
    # `Qwen3VLProcessor` — independent of torch/torchvision being installed. Tracing
    # `ProcessorMixin._get_arguments_from_pretrained` (transformers 5.14.1) shows the composite's
    # `image_processor` attribute is itself built via a bare
    # `AutoImageProcessor.from_pretrained(pretrained_model_name_or_path, subfolder="")` call with
    # no extra kwargs — exactly the call below. So this is not a loosened substitute; it is the
    # same code path the composite would have taken for the one attribute this fork reads.
    processor = AutoImageProcessor.from_pretrained(str(PROCESSOR_DIR))
    manifest = {}
    for name, (w, h) in CASES.items():
        # A deterministic gradient — content must be reproducible from the manifest alone.
        arr = np.zeros((h, w, 3), dtype=np.uint8)
        arr[..., 0] = np.linspace(0, 255, w, dtype=np.uint8)[None, :]
        arr[..., 1] = np.linspace(0, 255, h, dtype=np.uint8)[:, None]
        image = Image.fromarray(arr)

        out = processor(images=[image], return_tensors="np")
        np.savez_compressed(OUT / f"{name}.npz",
                            pixel_values=out["pixel_values"],
                            image_grid_thw=out["image_grid_thw"])
        manifest[name] = {"size": [w, h],
                          "pixel_values_shape": list(out["pixel_values"].shape),
                          "image_grid_thw": out["image_grid_thw"].tolist()}
        print(f"{name}: {out['pixel_values'].shape}")

    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
