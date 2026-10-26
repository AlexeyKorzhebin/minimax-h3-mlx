import json
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from h3_48gb.image_processor import TorchFreeProcessor, load_image_processor

PROCESSOR_DIR = Path.home() / "models/h3-converted/processor"
FIXTURES = Path(__file__).parent / "fixtures/processor"
pytestmark = pytest.mark.skipif(not PROCESSOR_DIR.exists(),
                                reason="converted checkpoint not present")


def test_processor_loads_without_torch():
    """The whole point: the composite AutoProcessor pulls torchvision, this must not.

    Checks both construction and processing paths, since torch is most likely to sneak in
    during image processing, not during initialization.
    """
    ip = load_image_processor(PROCESSOR_DIR)
    assert ip.merge_size == 2
    assert ip.patch_size == 16

    # Process an image to test the full path, not just construction
    ip(images=[Image.new("RGB", (640, 384), (200, 40, 40))], return_tensors="np")

    # Most critically: torch never entered during construction or processing
    assert "torch" not in sys.modules
    assert "torchvision" not in sys.modules


def test_the_guard_fires_when_torch_appears(monkeypatch):
    """Proves the guard in test_processor_loads_without_torch actually catches a regression.

    Rather than copying the assertion, this calls the real guard with a sentinel installed.
    If torch appears in sys.modules, the real guard will fire.
    """
    monkeypatch.setitem(sys.modules, "torch", object())
    with pytest.raises(AssertionError):
        test_processor_loads_without_torch()


def test_it_produces_what_upstream_consumes():
    """upstream reads exactly `pixel_values` and `image_grid_thw`, nothing else."""
    ip = load_image_processor(PROCESSOR_DIR)
    out = ip(images=[Image.new("RGB", (640, 384), (200, 40, 40))], return_tensors="np")
    assert out["pixel_values"].ndim == 2
    assert out["pixel_values"].dtype == np.float32
    assert out["image_grid_thw"].shape == (1, 3)


def test_wrapper_exposes_the_single_attribute_upstream_uses():
    proc = TorchFreeProcessor(PROCESSOR_DIR)
    assert proc.image_processor.merge_size == 2


def test_the_pixel_budget_survives_loading():
    """The keyframe's pixel budget decides how much of the sequence it occupies.

    `num_image_tokens = grid.prod() // merge**2` feeds `num_text`, and `packing` derives every
    media row's rotary clock from `num_text` — so this budget moves the whole timeline, silently.
    Checked against MiniMaxAI/MiniMax-H3, whose `FL2VA/processor`, `FL2VA/text_encoder` and
    `processor` all carry the same two numbers.
    """
    ip = load_image_processor(PROCESSOR_DIR)
    assert (ip.min_pixels, ip.max_pixels) == (65536, 16777216)


def test_the_official_key_spelling_would_be_ignored(tmp_path):
    """Why `convert_sawfwair.py` must not simply copy the official config verbatim.

    The official file writes the budget as `size: {shortest_edge, longest_edge}`. mlx-vlm's
    processor takes no such key: it silently keeps Qwen2-VL's defaults, capping a keyframe at
    1,003,520 pixels instead of 16,777,216. A 4K keyframe would then yield 943 image tokens
    where the released model produces 8160 — no exception, just a different clip. Only the
    `min_pixels`/`max_pixels` spelling this fork writes actually lands.
    """
    official = {"size": {"shortest_edge": 65536, "longest_edge": 16777216},
                "patch_size": 16, "temporal_patch_size": 2, "merge_size": 2,
                "image_mean": [0.5] * 3, "image_std": [0.5] * 3}
    (tmp_path / "preprocessor_config.json").write_text(json.dumps(official))
    ignored = load_image_processor(tmp_path)
    assert (ignored.min_pixels, ignored.max_pixels) != (65536, 16777216)


def test_a_broken_config_fails_loudly(tmp_path):
    """A silent fallback to AutoProcessor would reintroduce torch 28 GB into a run."""
    (tmp_path / "preprocessor_config.json").write_text("{ not json")
    with pytest.raises(Exception) as excinfo:
        load_image_processor(tmp_path)
    assert "preprocessor_config.json" in str(excinfo.value)


def test_encoder_serves_the_torch_free_processor(monkeypatch):
    """upstream's property would build AutoProcessor; ours must shadow it entirely."""
    import h3_48gb.text_encoder as te

    built = {}

    class _Spy(TorchFreeProcessor):
        def __init__(self, processor_dir):
            built["dir"] = Path(processor_dir)
            super().__init__(processor_dir)

    monkeypatch.setattr(te, "TorchFreeProcessor", _Spy)

    encoder = te.QuantizedTextEncoder.__new__(te.QuantizedTextEncoder)
    encoder._model_dir = Path.home() / "models/h3-converted/text_encoder"
    encoder._processor = None

    proc = encoder.processor
    assert proc.image_processor.merge_size == 2
    assert built["dir"].name == "processor", "must read the sibling processor directory"
    assert "torch" not in sys.modules


def _case_image(w: int, h: int) -> Image.Image:
    arr = np.zeros((h, w, 3), dtype=np.uint8)
    arr[..., 0] = np.linspace(0, 255, w, dtype=np.uint8)[None, :]
    arr[..., 1] = np.linspace(0, 255, h, dtype=np.uint8)[:, None]
    return Image.fromarray(arr)


@pytest.mark.parametrize("name", ["landscape", "portrait", "tiny", "huge"])
def test_matches_transformers_reference(name):
    """Two independent implementations can disagree in resize rounding or normalisation without
    raising — the clip is then conditioned on a subtly different image, and the only symptom is
    worse output hours later. These fixtures came from transformers itself."""
    manifest = json.loads((FIXTURES / "manifest.json").read_text())
    w, h = manifest[name]["size"]
    reference = np.load(FIXTURES / f"{name}.npz")

    out = load_image_processor(PROCESSOR_DIR)(images=[_case_image(w, h)], return_tensors="np")

    np.testing.assert_array_equal(out["image_grid_thw"], reference["image_grid_thw"])
    np.testing.assert_allclose(out["pixel_values"], reference["pixel_values"], atol=1e-5)
