"""Load mere.run's 8-bit Qwen3-VL conditioner.

The port's `MiniMaxH3TextEncoder` builds plain `nn.Linear` layers and then loads keys by name. On
an affine-quantized checkpoint that fails **silently and completely**:

* ``...q_proj.weight`` is packed uint32 of shape ``[8192, 1280]``; it matches the expected key
  ``...q_proj.weight`` and is written straight into a bf16 ``[8192, 5120]`` slot;
* ``...q_proj.scales`` / ``.biases`` are not in the expected set, so the loader counts them as
  "skipped" and drops them;
* the stock loader only raises on *missing* keys, and none are missing.

Worse, the stock loader casts every tensor with ``tensor.astype(dtype)``, which on packed uint32 is
an arithmetic conversion — the bit pattern is destroyed before it is even mis-stored.

This module fixes all of it by overriding `_load_weights`: quantize the module tree first, using
the recipe the converter recorded, then load with a **two-way** key check so a dropped ``scales``
is an error rather than a statistic.

The recipe cannot be a blanket `nn.quantize`: mere.run left `visual.blocks.N.mlp.linear_fc2` dense
(bf16 ``[1152, 4304]``, no scales), so quantizing it would invent a layer the checkpoint has no
weights for. `text_encoder/quant_config.json` lists the quantized module paths explicitly.
"""

from __future__ import annotations

import glob
import json
from pathlib import Path

from . import _upstream  # noqa: F401
from .image_processor import TorchFreeProcessor

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_unflatten

from minimax_h3_mlx.packing import TEXT_ENCODER_LAYER
from minimax_h3_mlx.text_encoder import MiniMaxH3TextEncoder

LANGUAGE_PREFIX = "model.language_model."
VISION_PREFIX = "model.visual."

#: Vision-tower weights stored (in the checkpoint) in a PyTorch conv layout that MLX does not use.
#: The only convolution in the vision tower is the patch-embedding stem; everything else is
#: attention and linear layers, which have no axis-order ambiguity to correct.
VISION_CONV3D_WEIGHTS = frozenset({"patch_embed.proj.weight"})


#: The expression `patches/0001-keyframe-masked-scatter.patch` removes. Its presence means the
#: vendored checkout is unpatched, and any run with a keyframe will die inside `encode` — after
#: the 28.2 GB encoder has loaded, which is a slow way to learn about a missing patch.
UNPATCHED_SCATTER = "mx.where(image_mask"


def keyframe_scatter_patch_applied(source: str | None = None) -> bool:
    """Whether the vendored `upstream/` carries this fork's keyframe scatter fix.

    Reads the live source rather than a file path, so it stays honest when `H3_UPSTREAM` points
    the checkout somewhere else. `source` overrides that, which is what lets a test hand it known
    patched and unpatched text — without it, replacing this whole body with `return True` passed
    the entire suite.

    Comment lines are stripped first, and that is load-bearing: the patch quotes the very
    expression it replaces so a reader can see what changed, which made the first version of this
    check report a patched file as unpatched.
    """
    if source is None:
        import inspect

        from minimax_h3_mlx.text_encoder import MiniMaxH3TextEncoder

        source = inspect.getsource(MiniMaxH3TextEncoder.encode)

    code = [line for line in source.splitlines() if not line.lstrip().startswith("#")]
    return not any(UNPATCHED_SCATTER in line for line in code)


def to_mlx_conv3d_layout(tensor: mx.array) -> mx.array:
    """PyTorch's ``(out, in, D, H, W)`` conv weight -> MLX's channels-last ``(out, D, H, W, in)``.

    ``mlx.nn.Conv3d`` builds its own weight as ``(out_channels, *kernel_size, in_channels)``
    (``mlx/nn/layers/convolution.py``), but the converted checkpoint stores
    ``model.visual.patch_embed.proj.weight`` unmodified from the source PyTorch checkpoint, as
    ``(out_channels, in_channels, kD, kH, kW)``. Loading it as-is passes every shape check that
    only counts elements or checks key presence — it is still a 5-D tensor of the right dtype and
    total size — and fails only inside ``mx.conv3d`` itself, the first time an image is actually
    encoded, with a channel-count mismatch (see ``docs/RESULTS.md``, "Keyframe conditioning").

    ``upstream/minimax_h3_mlx/load.py`` already applies this exact transpose for the video VAE's
    conv weights; the vision tower's loader (``upstream/minimax_h3_mlx/text_encoder.py``, not
    modified by this fork) never got the same treatment, which is why this lives here instead.
    """
    return tensor.transpose(0, 2, 3, 4, 1)


def split_recipe(quantized_modules: list[str]) -> dict[str, set[str]]:
    """Split the recorded module paths into the two sub-trees `nn.quantize` is called on.

    The recipe is written in checkpoint space (``model.language_model.layers.0.self_attn.q_proj``);
    `nn.quantize(self.language, ...)` sees paths relative to that sub-tree (``layers.0...``).
    """
    buckets: dict[str, set[str]] = {"language": set(), "vision": set()}
    for path in quantized_modules:
        if path.startswith(LANGUAGE_PREFIX):
            buckets["language"].add(path[len(LANGUAGE_PREFIX):])
        elif path.startswith(VISION_PREFIX):
            buckets["vision"].add(path[len(VISION_PREFIX):])
    return buckets


def prepare_loaded_tensor(bucket: str, path: str, tensor: mx.array, dtype: mx.Dtype) -> mx.array:
    """Everything `_load_weights` does to one tensor between reading it and storing it.

    Extracted from the load loop so it can be tested without materializing 28.2 GB. Both rules
    here fail silently if dropped, which is why they are worth a unit of their own:

    * the vision tower's one conv weight arrives in PyTorch's layout and must be transposed —
      and a *wrong* permutation would not raise either, since the kernel's H and W are both 16;
    * a quantized layer's packed storage must never be cast, because it is bit-packing rather
      than a number, and `astype` would reinterpret it arithmetically.
    """
    if bucket == "vision" and path in VISION_CONV3D_WEIGHTS:
        tensor = to_mlx_conv3d_layout(tensor)
    if tensor.dtype not in (mx.uint32, mx.int32, mx.uint8):
        tensor = tensor.astype(dtype)
    return tensor


def quantize_selected(module: nn.Module, paths: set[str], bits: int, group_size: int) -> int:
    """Quantize exactly the listed linears, and nothing else. Returns how many were converted."""
    converted = 0

    def predicate(path: str, layer: nn.Module):
        nonlocal converted
        if not isinstance(layer, nn.Linear) or path not in paths:
            return False
        converted += 1
        return {"group_size": group_size, "bits": bits}

    nn.quantize(module, group_size=group_size, bits=bits, class_predicate=predicate)
    return converted


def audit_keys(loaded, wanted, expected) -> tuple[dict[str, list[str]], int, list[str]]:
    """Sort checkpoint keys into kept / deliberately-skipped / dropped.

    ``dropped`` is the category the stock loader does not have: a key that maps into a sub-tree but
    matches no parameter there. For an affine-quantized checkpoint that is where every ``scales``
    and ``biases`` ends up when the tree was not quantized — silently, because the stock loader
    only ever checks the *missing* direction.
    """
    kept: dict[str, list[str]] = {bucket: [] for bucket in expected}
    dropped: list[str] = []
    skipped = 0
    for key in loaded:
        target = wanted(key)
        if target is None:
            skipped += 1
            continue
        bucket, path = target
        if path in expected[bucket]:
            kept[bucket].append(key)
        else:
            dropped.append(key)
    return kept, skipped, dropped


def describe_dropped(dropped: list[str]) -> str | None:
    """The error text for keys that had nowhere to go, or ``None`` when there are none."""
    if not dropped:
        return None
    quant = [k for k in dropped if k.endswith((".scales", ".biases"))]
    if quant:
        return (
            f"{len(quant)} quantization tensors had nowhere to go, e.g. {quant[:4]}. The module "
            "tree was quantized with a recipe that does not match the checkpoint — loading would "
            "silently discard them and leave the corresponding weights as raw packed integers."
        )
    return (
        f"{len(dropped)} checkpoint tensors do not correspond to any parameter, e.g. "
        f"{dropped[:4]}."
    )


class QuantizedTextEncoder(MiniMaxH3TextEncoder):
    """`MiniMaxH3TextEncoder` that can read an affine-quantized checkpoint.

    Everything else — the truncated 50-layer stack, the request presentation, the pre-norm hidden
    state H3 conditions on — is inherited untouched.
    """

    def __init__(
        self,
        model_dir: str | Path,
        num_layers: int = TEXT_ENCODER_LAYER,
        dtype: mx.Dtype = mx.bfloat16,
        load_vision: bool = True,
        verbose: bool = False,
    ):
        model_dir = Path(model_dir)
        recipe_path = model_dir / "quant_config.json"
        if recipe_path.exists():
            with open(recipe_path) as fh:
                self._recipe = json.load(fh)
        else:
            self._recipe = None
        self.quantized_layers = {"language": 0, "vision": 0}
        super().__init__(model_dir, num_layers=num_layers, dtype=dtype,
                         load_vision=load_vision, verbose=verbose)

    def _load_weights(self, model_dir: Path, dtype: mx.Dtype, verbose: bool) -> None:
        if self._recipe is not None:
            buckets = split_recipe(self._recipe.get("quantized_modules", []))
            bits = int(self._recipe["bits"])
            group_size = int(self._recipe["group_size"])
            for name, module in (("language", self.language), ("vision", self.vision)):
                if module is None:
                    continue
                self.quantized_layers[name] = quantize_selected(
                    module, buckets[name], bits, group_size
                )
            if verbose:
                print(f"  quantized {self.quantized_layers} linears at {bits}-bit "
                      f"(group {group_size})")

        shards = sorted(glob.glob(str(model_dir / "*.safetensors")))
        if not shards:
            raise FileNotFoundError(f"No safetensors in {model_dir}.")

        buckets_out: dict[str, dict[str, mx.array]] = {"language": {}, "vision": {}}
        expected = {
            "language": {k for k, _ in tree_flatten(self.language.parameters())},
            "vision": set() if self.vision is None else
                      {k for k, _ in tree_flatten(self.vision.parameters())},
        }
        dropped: list[str] = []
        skipped = 0

        for shard in shards:
            loaded = mx.load(shard)
            kept, shard_skipped, shard_dropped = audit_keys(loaded, self._wanted, expected)
            skipped += shard_skipped
            dropped += shard_dropped
            for bucket, keys in kept.items():
                for key in keys:
                    path = self._wanted(key)[1]
                    buckets_out[bucket][path] = prepare_loaded_tensor(
                        bucket, path, loaded[key], dtype)
            if verbose:
                total = len(buckets_out["language"]) + len(buckets_out["vision"])
                print(f"  {Path(shard).name}: kept {total}")

        problem = describe_dropped(dropped)
        if problem:
            raise KeyError(problem)

        for bucket, module in (("language", self.language), ("vision", self.vision)):
            if module is None:
                continue
            missing = sorted(expected[bucket] - buckets_out[bucket].keys())
            if missing:
                raise KeyError(
                    f"{bucket} encoder missing {len(missing)} tensors, e.g. {missing[:4]}."
                )
            module.update(tree_unflatten(list(buckets_out[bucket].items())))

        mx.eval(self.language.parameters())
        if self.vision is not None:
            mx.eval(self.vision.parameters())
        self.skipped_tensors = skipped

    @property
    def processor(self):
        """Serve a torch-free processor instead of upstream's `AutoProcessor`.

        Upstream builds the composite Qwen3VL processor, whose video half needs torchvision. It is
        never used — `encode()` reads only `image_processor` — so constructing it costs a hard
        dependency for nothing, and the failure surfaces only once an image is passed, after the
        encoder has loaded.
        """
        if self._processor is None:
            self._processor = TorchFreeProcessor(self._model_dir.parent / "processor")
        return self._processor

    def unload(self) -> int:
        """Drop every parameter. Returns the bytes released.

        Only safe once whatever the encoder produced has been materialized — a lazy graph still
        referencing these arrays keeps them alive no matter what is deleted here.
        """
        released = 0
        for module in (self.language, self.vision):
            if module is None:
                continue
            released += sum(v.nbytes for _, v in tree_flatten(module.parameters()))
        self.language = None
        self.vision = None
        self._tokenizer = None
        self._processor = None
        return released
