"""TAE: a 9.8 MB 2D decoder for in-flight previews.

The real video VAE cannot decode fewer than 7 latent frames — its temporal chunking is causal —
and tiles 28 times at 1344x768 regardless of how little is asked for. A preview therefore costs
49.3 s and 8.46 GB to show one frame. TAE is a plain 2D convolutional decoder: no temporal state,
no chunk floor, no tiling.

The weights are third party (`Kijai/MiniMax-H3-TAE`) and live outside the repository. This module
ships the loader, not the file.

Architecture is read from the checkpoint's own slot layout rather than guessed: 81 tensors in a
flat `Sequential` numbered 1..23, where the gaps (0, 2, 6, 11, 16, 20) are the parameterless
layers — an input clamp, a ReLU, and four nearest-neighbour x2 upsamples. Four upsamples make 16x,
which is exactly this VAE's `spatial_compression_ratio`, and the input is 24 channels, exactly its
`latent_channels`.
"""
from __future__ import annotations

from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

#: Third-party weights, deliberately outside the repo. Override for tests or another location.
TAE_WEIGHTS_PATH = Path.home() / "models/tae/taeh3.safetensors"

#: The video VAE's latent channel count, which the decoder's input convolution must match.
LATENT_CHANNELS = 24

#: Four x2 upsamples. Must equal the video VAE's `spatial_compression_ratio`.
SPATIAL_RATIO = 16

#: Does TAE want the latent as the sampler holds it (normalized), or after `latents * std + mean`?
#: Measured, not assumed — `scripts/measure_tae.py`, and the table in docs/RESULTS.md.
#:
#: The answer is normalized, and getting it right needed more than one metric. Against the real
#: VAE's own decode the denormalized form scores *higher* PSNR (17.50 vs 15.11 dB), which is what
#: a naive comparison would pick. It is wrong: that form wins on a colour shift that happens to
#: land near the reference's tone, while losing on structure (gradient correlation 0.457 vs 0.508)
#: and on correlation with the reference itself (0.883 vs 0.927).
#:
#: The decisive evidence is the second reference. The latent under test was produced by encoding a
#: known image, so that image is ground truth: normalized scores **21.78 dB / 0.939 correlation**
#: against it, where the real VAE manages 14.72 / 0.941 — the small decoder lands closer to the
#: original than the VAE does, because it skips the encoder round trip and the tiling.
TAE_EXPECTS_NORMALIZED = True


def to_mlx_conv2d_layout(tensor: mx.array) -> mx.array:
    """PyTorch's ``(out, in, kH, kW)`` -> MLX's channels-last ``(out, kH, kW, in)``.

    Every kernel in this checkpoint is square (3x3 or 1x1), so a permutation that swaps kH and kW
    yields the *same shape* as the correct one and raises nothing — it just scrambles the filter.
    `tests/test_tae.py` therefore pins values, not shape.
    """
    return tensor.transpose(0, 2, 3, 1)


class Block(nn.Module):
    """TAESD's residual block: three 3x3 convolutions with ReLUs, plus a skip, then a ReLU.

    The skip is the identity when the widths match and a 1x1 convolution when they do not — which
    happens exactly once in this checkpoint, at slot 13 (96 -> 64), the only slot carrying a
    seventh tensor.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = [
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
        ]
        self.skip = (nn.Conv2d(in_channels, out_channels, 1, bias=False)
                     if in_channels != out_channels else None)

    def __call__(self, x: mx.array) -> mx.array:
        h = self.conv[0](x)
        h = self.conv[1](nn.relu(h))
        h = self.conv[2](nn.relu(h))
        residual = x if self.skip is None else self.skip(x)
        return nn.relu(h + residual)


def _upsample(x: mx.array) -> mx.array:
    """Nearest-neighbour x2 on a channels-last ``(N, H, W, C)`` array."""
    x = mx.repeat(x, 2, axis=1)
    return mx.repeat(x, 2, axis=2)


class TAEDecoder(nn.Module):
    """The flat 23-slot decoder. Input ``(N, H, W, 24)``, output ``(N, 16H, 16W, 3)`` in [0, 1]."""

    def __init__(self):
        super().__init__()
        self.conv_in = nn.Conv2d(LATENT_CHANNELS, 96, 3, padding=1)          # slot 1
        self.stage1 = [Block(96, 96) for _ in range(3)]                      # slots 3, 4, 5
        self.up1 = nn.Conv2d(96, 96, 3, padding=1, bias=False)               # slot 7
        self.stage2 = [Block(96, 96) for _ in range(3)]                      # slots 8, 9, 10
        self.up2 = nn.Conv2d(96, 96, 3, padding=1, bias=False)               # slot 12
        self.stage3 = [Block(96, 64), Block(64, 64), Block(64, 64)]          # slots 13, 14, 15
        self.up3 = nn.Conv2d(64, 64, 3, padding=1, bias=False)               # slot 17
        self.stage4 = [Block(64, 64) for _ in range(2)]                      # slots 18, 19
        self.up4 = nn.Conv2d(64, 64, 3, padding=1, bias=False)               # slot 21
        self.stage5 = [Block(64, 64)]                                        # slot 22
        self.conv_out = nn.Conv2d(64, 3, 3, padding=1)                       # slot 23

    def __call__(self, latents: mx.array) -> mx.array:
        # Slot 0: TAESD clamps its input rather than trusting the sampler's tails.
        x = mx.tanh(latents / 3.0) * 3.0
        x = nn.relu(self.conv_in(x))                                         # slots 1, 2
        for block in self.stage1:
            x = block(x)
        x = self.up1(_upsample(x))                                           # slots 6, 7
        for block in self.stage2:
            x = block(x)
        x = self.up2(_upsample(x))                                           # slots 11, 12
        for block in self.stage3:
            x = block(x)
        x = self.up3(_upsample(x))                                           # slots 16, 17
        for block in self.stage4:
            x = block(x)
        x = self.up4(_upsample(x))                                           # slots 20, 21
        for block in self.stage5:
            x = block(x)
        # No `+ 0.5` here. TAESD's decoder for Stable Diffusion adds one, and this port did too
        # until it was measured: with it, frames came out at mean brightness 225/255 and scored
        # 5 dB against the real VAE; without it, 17.5 dB and a recognisable image. That
        # convention belongs to SD's latent space, not to this checkpoint.
        return self.conv_out(x)


#: Checkpoint slot -> module attribute. The gaps (0, 2, 6, 11, 16, 20) are parameterless layers.
SLOT_MAP = {
    1: "conv_in",
    3: "stage1.0", 4: "stage1.1", 5: "stage1.2",
    7: "up1",
    8: "stage2.0", 9: "stage2.1", 10: "stage2.2",
    12: "up2",
    13: "stage3.0", 14: "stage3.1", 15: "stage3.2",
    17: "up3",
    18: "stage4.0", 19: "stage4.1",
    21: "up4",
    22: "stage5.0",
    23: "conv_out",
}


#: Inside a checkpoint `Block`, the three convolutions sit at Sequential slots 0, 2, 4 (1 and 3
#: are the interleaved ReLUs). `Block` here holds them in a plain 3-element `self.conv` list
#: instead, at indices 0, 1, 2 — so the inner index needs the same kind of gap-aware remap as the
#: outer 23-slot layout, one level down.
_CONV_SUBSLOT = {0: 0, 2: 1, 4: 2}


def _parameter_path(key: str) -> str:
    """``13.conv.0.weight`` -> ``stage3.0.conv.0.weight``; ``1.weight`` -> ``conv_in.weight``."""
    slot, _, rest = key.partition(".")
    try:
        prefix = SLOT_MAP[int(slot)]
    except (ValueError, KeyError):
        raise KeyError(f"{key!r} does not belong to a known decoder slot") from None

    parts = rest.split(".")
    if parts[0] == "conv":
        try:
            parts[1] = str(_CONV_SUBSLOT[int(parts[1])])
        except (IndexError, ValueError, KeyError):
            raise KeyError(f"{key!r} has an unrecognized conv sub-slot") from None
        rest = ".".join(parts)
    return f"{prefix}.{rest}"


#: Parameter-path prefix -> the checkpoint slot number it came from, e.g. ``"stage5.0" -> 22``.
#: Used only to make a strict-loading failure name the slot a human can look up in the checkpoint,
#: since `missing` below is expressed in module parameter-path terms and never contains the
#: original slot number on its own.
_REVERSE_SLOT_MAP = {prefix: slot for slot, prefix in SLOT_MAP.items()}


def _missing_checkpoint_slots(missing_params: list[str]) -> list[int]:
    """Map missing module parameter paths back to the checkpoint slot numbers they belong to."""
    slots = set()
    for name in missing_params:
        prefix = max(
            (p for p in _REVERSE_SLOT_MAP if name == p or name.startswith(p + ".")),
            key=len, default=None,
        )
        if prefix is not None:
            slots.add(_REVERSE_SLOT_MAP[prefix])
    return sorted(slots)


def load_tae(path: Path | str = TAE_WEIGHTS_PATH, report: bool = False):
    """Build a `TAEDecoder` and fill every one of its parameters from `path`.

    Refuses a checkpoint that does not account for the module tree exactly. A decoder with a few
    blocks left at their initialization values does not raise and does not look obviously wrong —
    it produces plausible noise, which is the worst failure mode a preview can have.
    """
    from mlx.utils import tree_flatten, tree_unflatten

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No TAE weights at {path}")

    raw = mx.load(str(path))
    decoder = TAEDecoder()
    expected = {name: param.shape for name, param in tree_flatten(decoder.parameters())}

    resolved: dict[str, mx.array] = {}
    unused: list[str] = []
    wrong_shape: list[str] = []
    for key, tensor in raw.items():
        try:
            name = _parameter_path(key)
        except KeyError:
            unused.append(key)
            continue
        if name not in expected:
            unused.append(key)
            continue
        prepared = to_mlx_conv2d_layout(tensor) if tensor.ndim == 4 else tensor
        # Names alone are not enough. A checkpoint whose `1.weight` is 5x5 instead of 3x3 matches
        # by name, loads without complaint, and even runs a forward pass — MLX rebinds the
        # parameter rather than checking it against the module that declared it. The result is a
        # decoder quietly running a different architecture from the one this file describes.
        if prepared.shape != expected[name]:
            wrong_shape.append(f"{key} -> {name}: {tuple(prepared.shape)} != {tuple(expected[name])}")
            continue
        resolved[name] = prepared

    missing = sorted(expected.keys() - resolved.keys())
    if missing or unused or wrong_shape:
        missing_slots = _missing_checkpoint_slots(missing)
        raise KeyError(
            f"{path.name} does not match the decoder: {len(missing)} parameters unfilled "
            f"(e.g. {missing[:3]}) from checkpoint slot(s) {missing_slots}, "
            f"{len(unused)} tensors unused (e.g. {sorted(unused)[:3]}), "
            f"{len(wrong_shape)} of the wrong shape (e.g. {sorted(wrong_shape)[:3]})."
        )

    decoder.update(tree_unflatten(list(resolved.items())))
    mx.eval(decoder.parameters())
    if report:
        return decoder, {"loaded": len(resolved), "missing": missing, "unused": unused,
                         "wrong_shape": wrong_shape}
    return decoder


def decode_latent_frame(decoder: TAEDecoder, latents: mx.array, latents_mean: mx.array,
                        latents_std: mx.array, frame_index: int):
    """One latent frame -> one ``(H, W, 3)`` uint8 RGB frame at 16x the latent resolution.

    ``latents`` is ``(1, C, F, H, W)`` as the sampler holds it, i.e. normalized. Whether that is
    fed straight in or denormalized first is `TAE_EXPECTS_NORMALIZED`, settled by measurement.
    """
    import numpy as np

    if not TAE_EXPECTS_NORMALIZED:
        latents = latents * latents_std + latents_mean
    hwc = mx.transpose(latents[0, :, frame_index], (1, 2, 0))[None].astype(mx.float32)
    out = decoder(hwc)
    mx.eval(out)
    return (np.clip(np.array(out)[0], 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
