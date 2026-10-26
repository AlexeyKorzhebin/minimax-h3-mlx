"""MiniMax-H3 canvas arithmetic without `mlx`.

`resolve_canvas_size` is copied from `upstream/minimax_h3_mlx/packing.py`, which imports `mlx.core`
at module scope and so cannot be imported by anything that must stay mlx-free (a machine with no
MLX). The function is pure arithmetic over four constants; `tests/test_canvas.py` pins this copy
against the upstream one wherever MLX is installed, so a drift is a failing test on the Mac.
"""
from __future__ import annotations

SHORT_EDGE = 768
MAX_PIXELS = 768 * 1344
CANVAS_MULTIPLE = 32
MIN_ASPECT_RATIO = 1 / 4
MAX_ASPECT_RATIO = 4


def resolve_canvas_size(aspect_width: float, aspect_height: float) -> tuple[int, int]:
    """Display aspect ratio -> ``(height, width)`` of the canvas; see the upstream docstring."""
    if aspect_width <= 0 or aspect_height <= 0:
        raise ValueError(f"The aspect ratio must be positive, got {aspect_width}:{aspect_height}.")

    ratio = aspect_width / aspect_height
    if not MIN_ASPECT_RATIO <= ratio <= MAX_ASPECT_RATIO:
        raise ValueError(
            f"MiniMax-H3 supports aspect ratios from 1:4 to 4:1, got "
            f"{aspect_width}:{aspect_height} ({ratio:g})."
        )

    if ratio >= 1.0:
        width, height = SHORT_EDGE * ratio, float(SHORT_EDGE)
    else:
        width, height = float(SHORT_EDGE), SHORT_EDGE / ratio

    area = width * height
    if area > MAX_PIXELS:
        scale = (MAX_PIXELS / area) ** 0.5
        width, height = width * scale, height * scale

    m = CANVAS_MULTIPLE
    return max(m, round(height / m) * m), max(m, round(width / m) * m)
