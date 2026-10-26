"""`h3_48gb.canvas` is the mlx-free copy of upstream's canvas arithmetic: pin its numbers, and
(where MLX is installed) pin it against the upstream function itself."""
import pytest

from h3_48gb import canvas


@pytest.mark.parametrize("size,expected", [
    ((512, 512), (768, 768)),       # square: short edge 768
    ((1920, 1080), (768, 1344)),    # 16:9: 1365x768 is over the area cap, scaled to 1354x762, rounds to 1344x768
    ((1080, 1920), (1344, 768)),
    ((4000, 1000), (512, 2016)),    # 4:1: capped to 2032x508, 63.498 columns of 32 -> 2016
    ((300, 200), (768, 1152)),
])
def test_canvas_size_is_height_then_width(size, expected):
    assert canvas.resolve_canvas_size(*size) == expected


@pytest.mark.parametrize("size", [(100, 500), (500, 100), (0, 10), (10, -1)])
def test_unsupported_aspect_is_a_value_error(size):
    with pytest.raises(ValueError):
        canvas.resolve_canvas_size(*size)


@pytest.mark.mlx
@pytest.mark.parametrize("size", [(512, 512), (1920, 1080), (1080, 1920), (4000, 1000), (1000, 4000),
                                  (300, 200), (777, 333), (1, 4), (4, 1)])
def test_matches_upstream_packing(size):
    from h3_48gb._upstream import ensure_on_path  # noqa: F401
    from minimax_h3_mlx import packing
    assert canvas.resolve_canvas_size(*size) == packing.resolve_canvas_size(*size)
