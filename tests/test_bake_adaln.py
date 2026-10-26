"""The baked table must reproduce the shipped one — all of it, not just the blocks.

`scripts/bake_adaln.py` reconstructs the AdaLN modulation table from the pruned base's folded
time curve, which is what lets this fork run any step count instead of only 31. It was verified
against the shipped table block by block, and `final_modulations` was not checked at all — so a
bug that evaluated the output layer on the wrong clock and sliced its 10752 outputs into three
chunks of 3584 went unnoticed through a full day of experiments. Nothing raised: the output layer
indexes that table by timestep alone, so a wrong-width row reads as a valid one.
"""
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

CHECKPOINT = Path.home() / "models/h3-converted"
CURVE = Path.home() / "models/turbo/adaln_curve.safetensors"
SHIPPED = CHECKPOINT / "transformer/adaln_cache.safetensors"

pytestmark = pytest.mark.skipif(
    not (CURVE.exists() and SHIPPED.exists()),
    reason="needs the converted checkpoint and the pruned base's time curve")


@pytest.fixture(scope="module")
def baked(tmp_path_factory):
    """Bake the shipped grid (31 points) so it can be compared against the shipped table."""
    dest = tmp_path_factory.mktemp("adaln") / "baked31.safetensors"
    subprocess.run([sys.executable, "scripts/bake_adaln.py", "31", "--out", str(dest)],
                   cwd=Path(__file__).resolve().parent.parent, check=True, capture_output=True)
    import mlx.core as mx
    return mx.load(str(dest)), mx.load(str(SHIPPED))


def test_every_tensor_has_the_shipped_shape(baked):
    mine, shipped = baked
    mismatched = {k: (tuple(shipped[k].shape), tuple(mine[k].shape))
                  for k in shipped if k in mine and shipped[k].shape != mine[k].shape}
    assert not mismatched, f"shapes differ from the shipped table: {mismatched}"

    # Every key the loader reads must be present. `time_embeddings` is in the shipped file but
    # nothing reads it — mere.run left it as a reference, so it is deliberately not reproduced.
    required = {"video_sigmas", "audio_sigmas", "final_modulations"}
    required |= {f"blocks.{i}.modulations" for i in range(50)}
    assert required <= set(mine), f"the bake is missing keys the loader reads: {required - set(mine)}"
    assert set(mine) <= set(shipped), f"the bake invents keys: {set(mine) - set(shipped)}"


@pytest.mark.parametrize("key", ["blocks.0.modulations", "blocks.25.modulations",
                                 "blocks.49.modulations", "final_modulations"])
def test_values_match_the_shipped_table(baked, key):
    """`final_modulations` is in this list deliberately — it is the one that was wrong."""
    import mlx.core as mx

    mine, shipped = baked
    a = np.array(shipped[key].astype(mx.float32))
    b = np.array(mine[key].astype(mx.float32))
    relative = np.abs(a - b).max() / (np.abs(a).max() + 1e-9)
    # The curve is a folded 8-dim reconstruction, so it is not bit-exact. 1e-2 sits an order
    # above the observed 4e-3..8e-3 and two orders below the 0.165 this build's Q4 weights carry.
    assert relative < 1e-2, f"{key}: relative error {relative:.2e}"


def test_the_final_layer_is_not_a_sliced_video_clock(baked):
    """The specific bug: three variants must differ, and must not be thirds of one vector.

    Evaluating only the video clock and reshaping gives three chunks of a single vector. Variant 2
    is the conditioning anchor pinned at t=0.999, so it must differ from variant 0 at every step
    where the video clock is not itself 0.999.
    """
    import mlx.core as mx

    mine, _ = baked
    final = np.array(mine["final_modulations"].astype(mx.float32))
    assert final.shape[-1] == 10752, f"expected shift+scale = 2 * 5376, got {final.shape[-1]}"
    spread = np.abs(final[:, 0] - final[:, 2]).max(axis=-1)
    assert (spread > 1e-3).sum() > len(spread) // 2, (
        "the video clock and the conditioning anchor produce near-identical rows — the variants "
        "are probably slices of one evaluation rather than three")


def test_tail_split_keeps_the_simple_prefix_bit_identical():
    """The tail-split grid's only reason to exist is that its prefix is exactly `simple`'s.

    A run on it therefore reproduces a `simple` run of the same seed step for step until the
    final jump, which is what makes "did the extra tail steps help?" answerable — comparing 8
    steps against 16 cannot answer it, because a grid that differs from step one moves the
    composition, and two clips that frame the shot differently cannot be compared on detail.
    A float64 intermediate anywhere in this construction would break that silently: the sigmas
    would look right to three decimals and the trajectory would already have diverged.
    """
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
    from bake_adaln import tail_split_sigmas

    from h3_48gb import _upstream  # noqa: F401
    from minimax_h3_mlx.scheduler import MiniMaxH3Scheduler

    for steps, shift in ((8, 12.0), (8, 3.0), (16, 12.0), (31, 12.0)):
        sched = MiniMaxH3Scheduler(shift=shift)
        sched.set_timesteps(steps)
        simple = np.array(sched.sigmas.tolist(), np.float32)

        assert np.array_equal(tail_split_sigmas(steps, shift, 1), simple), (
            f"split=1 must leave `simple` untouched at {steps} steps, shift {shift}")

        for split in (2, 3, 4):
            grid = tail_split_sigmas(steps, shift, split)
            shared = len(simple) - 1          # everything up to and including the last live sigma
            assert np.array_equal(grid[:shared], simple[:shared]), (
                f"split={split} moved the shared prefix at {steps} steps, shift {shift}")
            assert len(grid) == len(simple) + split - 1
            assert grid[-1] == 0.0, "the grid must still end at the terminal sigma"
            assert np.all(np.diff(grid) < 0), "the scheduler requires a strictly decreasing grid"
