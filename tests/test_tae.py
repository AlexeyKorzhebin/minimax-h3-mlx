import pytest

pytest.importorskip("mlx.core", reason="mlx: needs the MLX stack, absent here")

from pathlib import Path

import numpy as np
import pytest

from h3_48gb.tae import TAE_WEIGHTS_PATH, to_mlx_conv2d_layout

pytestmark = [pytest.mark.mlx,
              pytest.mark.skipif(not TAE_WEIGHTS_PATH.exists(),
                                 reason=f"no TAE weights at {TAE_WEIGHTS_PATH}")]


def test_conv_layout_is_channels_last_by_value_not_by_shape():
    import mlx.core as mx

    source = mx.random.normal((5, 4, 3, 3))          # (out, in, kH, kW)
    out = to_mlx_conv2d_layout(source)
    assert out.shape == (5, 3, 3, 4)

    reference = np.zeros((5, 3, 3, 4), dtype=np.float32)
    src = np.array(source)
    for o in range(5):
        for i in range(4):
            for h in range(3):
                for w in range(3):
                    reference[o, h, w, i] = src[o, i, h, w]
    assert np.array_equal(np.array(out), reference)


def test_a_wrong_permutation_would_have_the_same_shape():
    """Without this, the test above could be satisfied by a transpose that swaps kH and kW."""
    import mlx.core as mx

    source = mx.random.normal((5, 4, 3, 3))
    right = to_mlx_conv2d_layout(source)
    wrong = source.transpose(0, 3, 2, 1)              # kW and kH swapped
    assert right.shape == wrong.shape, "the shapes must collide, or this test proves nothing"
    assert float(mx.abs(right - wrong).max()) > 0.0, "and the values must differ"


SLOT_TO_PARAM_SAMPLES = [
    ("1.weight", "conv_in.weight", (96, 3, 3, 24)),
    ("13.skip.weight", "stage3.0.skip.weight", (64, 1, 1, 96)),
    ("23.weight", "conv_out.weight", (3, 3, 3, 64)),
]


def test_every_tensor_lands_and_nothing_is_invented():
    from h3_48gb.tae import load_tae

    decoder, report = load_tae(TAE_WEIGHTS_PATH, report=True)
    assert report["missing"] == [], f"parameters no tensor filled: {report['missing']}"
    assert report["unused"] == [], f"tensors that matched no parameter: {report['unused']}"
    assert report["loaded"] == 81, f"expected all 81 tensors, got {report['loaded']}"


@pytest.mark.parametrize("checkpoint_key,param_path,expected_shape", SLOT_TO_PARAM_SAMPLES)
def test_named_slots_reach_their_parameters_with_the_right_shape(checkpoint_key, param_path,
                                                                 expected_shape):
    from mlx.utils import tree_flatten

    from h3_48gb.tae import load_tae

    decoder = load_tae(TAE_WEIGHTS_PATH)
    params = dict(tree_flatten(decoder.parameters()))
    assert param_path in params, f"{param_path} is not a parameter of the module tree"
    assert params[param_path].shape == expected_shape


def test_a_missing_weights_file_says_so_plainly(tmp_path):
    from h3_48gb.tae import load_tae

    with pytest.raises(FileNotFoundError) as excinfo:
        load_tae(tmp_path / "absent.safetensors")
    assert "absent.safetensors" in str(excinfo.value)


def test_a_truncated_checkpoint_is_refused_rather_than_half_loaded(tmp_path):
    """A decoder with three blocks silently left at their init values produces plausible noise."""
    import mlx.core as mx

    full = mx.load(str(TAE_WEIGHTS_PATH))
    partial = {k: v for k, v in full.items() if not k.startswith("22.")}
    path = tmp_path / "partial.safetensors"
    mx.save_safetensors(str(path), partial)

    from h3_48gb.tae import load_tae

    with pytest.raises(KeyError) as excinfo:
        load_tae(path)
    assert "22" in str(excinfo.value), "the message must name what is missing"


def test_the_decoder_upsamples_by_exactly_sixteen():
    import mlx.core as mx

    from h3_48gb.tae import SPATIAL_RATIO, load_tae

    decoder = load_tae(TAE_WEIGHTS_PATH)
    out = decoder(mx.zeros((1, 8, 12, 24)))
    assert out.shape == (1, 8 * SPATIAL_RATIO, 12 * SPATIAL_RATIO, 3)


@pytest.mark.skipif(
    not (Path.home() / "models/h3-converted/video_vae").exists(),
    reason="no converted video_vae checkpoint",
)
def test_the_ratio_matches_the_video_vae():
    """If these ever diverge, a TAE preview would be a different size from a VAE one."""
    from h3_48gb.pipeline import video_vae_config
    from h3_48gb.tae import LATENT_CHANNELS, SPATIAL_RATIO

    cfg = video_vae_config(Path.home() / "models/h3-converted/video_vae")
    assert SPATIAL_RATIO == cfg.spatial_compression_ratio
    assert LATENT_CHANNELS == cfg.latent_channels


# -- values, not just shapes ---------------------------------------------------------------------

GOLDEN = Path(__file__).resolve().parent / "fixtures/tae/golden_frame.npz"


def _deterministic_latent():
    """A fixed input with no RNG, so the fixture is reproducible across MLX versions."""
    import mlx.core as mx

    channels, height, width = 24, 3, 4
    index = np.arange(channels * height * width, dtype=np.float32)
    index = index.reshape(1, channels, 1, height, width)
    return mx.array(np.sin(index / 7.0) * 2.0)


def test_the_decoder_reproduces_the_golden_frame():
    """Every test above this one passes on a decoder that computes the wrong thing.

    Eight separate mutations survived the shape-only suite: swapping two entries in `SLOT_MAP`,
    swapping two convolutions inside `Block`, turning the nearest-neighbour upsample into
    something else, dropping either ReLU, dropping the input clamp, and adding the `+ 0.5` this
    port carried until it was measured. All of them change the pixels and none of them change a
    shape, so a value fixture is the only thing that catches them.
    """
    import mlx.core as mx

    from h3_48gb.tae import decode_latent_frame, load_tae

    expected = np.load(GOLDEN)["frame"]
    got = decode_latent_frame(
        load_tae(TAE_WEIGHTS_PATH), _deterministic_latent(),
        mx.zeros((1, 24, 1, 1, 1)), mx.ones((1, 24, 1, 1, 1)), frame_index=0,
    )
    assert got.shape == expected.shape
    # uint8 output, so an exact match is the right bar — any real defect moves pixels far more
    # than a rounding boundary would.
    difference = np.abs(got.astype(np.int16) - expected.astype(np.int16))
    assert difference.max() <= 1, (
        f"decoded frame differs from the fixture by up to {difference.max()} levels "
        f"({(difference > 1).sum()} pixels beyond tolerance)")


def test_the_golden_frame_is_not_uniform():
    """A fixture of flat grey would match a decoder that had stopped working entirely."""
    expected = np.load(GOLDEN)["frame"]
    assert expected.std() > 10, f"the fixture carries no structure: std {expected.std():.1f}"
    assert 20 < expected.mean() < 235, f"the fixture is saturated: mean {expected.mean():.1f}"


def test_a_wrongly_shaped_tensor_is_refused_even_when_the_name_matches(tmp_path):
    """Names alone let a 5x5 kernel load into a 3x3 slot: MLX rebinds rather than checking."""
    import mlx.core as mx

    from h3_48gb.tae import load_tae

    raw = dict(mx.load(str(TAE_WEIGHTS_PATH)))
    raw["1.weight"] = mx.zeros((96, 24, 5, 5))          # right name, wrong kernel
    path = tmp_path / "wrong_shape.safetensors"
    mx.save_safetensors(str(path), raw)

    with pytest.raises(KeyError) as excinfo:
        load_tae(path)
    assert "1.weight" in str(excinfo.value)


def test_every_fixture_is_actually_committed():
    """A fixture the repository does not carry is a test that only passes on this machine.

    This has now happened twice: `.gitignore`'s blanket `*.npz` swallowed the processor fixtures,
    and then the TAE golden frame. Both times the suite stayed green locally and lost its
    value-level protection for everyone else. Checked for the whole fixtures tree rather than for
    one file, because the next one will be a different name.
    """
    import subprocess

    root = Path(__file__).resolve().parent.parent
    fixtures = root / "tests/fixtures"
    on_disk = {p.relative_to(root).as_posix() for p in fixtures.rglob("*") if p.is_file()}
    tracked = set(subprocess.run(
        ["git", "ls-files", "tests/fixtures"], cwd=root, capture_output=True, text=True,
        check=True).stdout.split())

    missing = sorted(on_disk - tracked)
    assert not missing, (
        f"these fixtures exist but git does not carry them: {missing}. Add an exception to "
        ".gitignore — a fixture that ships only on the author's machine protects nobody.")


def test_the_normalization_constant_actually_changes_the_decode():
    """`TAE_EXPECTS_NORMALIZED` decides what the decoder is fed; flipping it must change the frame.

    The golden fixture is taken at mean=0/std=1, where denormalizing is the identity — so it pins
    the decoder but says nothing about this constant. Measured against non-trivial statistics, the
    two paths differ, which is what makes RESULTS.md's normalization table meaningful.
    """
    import mlx.core as mx

    from h3_48gb import tae as tae_module
    from h3_48gb.tae import decode_latent_frame, load_tae

    decoder = load_tae(TAE_WEIGHTS_PATH)
    latents = _deterministic_latent()
    mean = mx.full((1, 24, 1, 1, 1), 0.4)
    std = mx.full((1, 24, 1, 1, 1), 2.5)

    original = tae_module.TAE_EXPECTS_NORMALIZED
    try:
        tae_module.TAE_EXPECTS_NORMALIZED = True
        as_normalized = decode_latent_frame(decoder, latents, mean, std, 0)
        tae_module.TAE_EXPECTS_NORMALIZED = False
        as_denormalized = decode_latent_frame(decoder, latents, mean, std, 0)
    finally:
        tae_module.TAE_EXPECTS_NORMALIZED = original

    difference = np.abs(as_normalized.astype(np.int16) - as_denormalized.astype(np.int16))
    assert difference.mean() > 5, (
        f"the two normalizations decode to nearly the same frame (mean diff {difference.mean():.2f})"
        " — then the constant, and the table in RESULTS.md, decide nothing")
    assert original is True, "measurement settled this at normalized; see RESULTS.md"
