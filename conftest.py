"""Make the checkpoint-dependent skips loud.

Twelve tests — the whole of `tests/test_image_processor.py`, including the numerical equivalence
against real `transformers` that the committed fixtures exist for — are skipped when
`~/models/h3-converted` is absent. `pytest -q` reports that as twelve `s` characters among the
dots, which reads as "all green" on a machine that has never held the weights: every torch-free
guarantee in this fork would be unverified and nothing would say so.
"""
import atexit
import importlib.util
import os
import shutil
import tempfile
from pathlib import Path

import pytest


def _fake_models_home() -> None:
    """Give the CLI's default model paths something to point at when the weights are absent.

    `h3_48gb.cli` binds `~/models/h3-8bit-full` and `~/models/turbo/...` as argument defaults at
    import time, and the face-refine tests validate those paths before anything is mocked. On a
    machine without the weights (every fresh clone) they would fail for a reason that has nothing
    to do with the code under test. When the default checkpoint is missing, point HOME at a
    throwaway directory holding empty stand-ins. Tests that read real weights are unaffected: they
    already skip when the converted checkpoint is absent.
    """
    if (Path.home() / "models/h3-8bit-full").exists():
        return
    home = Path(tempfile.mkdtemp(prefix="h3-test-home-"))
    atexit.register(shutil.rmtree, home, ignore_errors=True)
    (home / "models/h3-8bit-full").mkdir(parents=True)
    (home / "models/turbo").mkdir(parents=True)
    (home / "models/turbo/minimax_h3_turbo_v4_step600_ema.safetensors").touch()
    (home / "models/turbo/adaln_8_l100.safetensors").touch()
    os.environ["HOME"] = str(home)


_fake_models_home()

CHECKPOINT = Path.home() / "models/h3-converted"

#: marker name -> (modules that must all be importable, the skip reason). The reason text is what
#: `pytest_terminal_summary` below counts, so it doubles as a stable tag.
_MODULE_MARKERS = {
    "mlx": (("mlx",), "mlx: needs the MLX stack, absent here"),
    "cv": (("cv2", "scipy"), "cv: needs opencv-python and scipy, absent here"),
}


def _missing(modules) -> bool:
    return any(importlib.util.find_spec(name) is None for name in modules)


def pytest_runtest_setup(item):
    for marker, (modules, reason) in _MODULE_MARKERS.items():
        mark = item.get_closest_marker(marker)
        if mark is not None and _missing(modules):
            pytest.skip(mark.kwargs.get("reason", reason))


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    skipped_reports = terminalreporter.stats.get("skipped", [])
    for marker, (_modules, reason) in _MODULE_MARKERS.items():
        count = sum(1 for report in skipped_reports
                    if reason.split(" ")[0] + " needs" in str(getattr(report, "longrepr", "")))
        if count:
            terminalreporter.write_sep("=", f"{count} skipped: {reason}", yellow=True)
    if CHECKPOINT.exists():
        return
    skipped = len(terminalreporter.stats.get("skipped", []))
    if not skipped:
        return
    terminalreporter.write_sep("=", f"{skipped} tests skipped: no converted checkpoint",
                               yellow=True, bold=True)
    terminalreporter.write_line(
        f"  {CHECKPOINT} is absent, so the tests that read real weights did not run — including "
        "the processor's equivalence with transformers. A green run here is not a full run.")
