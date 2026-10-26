"""The CLI and its mlx-free helpers import where `upstream/` does not exist (a fresh clone)."""
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PANEL = ("h3_48gb", "h3_48gb.cli", "h3_48gb.runs", "h3_48gb.canvas", "h3_48gb.framecheck")


def _run(tmp_path, code):
    empty = tmp_path / "no-upstream"
    empty.mkdir(exist_ok=True)
    env = {**os.environ, "H3_UPSTREAM": str(empty), "PYTHONPATH": str(ROOT)}
    return subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                          env=env, cwd=tmp_path)


def test_mlx_free_modules_import_without_upstream(tmp_path):
    done = _run(tmp_path, "import importlib\n"
                f"for m in {PANEL!r}:\n    importlib.import_module(m)\n"
                "import sys; print('minimax_h3_mlx' in sys.modules)")
    assert done.returncode == 0, done.stderr
    assert done.stdout.strip() == "False"


def test_missing_upstream_is_reported_when_an_mlx_module_is_used(tmp_path):
    done = _run(tmp_path, "import h3_48gb\nh3_48gb.ensure_on_path()")
    assert done.returncode != 0
    assert "No `minimax_h3_mlx` package under" in done.stderr
    assert str(tmp_path / "no-upstream") in done.stderr
    done = _run(tmp_path, "import h3_48gb._upstream")
    assert "No `minimax_h3_mlx` package under" in done.stderr
