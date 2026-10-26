"""The `mlx`/`cv` markers: a test that needs a module this environment lacks is *skipped with a
reason*, never collected into an ImportError and never silently passed. Driven through a real
`pytest` subprocess on a throwaway test file, because the hook under test runs at collection time
of the outer session and cannot be observed from inside one test.
"""
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _run(tmp_path, body: str) -> subprocess.CompletedProcess:
    """The probe file lives *inside* tests/ (and is removed afterwards): a file outside the
    project root would not load the root conftest.py whose hook is under test."""
    test_file = PROJECT_ROOT / "tests" / f"test_zz_probe_marker_{tmp_path.name}.py"
    test_file.write_text(body, encoding="utf-8")
    try:
        return subprocess.run(
            [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "-rs",
             str(test_file)],
            capture_output=True, text=True, cwd=PROJECT_ROOT, timeout=120)
    finally:
        test_file.unlink(missing_ok=True)


def test_a_marked_test_whose_module_is_missing_is_skipped_with_the_marker_reason(tmp_path):
    result = _run(tmp_path, (
        "import pytest\n"
        "@pytest.mark.mlx\n"
        "def test_needs_mlx():\n"
        "    import module_that_never_exists_h3  # noqa: F401\n"
    ))
    # Only meaningful where mlx is absent (a venv without the MLX stack).
    import importlib.util
    if importlib.util.find_spec("mlx") is not None:
        assert "1 passed" in result.stdout or "1 failed" in result.stdout
        return
    assert result.returncode == 0, result.stdout + result.stderr
    assert "SKIPPED [1]" in result.stdout, result.stdout
    assert "mlx: needs the MLX stack, absent here" in result.stdout, result.stdout


def test_an_unmarked_test_still_fails_on_a_missing_module(tmp_path):
    result = _run(tmp_path, (
        "def test_unmarked():\n"
        "    import module_that_never_exists_h3  # noqa: F401\n"
    ))
    assert result.returncode == 1, result.stdout
    assert "ModuleNotFoundError" in result.stdout, result.stdout
