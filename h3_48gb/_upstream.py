"""Import `minimax_h3_mlx` out of the vendored `upstream/` checkout.

Almost everything this package changes is applied from the outside — subclasses, instance-level
substitutions, replacement loaders — so upstream can be fast-forwarded without unpicking our work.
The one exception is `patches/0001-keyframe-masked-scatter.patch`, a control-flow bug in the middle
of `encode()` that no loader-side substitution can reach; it is carried as a patch file, applied
during setup, and `h3_48gb.text_encoder.keyframe_scatter_patch_applied` checks for it before any
keyframe run starts.
"""

from __future__ import annotations

import sys
from pathlib import Path

#: The vendored checkout. Override with `H3_UPSTREAM` if the tree lives elsewhere.
UPSTREAM = Path(__file__).resolve().parent.parent / "upstream"


def ensure_on_path(upstream: Path | str | None = None) -> Path:
    """Put the upstream checkout on ``sys.path`` (idempotent) and return it."""
    import os

    root = Path(upstream or os.environ.get("H3_UPSTREAM") or UPSTREAM).expanduser().resolve()
    if not (root / "minimax_h3_mlx").is_dir():
        raise FileNotFoundError(
            f"No `minimax_h3_mlx` package under {root}. Clone PipeNetwork/minimax-h3-mlx into "
            "`upstream/` or set H3_UPSTREAM."
        )
    path = str(root)
    if path not in sys.path:
        sys.path.insert(0, path)
    return root


ensure_on_path()
