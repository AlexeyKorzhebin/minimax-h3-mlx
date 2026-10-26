"""Entry point for `python -m h3_48gb`.

`h3_48gb/cli.py` defines `main()` but, on its own, `python -m h3_48gb.cli` does nothing: a module
run with `-m` only executes if it has an `if __name__ == "__main__":` guard, and CLI modules are
conventionally imported (e.g. by tests) without wanting that side effect. This file is the
package's dedicated entry point instead. See also the `h3` console script installed by
`pyproject.toml`, which is the recommended way to invoke the CLI.
"""

from __future__ import annotations

from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())
