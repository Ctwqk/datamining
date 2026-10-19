"""Execute notebooks headlessly (nbclient) so their committed outputs are current.

    uv run python scripts/run_notebooks.py                      # all notebooks, in place
    uv run python scripts/run_notebooks.py --out-dir /tmp/nb    # leave the originals untouched
    uv run python scripts/run_notebooks.py hw7_titanic_survival/titanic_visualization.ipynb

Each notebook runs with its own folder as working directory, so relative data paths work.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient

REPO = Path(__file__).resolve().parents[1]


def find_notebooks() -> list[Path]:
    skip = {".venv", ".ipynb_checkpoints", "node_modules"}
    return sorted(p for p in REPO.rglob("*.ipynb") if not skip & set(p.relative_to(REPO).parts))


def execute(path: Path, out: Path, timeout: int) -> None:
    nb = nbformat.read(path, as_version=4)
    NotebookClient(
        nb,
        timeout=timeout,
        kernel_name="python3",
        resources={"metadata": {"path": str(path.parent)}},
    ).execute()
    nbformat.validate(nb)
    out.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(nb, out)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("notebooks", nargs="*", type=Path, help="notebooks (default: all)")
    parser.add_argument("--out-dir", type=Path, help="write executed copies here, not in place")
    parser.add_argument("--timeout", type=int, default=900, help="seconds per cell")
    args = parser.parse_args(argv)

    paths = [p.resolve() for p in args.notebooks] or find_notebooks()
    for path in paths:
        rel = path.relative_to(REPO)
        target = args.out_dir / rel if args.out_dir else path
        print(f"executing {rel} ...", flush=True)
        execute(path, target, args.timeout)
    print(f"{len(paths)} notebook(s) executed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
