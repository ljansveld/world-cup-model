"""
Repo-relative paths.

Everything that touches a file resolves through here, so the code works no matter
which directory you run it from -- `python simulate.py`, `python validation/calibrate.py`
from the repo root, or any of them from somewhere else entirely.
"""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"


def data_path(name: str) -> Path:
    """Absolute path to a file in data/ (whether or not it exists yet)."""
    return DATA_DIR / name


def ensure_data_dir() -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    return DATA_DIR
