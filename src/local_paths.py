"""
Machine-specific data paths, kept out of the repository.

Some scripts default to directories on the author's cluster. The code only
contains the part of these paths below a root directory; the root directories
are read from ``local_paths.json`` in the repository root, which is git-ignored
(format: ``local_paths.example.json``):

    {"scratch": "/path/to/scratch/data", "cfs": "/path/to/project/directory"}

Without the file or the key, ``local_path`` returns None.
"""

from __future__ import annotations

import json
from pathlib import Path

# src/local_paths.py -> repository root
LOCAL_PATHS_FILE = Path(__file__).resolve().parents[1] / "local_paths.json"


def local_path(root: str, subpath: str) -> str | None:
    """Return ``<root directory>/<subpath>``, or None if ``root`` is not configured.

    Parameters
    ----------
    root : str
        Key in ``local_paths.json`` ("scratch" or "cfs").
    subpath : str
        Path below the root directory, with "/" separators.
    """
    if not LOCAL_PATHS_FILE.is_file():
        return None
    roots = json.loads(LOCAL_PATHS_FILE.read_text(encoding="utf-8"))
    base = roots.get(root)
    if not base:
        return None
    return base.rstrip("/") + "/" + subpath.lstrip("/")


def required_local_path(root: str, subpath: str) -> str:
    """Like :func:`local_path`, but raise an error if ``root`` is not configured."""
    path = local_path(root, subpath)
    if path is None:
        raise RuntimeError(
            f"No '{root}' directory configured: create {LOCAL_PATHS_FILE} "
            f"(see local_paths.example.json) or set the path in the script."
        )
    return path
