"""Resolve project root for local dev, Docker, and packaged installs."""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=1)
def studio_root() -> Path:
    for key in ("VIDEO_CLIPPY_ROOT", "VIDEO_STUDIO_ROOT"):
        if env := os.environ.get(key, "").strip():
            root = Path(env).expanduser().resolve()
            if root.is_dir():
                return root

    candidate = Path(__file__).resolve().parent.parent
    if _looks_like_root(candidate):
        return candidate

    raise RuntimeError(
        "Cannot find Video Clippy root. Set VIDEO_CLIPPY_ROOT to the project directory."
    )


def _looks_like_root(path: Path) -> bool:
    return (path / "video-use" / "helpers" / "render.py").is_file()


def footage_dir() -> Path:
    return studio_root() / "footage"


def edit_dir() -> Path:
    return footage_dir() / "edit"


def helpers_dir() -> Path:
    return studio_root() / "video-use" / "helpers"


def outro_path() -> Path:
    return footage_dir() / "outro.mp4"
