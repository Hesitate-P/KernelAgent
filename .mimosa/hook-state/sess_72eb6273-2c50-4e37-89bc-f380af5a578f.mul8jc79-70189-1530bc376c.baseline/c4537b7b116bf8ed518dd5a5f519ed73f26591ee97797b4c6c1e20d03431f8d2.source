"""Shared filesystem safety helpers (contest-hardening fork).

The upstream code base writes generated kernels, prompts, logs and metrics to
paths that are partly derived from configuration or CLI input.  To make every
such write provably confined, all artifact writes in this fork go through
``safe_join`` / ``assert_within``, which resolve the path and refuse anything
that escapes its intended base directory.
"""

from __future__ import annotations

import os
from pathlib import Path


def _resolve(path: os.PathLike[str] | str) -> str:
    return os.path.realpath(os.path.abspath(os.fspath(path)))


def assert_within(base: os.PathLike[str] | str, target: os.PathLike[str] | str) -> Path:
    """Return ``target`` as a Path after verifying it stays inside ``base``.

    Raises ``ValueError`` when the resolved ``target`` escapes ``base``
    (covers ``..`` segments, absolute-path overrides and symlink tricks).
    """
    base_r = _resolve(base)
    target_r = _resolve(target)
    if target_r != base_r and not target_r.startswith(base_r + os.sep):
        raise ValueError(
            f"unsafe path {os.fspath(target)!r} escapes base directory {base_r!r}"
        )
    return Path(target_r)


def safe_join(base: os.PathLike[str] | str, *names: str) -> Path:
    """Join ``names`` under ``base`` and verify the result stays there.

    Each name is additionally sanity-checked: it must be a single path
    component (no separators, no ``..``), which keeps the join predictable
    even before resolution.
    """
    for name in names:
        parts = os.fspath(name).split("/")
        if os.name == "nt":  # pragma: no cover - windows is not a target here
            parts = [p for seg in parts for p in seg.split("\\")]
        for part in parts:
            if part in ("", ".", ".."):
                raise ValueError(f"unsafe path component {name!r}")
    return assert_within(base, os.path.join(os.fspath(base), *names))


def sanitize_component(name: str, *, fallback: str = "unnamed") -> str:
    """Reduce ``name`` to a filename-safe single component."""
    cleaned = "".join(c if (c.isalnum() or c in "-_.") else "_" for c in os.path.basename(name))
    cleaned = cleaned.strip("._")
    return cleaned or fallback
