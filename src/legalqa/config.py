from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg


def resolve_path(project_root: str | Path, value: str | None) -> Path | None:
    if value is None or value == "":
        return None
    p = Path(value)
    if p.is_absolute():
        return p
    return Path(project_root) / p


def ensure_parent(path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
