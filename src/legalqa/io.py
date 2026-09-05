from __future__ import annotations

import json
import os
import zipfile
from pathlib import Path
from typing import Iterator


def iter_context_documents(source_path: str | Path) -> Iterator[dict]:
    """Yield context_*.json-like objects from a ZIP archive or directory."""
    source_path = Path(source_path)

    if source_path.is_file() and source_path.suffix.lower() == ".zip":
        with zipfile.ZipFile(source_path) as zf:
            for filename in sorted(zf.namelist()):
                if not filename.lower().endswith(".json"):
                    continue
                raw = zf.read(filename)
                data = json.loads(raw.decode("utf-8-sig"))
                if isinstance(data, dict) and "passage" in data:
                    yield data
        return

    if source_path.is_dir():
        for root, _, files in os.walk(source_path):
            for filename in sorted(files):
                if not filename.lower().endswith(".json"):
                    continue
                path = Path(root) / filename
                with path.open("r", encoding="utf-8-sig") as f:
                    data = json.load(f)
                if isinstance(data, dict) and "passage" in data:
                    yield data
        return

    raise FileNotFoundError(f"Context source not found or unsupported: {source_path}")
