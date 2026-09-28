"""Minimal .env loader (no dependency). Existing environment variables win."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parents[1]


def load_dotenv(path: Optional[Path] = None) -> Optional[Path]:
    candidates = [path] if path else [Path.cwd() / ".env", REPO_ROOT / ".env"]
    for p in candidates:
        if p and p.is_file():
            for raw in p.read_text(encoding="utf-8").splitlines():
                line = raw.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                if line.startswith("export "):
                    line = line[len("export "):]
                key, _, value = line.partition("=")
                key, value = key.strip(), value.strip()
                if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
                    value = value[1:-1]
                elif " #" in value:
                    value = value.split(" #", 1)[0].rstrip()
                if key and key not in os.environ:
                    os.environ[key] = value
            return p
    return None
