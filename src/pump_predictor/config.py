"""Load config.yaml into a simple attribute-access config object."""

from __future__ import annotations

from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Cfg(dict):
    """Dict with attribute access, recursively wrapping nested dicts."""

    def __getattr__(self, name):
        try:
            v = self[name]
        except KeyError as e:
            raise AttributeError(name) from e
        return Cfg(v) if isinstance(v, dict) else v


def load_config(path: str | Path | None = None) -> Cfg:
    path = Path(path) if path else PROJECT_ROOT / "config.yaml"
    with open(path, encoding="utf-8") as f:
        cfg = Cfg(yaml.safe_load(f))
    # Resolve path entries relative to the project root.
    for key, val in cfg["paths"].items():
        p = Path(val)
        cfg["paths"][key] = str(p if p.is_absolute() else PROJECT_ROOT / p)
    return cfg
