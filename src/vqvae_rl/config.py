from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml


class Config(dict):
    """Dict with attribute access for small experiment configs."""

    def __getattr__(self, key: str) -> Any:
        try:
            value = self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc
        if isinstance(value, dict) and not isinstance(value, Config):
            value = Config(value)
            self[key] = value
        return value


def _wrap(value: Any) -> Any:
    if isinstance(value, dict):
        return Config({k: _wrap(v) for k, v in value.items()})
    if isinstance(value, list):
        return [_wrap(v) for v in value]
    return value


def _expand_environment(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _expand_environment(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_expand_environment(item) for item in value]
    if isinstance(value, str):
        return os.path.expanduser(os.path.expandvars(value))
    return value


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _load_raw(path: Path, parents: tuple[Path, ...] = ()) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    if resolved in parents:
        chain = " -> ".join(str(item) for item in (*parents, resolved))
        raise ValueError(f"Circular base_config chain: {chain}")
    with resolved.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config root must be a mapping: {resolved}")
    base_config = data.pop("base_config", None)
    if base_config is None:
        return data
    base_path = Path(str(base_config)).expanduser()
    if not base_path.is_absolute():
        base_path = resolved.parent / base_path
    base = _load_raw(base_path, (*parents, resolved))
    return _deep_merge(base, data)


def load_config(path: str | Path) -> Config:
    return _wrap(_expand_environment(_load_raw(Path(path))))
