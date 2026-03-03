from __future__ import annotations

import json
from pathlib import Path
from typing import Any

"""
config_file.py
--------------
Safe, section-isolated helpers for reading and writing config.json.

The file structure is:
{
  "extensions": [...],   <- managed by ExtensionRegistry / exiv register
  "settings":   {...}    <- managed by the server config API
}
"""

DEFAULT_SETTINGS: dict[str, Any] = {
    "log_level": 3,
    "low_vram": True,
    "no_oom": False,
    "normal_load": False,
    "auto_download": False,
}

DEFAULT_CONFIG: dict[str, Any] = {
    "extensions": [],
    "settings": DEFAULT_SETTINGS.copy(),
}

def load_config(config_file: Path) -> dict:
    """
    - reads config.json and returns the full dict
    - creates the file with defaults if it does not exist
    - ensures both 'extensions' and 'settings' keys are present
    """
    if not config_file.exists():
        _write_full(config_file, DEFAULT_CONFIG.copy())
        return DEFAULT_CONFIG.copy()

    try:
        with config_file.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        data = {}

    changed = False
    if not isinstance(data.get("extensions"), list):
        data["extensions"] = []
        changed = True
    if not isinstance(data.get("settings"), dict):
        data["settings"] = DEFAULT_SETTINGS.copy()
        changed = True

    if changed:
        _write_full(config_file, data)

    return data

def save_section(config_file: Path, section: str, value: Any) -> None:
    """
    - read-modify-write: updates ONLY `section` inside config.json
    - creates the file with defaults first if it does not exist
    """
    data = load_config(config_file)
    data[section] = value
    _write_full(config_file, data)

def _write_full(config_file: Path, data: dict) -> None:
    config_file.parent.mkdir(parents=True, exist_ok=True)
    with config_file.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
        f.write("\n")
