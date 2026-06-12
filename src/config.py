from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

try:
    import tomllib
except ModuleNotFoundError: # pragma: no cover - py311+ usa tomllib stdlib
    import tomli as tomllib


def _default_config_path() -> Path:
    return Path(__file__).resolve().parents[1] / "config.toml"


_MISSING = object()


class Config:
    def __init__(self, raw: Mapping[str, Any], path: Path) -> None:
        self._raw = dict(raw)
        self.path = path

    @classmethod
    def from_toml(cls, config_path: Path | str | None = None) -> "Config":
        path = Path(config_path) if config_path else _default_config_path()
        if not path.exists():
            return cls(raw={}, path=path)

        with path.open("rb") as f:
            raw = tomllib.load(f)
        if not isinstance(raw, Mapping):
            raise TypeError("Root of TOML config must be a table.")
        return cls(raw=raw, path=path)

    def _resolve_section(self, section: str) -> Mapping[str, Any]:
        if not section:
            return self._raw

        current: Any = self._raw
        for part in section.split("."):
            if not isinstance(current, Mapping) or part not in current:
                raise KeyError(f"Section `{section}` not found.")
            current = current[part]

        if not isinstance(current, Mapping):
            raise KeyError(f"`{section}` is not a section.")
        return current

    def get(self, section: str, key: str, default: Any = _MISSING) -> Any:
        section_data = self._resolve_section(section)
        if key in section_data:
            return section_data[key]
        if default is _MISSING:
            raise KeyError(f"Key `{key}` not found in section `{section}`.")
        return default


def load_config(config_path: Path | str | None = None) -> Config:
    return Config.from_toml(config_path)


@lru_cache(maxsize=1)
def get_config() -> Config:
    return load_config()


config = get_config()
