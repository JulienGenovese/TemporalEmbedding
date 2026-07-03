from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - py311+ usa tomllib stdlib
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

    def get(
        self,
        section: str,
        key: str,
        default: Any = _MISSING,
        *,
        value_type: type | tuple[type, ...] | None = None,
    ) -> Any:
        section_data = self._resolve_section(section)
        if key in section_data:
            value = section_data[key]
        elif default is _MISSING:
            raise KeyError(f"Key `{key}` not found in section `{section}`.")
        else:
            value = default

        if value_type is None:
            return value
        return self._validate_type(section, key, value, value_type)

    @staticmethod
    def _validate_type(
        section: str,
        key: str,
        value: Any,
        value_type: type | tuple[type, ...],
    ) -> Any:
        if value_type is int:
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(
                    f"`{section}.{key}` must be an integer, got {type(value).__name__}.",
                )
            return value

        if value_type is float:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(
                    f"`{section}.{key}` must be numeric, got {type(value).__name__}.",
                )
            return float(value)

        if value_type is Path:
            if isinstance(value, Path):
                return value
            if not isinstance(value, str):
                raise TypeError(
                    f"`{section}.{key}` must be a string path, got {type(value).__name__}.",
                )
            return Path(value)

        if not isinstance(value, value_type):
            expected = (
                " or ".join(t.__name__ for t in value_type)
                if isinstance(value_type, tuple)
                else value_type.__name__
            )
            raise TypeError(
                f"`{section}.{key}` must be {expected}, got {type(value).__name__}.",
            )
        return value


def load_config(config_path: Path | str | None = None) -> Config:
    return Config.from_toml(config_path)


@lru_cache(maxsize=1)
def get_config() -> Config:
    return load_config()


config = get_config()
