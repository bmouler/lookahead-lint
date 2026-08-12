"""Optional ``[tool.lookahead_lint]`` configuration read from ``pyproject.toml``."""

from __future__ import annotations

import tomllib
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from fnmatch import fnmatch
from pathlib import Path
from typing import cast

from .rules import ALL_CODES

__all__ = [
    "ALWAYS_EXCLUDED",
    "TABLE",
    "Config",
    "ConfigError",
    "discover_config",
    "load_config",
]

TABLE = "lookahead_lint"
"""Name of the table under ``[tool]`` that holds the configuration."""

ALWAYS_EXCLUDED: frozenset[str] = frozenset(
    {
        ".git",
        ".hg",
        ".venv",
        "venv",
        "__pycache__",
        ".ipynb_checkpoints",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        "build",
        "dist",
        "node_modules",
    }
)
"""Directory names skipped during recursion regardless of configuration."""

_KEYS = frozenset({"select", "ignore", "exclude"})


class ConfigError(ValueError):
    """Raised for a malformed ``[tool.lookahead_lint]`` table or bad CLI codes."""


@dataclass(frozen=True)
class Config:
    """Resolved configuration for a run.

    Attributes:
        select: Codes to enable. ``None`` means every rule.
        ignore: Codes to disable, applied after ``select``.
        exclude: Glob patterns skipped during directory recursion. Patterns are
            matched against the file name, each parent directory name, and the
            full POSIX path, so both ``"*_generated.py"`` and ``"notebooks"``
            behave as expected.
        source: The ``pyproject.toml`` the settings came from, if any.
    """

    select: frozenset[str] | None = None
    ignore: frozenset[str] = frozenset()
    exclude: tuple[str, ...] = ()
    source: Path | None = None

    def enabled_codes(self) -> frozenset[str]:
        """Return the rule codes that should run.

        Raises:
            ConfigError: If the result is empty, which would make the run pointless.
        """
        enabled = (ALL_CODES if self.select is None else self.select) - self.ignore
        if not enabled:
            raise ConfigError("no rules left to run: select and ignore cancel each other out")
        return enabled

    def is_excluded(self, path: Path) -> bool:
        """Report whether ``path`` should be skipped during directory recursion."""
        parts = path.parts
        if ALWAYS_EXCLUDED.intersection(parts[:-1]):
            return True
        if not self.exclude:
            return False
        posix = path.as_posix()
        return any(
            fnmatch(posix, pattern) or any(fnmatch(part, pattern) for part in parts)
            for pattern in self.exclude
        )

    def with_overrides(
        self,
        select: Iterable[str] | None = None,
        ignore: Iterable[str] | None = None,
    ) -> Config:
        """Return a copy with command-line overrides applied.

        Args:
            select: Replacement for :attr:`select`; ``None`` keeps the configured value.
            ignore: Replacement for :attr:`ignore`; ``None`` keeps the configured value.

        Returns:
            The updated configuration.

        Raises:
            ConfigError: If any code is not a known rule code.
        """
        new_select = self.select if select is None else _validate_codes(select, "--select")
        new_ignore = self.ignore if ignore is None else _validate_codes(ignore, "--ignore")
        return replace(self, select=new_select, ignore=new_ignore)


def discover_config(paths: Sequence[Path]) -> Config:
    """Find and load the nearest configuration for the given paths.

    The search starts at the first path (its parent, for a file) and walks up to
    the filesystem root, returning the first ``pyproject.toml`` that actually
    contains a ``[tool.lookahead_lint]`` table.

    Args:
        paths: Paths given on the command line. An empty sequence searches from
            the current directory.

    Returns:
        The loaded configuration, or the defaults when no table is found.

    Raises:
        ConfigError: If a table is found but malformed.
    """
    start = Path.cwd() if not paths else paths[0].resolve()
    if start.is_file():
        start = start.parent
    for directory in (start, *start.parents):
        candidate = directory / "pyproject.toml"
        if candidate.is_file():
            config = load_config(candidate)
            if config.source is not None:
                return config
    return Config()


def load_config(pyproject: Path) -> Config:
    """Load ``[tool.lookahead_lint]`` from a specific ``pyproject.toml``.

    Args:
        pyproject: Path to the TOML file.

    Returns:
        The configuration. :attr:`Config.source` is ``None`` when the file has no
        ``[tool.lookahead_lint]`` table.

    Raises:
        ConfigError: If the file is not readable TOML, contains unknown keys, or
            holds values of the wrong type.
    """
    try:
        raw_data = cast(dict[str, object], tomllib.loads(pyproject.read_text(encoding="utf-8")))
        data: Mapping[str, object] = raw_data
    except (OSError, UnicodeDecodeError) as error:
        raise ConfigError(f"{pyproject}: cannot read: {error}") from error
    except tomllib.TOMLDecodeError as error:
        raise ConfigError(f"{pyproject}: invalid TOML: {error}") from error

    tool_value = data.get("tool")
    tool = cast(Mapping[str, object], tool_value) if isinstance(tool_value, dict) else None
    table_value = tool.get(TABLE) if tool is not None else None
    if table_value is None:
        return Config()
    if not isinstance(table_value, dict):
        raise ConfigError(f"{pyproject}: [tool.{TABLE}] must be a table")
    table = cast(Mapping[str, object], table_value)
    unknown = sorted(set(table) - _KEYS)
    if unknown:
        allowed = ", ".join(sorted(_KEYS))
        raise ConfigError(f"{pyproject}: unknown key(s) {', '.join(unknown)}; allowed: {allowed}")

    origin = f"[tool.{TABLE}]"
    select_raw = _string_list(table, "select", pyproject)
    ignore_raw = _string_list(table, "ignore", pyproject)
    exclude_raw = _string_list(table, "exclude", pyproject)
    select = None if select_raw is None else _validate_codes(select_raw, f"{origin}.select")
    ignore: frozenset[str] = frozenset()
    if ignore_raw is not None:
        ignore = _validate_codes(ignore_raw, f"{origin}.ignore")
    return Config(
        select=select,
        ignore=ignore,
        exclude=() if exclude_raw is None else tuple(exclude_raw),
        source=pyproject,
    )


def _string_list(
    table: Mapping[str, object],
    key: str,
    pyproject: Path,
) -> list[str] | None:
    value = table.get(key)
    if value is None:
        return None
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ConfigError(f"{pyproject}: [tool.{TABLE}].{key} must be a list of strings")
    return cast(list[str], value)


def _validate_codes(codes: Iterable[str], origin: str) -> frozenset[str]:
    """Normalize codes to upper case and reject anything unknown."""
    normalized = frozenset(code.strip().upper() for code in codes if code.strip())
    unknown = sorted(normalized - ALL_CODES)
    if unknown:
        known = ", ".join(sorted(ALL_CODES))
        raise ConfigError(f"{origin}: unknown rule code(s) {', '.join(unknown)}; known: {known}")
    return normalized
