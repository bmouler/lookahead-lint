"""Static detection of look-ahead bias and target leakage in research code.

The public entry points are :func:`analyze_paths` for a run over files and
directories, :func:`analyze_source` for in-memory source, and :data:`RULES` for
the rule catalogue. The package depends only on the standard library.
"""

from __future__ import annotations

from .analyzer import (
    LintError,
    Report,
    analyze_file,
    analyze_paths,
    analyze_source,
    collect_files,
)
from .config import Config, ConfigError, discover_config, load_config
from .notebook import NotebookError, NotebookSource, load_notebook, notebook_source
from .reporter import FORMATS, SCHEMA_VERSION, render
from .rules import ALL_CODES, RULES, Finding, Rule, run_checks
from .suppression import Suppressions, collect_suppressions

__version__ = "0.1.0"

__all__ = [
    "ALL_CODES",
    "FORMATS",
    "RULES",
    "SCHEMA_VERSION",
    "Config",
    "ConfigError",
    "Finding",
    "LintError",
    "NotebookError",
    "NotebookSource",
    "Report",
    "Rule",
    "Suppressions",
    "__version__",
    "analyze_file",
    "analyze_paths",
    "analyze_source",
    "collect_files",
    "collect_suppressions",
    "discover_config",
    "load_config",
    "load_notebook",
    "notebook_source",
    "render",
    "run_checks",
]
