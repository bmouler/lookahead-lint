"""File discovery and the analysis pipeline that ties the modules together."""

from __future__ import annotations

import ast
import tokenize
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path

from .config import Config
from .notebook import NotebookError, NotebookSource, load_notebook
from .rules import Finding, run_checks
from .suppression import collect_suppressions

__all__ = [
    "SOURCE_SUFFIXES",
    "LintError",
    "Report",
    "analyze_file",
    "analyze_paths",
    "analyze_source",
    "collect_files",
]

SOURCE_SUFFIXES: frozenset[str] = frozenset({".py", ".ipynb"})
"""Suffixes picked up when a directory is walked."""


@dataclass(frozen=True)
class LintError:
    """A file that could not be analyzed, reported instead of being skipped."""

    path: Path
    line: int
    col: int
    message: str
    cell: int | None = None
    cell_line: int | None = None

    @property
    def location(self) -> str:
        """Human-readable ``line:col`` (or ``cell N, line M:col``) location."""
        if self.cell is None:
            return f"{self.line}:{self.col}"
        return f"cell {self.cell}, line {self.cell_line}:{self.col}"


@dataclass(frozen=True)
class Report:
    """Outcome of a run over one or more paths."""

    findings: tuple[Finding, ...] = ()
    errors: tuple[LintError, ...] = ()
    files_checked: int = 0

    @property
    def exit_code(self) -> int:
        """``2`` if any file failed to analyze, ``1`` if findings remain, else ``0``."""
        if self.errors:
            return 2
        return 1 if self.findings else 0


def analyze_source(
    source: str,
    path: Path,
    codes: Iterable[str] | None = None,
    notebook: NotebookSource | None = None,
) -> tuple[list[Finding], list[LintError]]:
    """Analyze in-memory source text.

    Args:
        source: Python source. For notebooks this is the concatenated code cells.
        path: Path reported on findings and errors; never read from disk.
        codes: Rule codes to run; ``None`` runs every rule.
        notebook: Line map used to translate positions back to cell coordinates.

    Returns:
        A ``(findings, errors)`` pair. ``errors`` holds at most one entry, since a
        file that does not parse produces no findings.
    """
    try:
        tree = ast.parse(source, filename=str(path))
    except (SyntaxError, ValueError) as error:
        # CPython reports a rejected source as SyntaxError on some versions and a
        # plain ValueError on others; both mean the same thing to a caller.
        if isinstance(error, SyntaxError):
            line = error.lineno or 1
            col = error.offset or 1
            detail = error.msg
        else:
            line = 1
            col = 1
            detail = str(error)
        message = f"{type(error).__name__}: {detail}"
        return [], [_locate_error(LintError(path, line, col, message), notebook)]

    suppressions = collect_suppressions(source)
    findings = [
        finding
        for finding in run_checks(tree, path, codes)
        if not suppressions.covers(finding.code, finding.line, finding.end_line)
    ]
    if notebook is None:
        return findings, []
    return [_locate_finding(finding, notebook) for finding in findings], []


def analyze_file(
    path: Path,
    codes: Iterable[str] | None = None,
) -> tuple[list[Finding], list[LintError]]:
    """Read and analyze a single file.

    ``.ipynb`` files are converted to source first; every other suffix is treated
    as Python.

    Args:
        path: File to analyze.
        codes: Rule codes to run; ``None`` runs every rule.

    Returns:
        A ``(findings, errors)`` pair. Unreadable or unparseable files yield an
        error rather than raising.
    """
    if path.suffix == ".ipynb":
        try:
            notebook = load_notebook(path)
        except (NotebookError, OSError) as error:
            return [], [LintError(path, 1, 1, str(error))]
        return analyze_source(notebook.source, path, codes, notebook)
    try:
        with tokenize.open(path) as handle:
            source = handle.read()
    except SyntaxError as error:
        return [], [LintError(path, 1, 1, f"cannot decode: {error.msg}")]
    except (OSError, UnicodeDecodeError) as error:
        return [], [LintError(path, 1, 1, f"cannot read: {error}")]
    return analyze_source(source, path, codes)


def collect_files(paths: Sequence[Path], config: Config) -> tuple[list[Path], list[LintError]]:
    """Expand command-line paths into the list of files to analyze.

    Directories are walked recursively for ``.py`` and ``.ipynb`` files, honouring
    the configured exclude patterns. Explicitly named files are always analyzed,
    so a targeted invocation is never silently ignored.

    Args:
        paths: Files and directories as given on the command line.
        config: Configuration supplying the exclude patterns.

    Returns:
        A ``(files, errors)`` pair; ``errors`` records paths that do not exist.
        Files are de-duplicated and each directory's contents are sorted.
    """
    files: list[Path] = []
    errors: list[LintError] = []
    seen: set[Path] = set()
    for path in paths:
        if path.is_dir():
            candidates = [
                candidate
                for candidate in sorted(path.rglob("*"))
                if candidate.suffix in SOURCE_SUFFIXES
                and candidate.is_file()
                and not config.is_excluded(candidate)
            ]
        elif path.is_file():
            candidates = [path]
        else:
            errors.append(LintError(path, 1, 1, "no such file or directory"))
            continue
        for candidate in candidates:
            resolved = candidate.resolve()
            if resolved not in seen:
                seen.add(resolved)
                files.append(candidate)
    return files, errors


def analyze_paths(paths: Sequence[Path], config: Config) -> Report:
    """Analyze every file reachable from ``paths``.

    Args:
        paths: Files and directories to analyze.
        config: Resolved configuration selecting rules and exclusions.

    Returns:
        The aggregated report, with findings and errors sorted by path and line.

    Raises:
        ConfigError: If the configuration disables every rule.
    """
    codes = config.enabled_codes()
    files, errors = collect_files(paths, config)
    findings: list[Finding] = []
    for file in files:
        file_findings, file_errors = analyze_file(file, codes)
        findings.extend(file_findings)
        errors.extend(file_errors)
    findings.sort(key=lambda item: (str(item.path), item.line, item.col, item.code))
    errors.sort(key=lambda item: (str(item.path), item.line, item.col))
    return Report(tuple(findings), tuple(errors), len(files))


def _locate_finding(finding: Finding, notebook: NotebookSource) -> Finding:
    cell, cell_line = notebook.locate(finding.line)
    return replace(finding, cell=cell, cell_line=cell_line)


def _locate_error(error: LintError, notebook: NotebookSource | None) -> LintError:
    if notebook is None or not notebook.line_map:
        return error
    cell, cell_line = notebook.locate(error.line)
    return replace(error, cell=cell, cell_line=cell_line)
