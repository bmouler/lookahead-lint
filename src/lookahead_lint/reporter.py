"""Output formats: a grouped text table, stable JSON, and CI annotations."""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass

from .analyzer import LintError, Report
from .rules import RULES, Finding

__all__ = ["FORMATS", "SCHEMA_VERSION", "render", "render_github", "render_json", "render_text"]

FORMATS: tuple[str, ...] = ("text", "json", "github")
"""Supported values of the ``--format`` option, in help order."""

SCHEMA_VERSION = 1
"""Version of the JSON payload. Incremented only on a breaking field change."""


def render(report: Report, output_format: str = "text") -> str:
    """Render a report in the requested format.

    Args:
        report: Analysis result to render.
        output_format: One of :data:`FORMATS`.

    Returns:
        The rendered text, newline-terminated unless empty.

    Raises:
        ValueError: If ``output_format`` is not a supported format.
    """
    if output_format == "text":
        return render_text(report)
    if output_format == "json":
        return render_json(report)
    if output_format == "github":
        return render_github(report)
    raise ValueError(f"unknown format {output_format!r}; choose from {', '.join(FORMATS)}")


def render_text(report: Report) -> str:
    """Render findings grouped by file, followed by the rules they triggered."""
    grouped: dict[str, list[_Row]] = defaultdict(list)
    for finding in report.findings:
        grouped[str(finding.path)].append(
            _Row(finding.line, finding.col, finding.location, finding.code, finding.message)
        )
    for error in report.errors:
        grouped[str(error.path)].append(
            _Row(error.line, error.col, error.location, "error", error.message)
        )

    blocks: list[str] = []
    for path in sorted(grouped):
        rows = sorted(grouped[path], key=lambda row: (row.line, row.col, row.code))
        width = max(len(row.location) for row in rows)
        lines = [path]
        lines += [f"  {row.location:<{width}}  {row.code:<5}  {row.message}" for row in rows]
        blocks.append("\n".join(lines))

    codes = sorted({finding.code for finding in report.findings})
    if codes:
        legend = ["rules"]
        for code in codes:
            rule = RULES[code]
            legend.append(f"  {code} {rule.name}")
            legend.append(f"    why: {rule.rationale}")
            legend.append(f"    fix: {rule.fix}")
        blocks.append("\n".join(legend))

    blocks.append(_summary(report))
    return "\n\n".join(blocks) + "\n"


def render_json(report: Report) -> str:
    """Render a stable machine-readable payload.

    The schema is ``{tool, schema_version, summary, findings, errors}``. Fields
    are only ever added within a schema version, never removed or retyped.
    """
    payload = {
        "tool": "lookahead-lint",
        "schema_version": SCHEMA_VERSION,
        "summary": {
            "files_checked": report.files_checked,
            "findings": len(report.findings),
            "errors": len(report.errors),
        },
        "findings": [_finding_payload(finding) for finding in report.findings],
        "errors": [_error_payload(error) for error in report.errors],
    }
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def render_github(report: Report) -> str:
    """Render GitHub Actions workflow commands, one annotation per line.

    Findings become ``::warning`` annotations and unanalyzable files become
    ``::error`` annotations, so a pull request shows them inline on the diff.
    """
    lines: list[str] = []
    for finding in report.findings:
        rule = RULES[finding.code]
        message = f"{finding.message}. Fix: {rule.fix}"
        if finding.cell is not None:
            message = f"[cell {finding.cell}, line {finding.cell_line}] {message}"
        properties = _properties(str(finding.path), finding.line, finding.col)
        title = _escape_property(f"{finding.code} {rule.name}")
        lines.append(f"::warning {properties},title={title}::{_escape_data(message)}")
    for error in report.errors:
        properties = _properties(str(error.path), error.line, error.col)
        message = error.message
        if error.cell is not None:
            message = f"[cell {error.cell}, line {error.cell_line}] {message}"
        lines.append(
            f"::error {properties},title=lookahead-lint::{_escape_data(message)}",
        )
    if not lines:
        return ""
    return "\n".join(lines) + "\n"


def _finding_payload(finding: Finding) -> dict[str, object]:
    rule = RULES[finding.code]
    return {
        "path": str(finding.path),
        "line": finding.line,
        "column": finding.col,
        "end_line": finding.end_line,
        "cell": finding.cell,
        "cell_line": finding.cell_line,
        "code": finding.code,
        "name": rule.name,
        "message": finding.message,
        "rationale": rule.rationale,
        "fix": rule.fix,
    }


def _error_payload(error: LintError) -> dict[str, object]:
    return {
        "path": str(error.path),
        "line": error.line,
        "column": error.col,
        "cell": error.cell,
        "cell_line": error.cell_line,
        "message": error.message,
    }


@dataclass(frozen=True)
class _Row:
    """One rendered line of the text report."""

    line: int
    col: int
    location: str
    code: str
    message: str


def _summary(report: Report) -> str:
    if not report.findings and not report.errors:
        return f"no findings ({_count(report.files_checked, 'file')} checked)"
    files = len({str(finding.path) for finding in report.findings})
    parts = [f"{_count(len(report.findings), 'finding')} in {_count(files, 'file')}"]
    if report.errors:
        parts.append(_count(len(report.errors), "error"))
    return f"{', '.join(parts)} ({_count(report.files_checked, 'file')} checked)"


def _count(value: int, noun: str) -> str:
    return f"{value} {noun}" if value == 1 else f"{value} {noun}s"


def _escape_data(value: str) -> str:
    return value.replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")


def _escape_property(value: str) -> str:
    return _escape_data(value).replace(":", "%3A").replace(",", "%2C")


def _properties(path: str, line: int, col: int) -> str:
    return f"file={_escape_property(path)},line={line},col={col}"
