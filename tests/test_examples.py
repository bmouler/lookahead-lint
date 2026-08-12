"""The shipped examples are the contract: they cannot drift away from the rules."""

from __future__ import annotations

import io
import re
import tokenize
from pathlib import Path

from lookahead_lint import ALL_CODES, analyze_file

_SEEDED_CODE = re.compile(r"#\s*(?P<code>LA\d{3})\s*$")


def _seeded_expectations(path: Path) -> set[tuple[int, str]]:
    """Read the ``# LAxxx`` markers the example annotates each leaking line with."""
    source = path.read_text(encoding="utf-8")
    expectations: set[tuple[int, str]] = set()
    for token in tokenize.generate_tokens(io.StringIO(source).readline):
        if token.type != tokenize.COMMENT:
            continue
        match = _SEEDED_CODE.match(token.string)
        if match is not None:
            expectations.add((token.start[0], match.group("code")))
    return expectations


def test_leaky_example_seeds_every_rule(examples: Path) -> None:
    findings, errors = analyze_file(examples / "leaky_research.py")
    assert errors == []
    assert {finding.code for finding in findings} == ALL_CODES


def test_leaky_example_findings_match_its_own_annotations(examples: Path) -> None:
    path = examples / "leaky_research.py"
    findings, errors = analyze_file(path)
    assert errors == []
    reported = {(finding.line, finding.code) for finding in findings}
    expected = _seeded_expectations(path)
    assert expected, "the example must annotate its seeded lines"
    assert reported == expected


def test_clean_example_is_clean(examples: Path) -> None:
    findings, errors = analyze_file(examples / "clean_research.py")
    assert (findings, errors) == ([], [])


def test_clean_example_documents_its_one_suppression(examples: Path) -> None:
    source = (examples / "clean_research.py").read_text(encoding="utf-8")
    assert source.count("lookahead-lint: ignore") == 1
    findings, _ = analyze_file(examples / "clean_research.py", ["LA001"])
    assert findings == []


def test_leaky_notebook_reports_cell_coordinates(examples: Path) -> None:
    findings, errors = analyze_file(examples / "leaky_notebook.ipynb")
    assert errors == []
    assert [(finding.cell, finding.cell_line, finding.code) for finding in findings] == [
        (2, 2, "LA001"),
        (2, 3, "LA003"),
        (3, 4, "LA005"),
    ]
