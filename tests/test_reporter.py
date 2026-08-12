"""The three output formats."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from lookahead_lint import RULES, Finding, LintError, Report, render
from lookahead_lint.reporter import SCHEMA_VERSION, render_github, render_json, render_text

_DEFAULT_PATH = Path("research/features.py")


def _finding(
    path: Path = _DEFAULT_PATH,
    line: int = 12,
    col: int = 5,
    code: str = "LA001",
    message: str = "negative shift pulls future rows into the present",
    cell: int | None = None,
    cell_line: int | None = None,
) -> Finding:
    return Finding(
        path=path,
        line=line,
        col=col,
        end_line=line,
        code=code,
        message=message,
        cell=cell,
        cell_line=cell_line,
    )


def test_text_groups_by_file_and_aligns_locations() -> None:
    report = Report(
        findings=(
            _finding(line=12, col=5),
            _finding(line=140, col=11, code="LA003", message="centered window"),
        ),
        errors=(LintError(Path("research/broken.py"), 3, 9, "SyntaxError: invalid syntax"),),
        files_checked=2,
    )
    lines = render_text(report).splitlines()
    assert lines[0] == "research/broken.py"
    assert lines[1] == "  3:9  error  SyntaxError: invalid syntax"
    assert lines[3] == "research/features.py"
    assert lines[4].startswith("  12:5    LA001  ")
    assert lines[5].startswith("  140:11  LA003  ")
    assert lines[-1] == "2 findings in 1 file, 1 error (2 files checked)"


def test_text_explains_every_triggered_rule_once() -> None:
    report = Report(findings=(_finding(), _finding(line=13)), files_checked=1)
    text = render_text(report)
    assert text.count("  LA001 negative-shift") == 1
    assert f"    why: {RULES['LA001'].rationale}" in text
    assert f"    fix: {RULES['LA001'].fix}" in text


def test_text_reports_a_clean_run() -> None:
    assert render_text(Report(files_checked=3)) == "no findings (3 files checked)\n"


def test_text_uses_cell_coordinates_for_notebooks() -> None:
    report = Report(
        findings=(_finding(path=Path("explore.ipynb"), line=9, cell=2, cell_line=3),),
        files_checked=1,
    )
    assert "  cell 2, line 3:5  LA001  " in render_text(report)


def test_json_payload_is_stable_and_complete() -> None:
    report = Report(
        findings=(_finding(),),
        errors=(LintError(Path("broken.py"), 1, 1, "SyntaxError: invalid syntax"),),
        files_checked=4,
    )
    payload = json.loads(render_json(report))
    assert payload["tool"] == "lookahead-lint"
    assert payload["schema_version"] == SCHEMA_VERSION
    assert payload["summary"] == {"files_checked": 4, "findings": 1, "errors": 1}
    assert payload["findings"] == [
        {
            "cell": None,
            "cell_line": None,
            "code": "LA001",
            "column": 5,
            "end_line": 12,
            "fix": RULES["LA001"].fix,
            "line": 12,
            "message": "negative shift pulls future rows into the present",
            "name": "negative-shift",
            "path": "research/features.py",
            "rationale": RULES["LA001"].rationale,
        }
    ]
    assert payload["errors"] == [
        {
            "cell": None,
            "cell_line": None,
            "column": 1,
            "line": 1,
            "message": "SyntaxError: invalid syntax",
            "path": "broken.py",
        }
    ]


def test_json_of_a_clean_run_still_has_every_key() -> None:
    payload = json.loads(render_json(Report(files_checked=1)))
    assert payload["findings"] == []
    assert payload["errors"] == []
    assert payload["summary"]["findings"] == 0


def test_github_emits_one_annotation_per_finding() -> None:
    report = Report(findings=(_finding(),), files_checked=1)
    line = render_github(report).splitlines()[0]
    assert line.startswith(
        "::warning file=research/features.py,line=12,col=5,title=LA001 negative-shift::"
    )
    assert line.endswith(f"Fix: {RULES['LA001'].fix}")


def test_github_escapes_separators_and_percent_signs() -> None:
    report = Report(
        findings=(_finding(path=Path("a,b:c.py"), message="drops 50% of rows\nplus a newline"),),
        errors=(LintError(Path("broken.py"), 2, 4, "SyntaxError: bad"),),
        files_checked=1,
    )
    warning, error = render_github(report).splitlines()
    assert "file=a%2Cb%3Ac.py" in warning
    assert "drops 50%25 of rows%0Aplus a newline" in warning
    assert error.startswith("::error file=broken.py,line=2,col=4,title=lookahead-lint::")


def test_github_annotates_notebook_cells_in_the_message() -> None:
    report = Report(
        findings=(_finding(path=Path("explore.ipynb"), line=9, cell=2, cell_line=3),),
        files_checked=1,
    )
    assert "::[cell 2, line 3] negative shift" in render_github(report)


def test_github_output_is_empty_for_a_clean_run() -> None:
    assert render_github(Report(files_checked=2)) == ""


@pytest.mark.parametrize("output_format", ["text", "json", "github"])
def test_render_dispatches_to_every_format(output_format: str) -> None:
    report = Report(findings=(_finding(),), files_checked=1)
    assert render(report, output_format) == {
        "text": render_text,
        "json": render_json,
        "github": render_github,
    }[output_format](report)


def test_render_rejects_an_unknown_format() -> None:
    with pytest.raises(ValueError, match="unknown format 'sarif'"):
        render(Report(), "sarif")


def test_exit_code_reflects_the_worst_outcome() -> None:
    assert Report(files_checked=1).exit_code == 0
    assert Report(findings=(_finding(),), files_checked=1).exit_code == 1
    assert Report(errors=(LintError(Path("x.py"), 1, 1, "boom"),)).exit_code == 2
