"""Notebook extraction, magic handling, and cell/line mapping."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from lookahead_lint import NotebookError, analyze_file, notebook_source


def _notebook(*cells: tuple[str, str]) -> str:
    return json.dumps(
        {
            "cells": [
                {"cell_type": kind, "metadata": {}, "outputs": [], "source": source}
                for kind, source in cells
            ],
            "metadata": {},
            "nbformat": 4,
            "nbformat_minor": 5,
        }
    )


def test_example_notebook_maps_findings_to_cell_coordinates(examples: Path) -> None:
    findings, errors = analyze_file(examples / "leaky_notebook.ipynb")
    assert errors == []
    located = [(finding.cell, finding.cell_line, finding.code) for finding in findings]
    assert located == [(2, 2, "LA001"), (2, 3, "LA003"), (3, 4, "LA005")]
    assert findings[0].location == "cell 2, line 2:20"


def test_markdown_cells_are_skipped_without_consuming_a_number() -> None:
    text = _notebook(
        ("markdown", "# heading"),
        ("code", "import pandas as pd"),
        ("markdown", "prose"),
        ("code", "target = close.shift(-1)"),
    )
    extracted = notebook_source(text)
    assert extracted.locate(1) == (1, 1)
    assert extracted.locate(3) == (2, 1)


def test_magics_are_stripped_only_from_cells_that_do_not_parse() -> None:
    text = _notebook(
        ("code", "%matplotlib inline\n!pip install pandas\nfiles = !ls\ntarget = close.shift(-1)"),
    )
    extracted = notebook_source(text)
    lines = extracted.source.splitlines()
    assert lines[:3] == ["", "", ""]
    assert lines[3] == "target = close.shift(-1)"
    assert extracted.locate(4) == (1, 4)


def test_valid_python_is_never_rewritten() -> None:
    text = _notebook(("code", "remainder = (numerator\n% divisor)\ntarget = close.shift(-1)"))
    extracted = notebook_source(text)
    assert "% divisor" in extracted.source
    assert extracted.locate(3) == (1, 3)


def test_cell_source_may_be_a_list_of_lines() -> None:
    text = _notebook(("code", "unused"))
    document = json.loads(text)
    document["cells"][0]["source"] = ["a = 1\n", "b = 2\n"]
    extracted = notebook_source(json.dumps(document))
    assert extracted.source == "a = 1\nb = 2\n"


def test_notebook_without_code_cells_yields_nothing(tmp_path: Path) -> None:
    path = tmp_path / "prose.ipynb"
    path.write_text(_notebook(("markdown", "just prose")), encoding="utf-8")
    findings, errors = analyze_file(path)
    assert (findings, errors) == ([], [])


def test_locate_rejects_a_notebook_without_code_cells() -> None:
    extracted = notebook_source(_notebook(("markdown", "prose")))
    with pytest.raises(ValueError, match="no code cells"):
        extracted.locate(1)


def test_locate_rejects_a_non_positive_line() -> None:
    extracted = notebook_source(_notebook(("code", "a = 1")))
    with pytest.raises(ValueError, match="1-based"):
        extracted.locate(0)


def test_invalid_json_is_reported_as_an_error(tmp_path: Path) -> None:
    path = tmp_path / "broken.ipynb"
    path.write_text('{"cells": [', encoding="utf-8")
    findings, errors = analyze_file(path)
    assert findings == []
    assert len(errors) == 1
    assert "invalid notebook JSON" in errors[0].message


def test_json_that_is_not_a_notebook_is_reported_as_an_error(tmp_path: Path) -> None:
    path = tmp_path / "plain.ipynb"
    path.write_text('{"nbformat": 4}', encoding="utf-8")
    findings, errors = analyze_file(path)
    assert findings == []
    assert "not a notebook document" in errors[0].message


def test_unreadable_cell_source_is_rejected() -> None:
    document = {"cells": [{"cell_type": "code", "source": 42}]}
    with pytest.raises(NotebookError, match="unreadable 'source'"):
        notebook_source(json.dumps(document))


def test_syntax_error_in_a_cell_maps_to_cell_coordinates(tmp_path: Path) -> None:
    path = tmp_path / "bad.ipynb"
    path.write_text(_notebook(("code", "a = 1"), ("code", "def broken(:\n    pass")), "utf-8")
    findings, errors = analyze_file(path)
    assert findings == []
    assert len(errors) == 1
    assert (errors[0].cell, errors[0].cell_line) == (2, 1)
