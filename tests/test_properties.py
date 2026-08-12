"""Deterministic property tests for the public analysis and reporting contracts."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st

from lookahead_lint import FORMATS, Report, analyze_source, render

PATH = Path("t.py")

CLEAN_STATEMENTS = (
    "constant = 1",
    "lagged = prices.shift(1)",
    "filled = prices.ffill()",
    "windowed = prices.rolling(5, center=False).mean()",
    'joined = merge_asof(left, right, direction="backward")',
    "history = [series[i - 1] for i in range(1, 4)]",
)

LEAKY_SNIPPETS = (
    ("LA001", "target = prices.shift(-1)"),
    ("LA002", "filled = prices.bfill()"),
    ("LA003", "smooth = prices.rolling(5, center=True).mean()"),
    (
        "LA004",
        "scaler.fit(data)\ntrain, test = train_test_split(data, shuffle=False)",
    ),
    ("LA005", "train, test = train_test_split(data)"),
    ("LA006", "future = [series[i + 1] for i in range(3)]"),
    ("LA007", 'joined = merge_asof(left, right, direction="forward")'),
)

clean_programs = st.lists(st.sampled_from(CLEAN_STATEMENTS), max_size=12)
all_programs = st.lists(
    st.sampled_from(CLEAN_STATEMENTS + tuple(snippet for _, snippet in LEAKY_SNIPPETS)),
    max_size=12,
)


@given(statements=clean_programs)
def test_clean_statement_compositions_have_no_findings(statements: list[str]) -> None:
    findings, errors = analyze_source("\n".join(statements), PATH)

    assert errors == []
    assert findings == []


@pytest.mark.parametrize(("code", "snippet"), LEAKY_SNIPPETS)
@given(statements=clean_programs, boundary=st.data())
def test_inserted_leak_is_reported_at_any_statement_boundary(
    code: str,
    snippet: str,
    statements: list[str],
    boundary: st.DataObject,
) -> None:
    position = boundary.draw(st.integers(min_value=0, max_value=len(statements)))
    program = [*statements[:position], snippet, *statements[position:]]
    findings, errors = analyze_source("\n".join(program), PATH)

    assert errors == []
    assert code in {finding.code for finding in findings}


@given(statements=all_programs)
def test_analysis_is_pure_and_every_public_format_renders(statements: list[str]) -> None:
    source = "\n".join(statements)
    first = analyze_source(source, PATH)
    second = analyze_source(source, PATH)

    assert first == second
    findings, errors = first
    report = Report(findings=tuple(findings), errors=tuple(errors), files_checked=1)
    rendered = {output_format: render(report, output_format) for output_format in FORMATS}

    assert set(rendered) == set(FORMATS)
    payload = json.loads(rendered["json"])
    assert json.loads(json.dumps(payload)) == payload
