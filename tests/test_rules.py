"""Per-rule behaviour: each check must fire on the leak and stay quiet on the fix."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from lookahead_lint import ALL_CODES, RULES, run_checks


def test_rule_catalogue_is_complete_and_stable() -> None:
    assert set(RULES) == ALL_CODES == {f"LA{index:03d}" for index in range(1, 8)}
    for code, rule in RULES.items():
        assert rule.code == code
        assert rule.name and rule.message and rule.rationale and rule.fix
        assert rule.name.islower()


def test_run_checks_rejects_unknown_codes() -> None:
    tree = ast.parse("x = 1\n")
    with pytest.raises(KeyError, match="LA999"):
        run_checks(tree, Path("snippet.py"), ["LA999"])


def test_analyze_source_reports_value_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    from lookahead_lint import analyze_source, analyzer

    def reject_source(*args: object, **kwargs: object) -> None:
        raise ValueError("embedded null byte")

    monkeypatch.setattr(analyzer.ast, "parse", reject_source)
    findings, errors = analyze_source("x = 1", Path("snippet.py"))

    assert findings == []
    assert len(errors) == 1
    assert errors[0].line == errors[0].col == 1
    assert errors[0].message == "ValueError: embedded null byte"


def test_indirect_call_expression_is_not_treated_as_a_named_call(codes_for) -> None:
    assert codes_for("factory()()") == []


@pytest.mark.parametrize(
    "source",
    [
        'frame["y"] = close.shift(-1)',
        "close.shift(periods=-3)",
        "close.shift(-10).rolling(5).mean()",
    ],
)
def test_la001_fires_on_negative_shift(codes_for, source: str) -> None:
    assert codes_for(source) == ["LA001"]


@pytest.mark.parametrize(
    "source",
    [
        "close.shift(1)",
        "close.shift(periods=2)",
        "close.shift(horizon)",
        "close.shift()",
        "close.shift(-horizon)",
    ],
)
def test_la001_stays_quiet_on_backward_shift(codes_for, source: str) -> None:
    assert codes_for(source) == []


def test_la001_stays_quiet_on_negative_boolean_literal(codes_for) -> None:
    assert codes_for("close.shift(-True)") == []


@pytest.mark.parametrize(
    "source",
    [
        "series.bfill()",
        'series.fillna(method="bfill")',
        'series.fillna(method="backfill")',
        'series.reindex(index, method="bfill")',
        'series.asfreq("D", method="backfill")',
        'left.align(right, method="bfill")',
        'series.interpolate(limit_direction="backward")',
        'series.interpolate("linear", limit_direction="both")',
    ],
)
def test_la002_fires_on_backward_fill(codes_for, source: str) -> None:
    assert codes_for(source) == ["LA002"]


@pytest.mark.parametrize(
    "source",
    [
        "series.ffill()",
        "series.fillna(0.0)",
        'series.fillna(method="ffill")',
        'series.reindex(index, method="ffill")',
        'series.asfreq("D")',
        "left.align(right)",
        "series.interpolate()",
        'series.interpolate(limit_direction="forward")',
        "series.fillna(method=chosen)",
    ],
)
def test_la002_stays_quiet_on_forward_fill(codes_for, source: str) -> None:
    assert codes_for(source) == []


def test_la003_fires_on_centered_window(codes_for) -> None:
    assert codes_for("close.rolling(20, center=True).mean()") == ["LA003"]


@pytest.mark.parametrize(
    "source",
    [
        "close.rolling(20).mean()",
        "close.rolling(20, center=False).mean()",
        "close.rolling(window=20, center=centered).mean()",
    ],
)
def test_la003_stays_quiet_on_trailing_window(codes_for, source: str) -> None:
    assert codes_for(source) == []


def test_la004_fires_when_fit_precedes_split_in_a_function(lint) -> None:
    findings = lint(
        """
        def run(frame, labels):
            scaled = scaler.fit_transform(frame)
            return train_test_split(scaled, labels, shuffle=False)
        """
    )
    assert [finding.code for finding in findings] == ["LA004"]
    assert findings[0].line == 3


def test_la004_fires_at_module_scope(codes_for) -> None:
    assert codes_for(
        """
        scaler.fit(matrix)
        train, test = train_test_split(matrix, shuffle=False)
        """
    ) == ["LA004"]


def test_la004_checks_async_function_and_class_scopes(codes_for) -> None:
    assert codes_for(
        """
            class Pipeline:
                scaler.fit(data)
                train_test_split(data, shuffle=False)

            async def prepare(data):
                scaler.fit_transform(data)
                train_test_split(data, shuffle=False)
            """
    ) == ["LA004", "LA004"]


@pytest.mark.parametrize(
    "source",
    [
        """
        def run(frame, labels):
            a, b = train_test_split(frame, labels, shuffle=False)
            scaler.fit(a)
            return scaler.transform(b)
        """,
        """
        def run(frame):
            return scaler.fit_transform(frame)
        """,
        """
        def outer(frame, labels):
            def helper(part):
                return scaler.fit_transform(part)

            a, b = train_test_split(frame, labels, shuffle=False)
            return helper(a), b
        """,
    ],
)
def test_la004_stays_quiet_when_ordering_is_correct(codes_for, source: str) -> None:
    assert codes_for(source) == []


@pytest.mark.parametrize(
    "source",
    [
        "train_test_split(features, labels)",
        "train_test_split(features, labels, shuffle=True)",
        "model_selection.train_test_split(features, labels, test_size=0.2)",
    ],
)
def test_la005_fires_on_shuffled_split(codes_for, source: str) -> None:
    assert codes_for(source) == ["LA005"]


@pytest.mark.parametrize(
    "source",
    [
        "train_test_split(features, labels, shuffle=False)",
        "train_test_split(features, labels, **options)",
        "split(features, labels)",
    ],
)
def test_la005_stays_quiet_on_ordered_split(codes_for, source: str) -> None:
    assert codes_for(source) == []


@pytest.mark.parametrize(
    "source",
    [
        """
        for i in range(n):
            total += close[i + 1]
        """,
        """
        for i in range(n):
            total += close[1 + i]
        """,
        """
        for i in range(n):
            total += frame.iloc[i + 1, 0]
        """,
        """
        for step in range(n):
            for i in range(m):
                total += close[i + step]
        """,
        "values = [close[i + 1] for i in range(n)]",
    ],
)
def test_la006_fires_on_future_row_index(codes_for, source: str) -> None:
    assert codes_for(source) == ["LA006"]


@pytest.mark.parametrize(
    "source",
    [
        """
        async def consume(rows):
            async for i in rows:
                value = close[i + 1]
        """,
        "values = {close[i + 1] for i in range(n)}",
        "values = {i: close[i + 1] for i in range(n)}",
        "values = (close[i + 1] for i in range(n))",
    ],
)
def test_la006_covers_every_loop_form(codes_for, source: str) -> None:
    assert codes_for(source) == ["LA006"]


@pytest.mark.parametrize(
    "source",
    [
        """
        for i in range(n):
            total += close[i]
        """,
        """
        for i in range(1, n):
            total += close[i - 1]
        """,
        """
        for i in range(n):
            total += close[i + -1]
        """,
        """
        for i in range(n):
            total += close[j + 1]
        """,
        """
        for i in range(n):
            total += close[i : i + 1].sum()
        """,
        "total = close[i + 1]",
    ],
)
def test_la006_stays_quiet_outside_a_forward_read(codes_for, source: str) -> None:
    assert codes_for(source) == []


def test_la006_loop_variable_scope_ends_with_the_loop(codes_for) -> None:
    assert (
        codes_for(
            """
            for i in range(n):
                total += close[i]
            tail = close[i + 1]
            """
        )
        == []
    )


@pytest.mark.parametrize(
    "source",
    [
        'pd.merge_asof(left, right, on="ts", direction="forward")',
        'merge_asof(left, right, on="ts", direction="nearest")',
    ],
)
def test_la007_fires_on_forward_asof(codes_for, source: str) -> None:
    assert codes_for(source) == ["LA007"]


@pytest.mark.parametrize(
    "source",
    [
        'pd.merge_asof(left, right, on="ts")',
        'pd.merge_asof(left, right, on="ts", direction="backward")',
        'pd.merge(left, right, on="ts")',
    ],
)
def test_la007_stays_quiet_on_point_in_time_join(codes_for, source: str) -> None:
    assert codes_for(source) == []


def test_findings_carry_actionable_positions(lint) -> None:
    findings = lint('frame["y"] = close.shift(-1)\n')
    assert len(findings) == 1
    finding = findings[0]
    assert (finding.line, finding.col, finding.end_line) == (1, 14, 1)
    assert finding.location == "1:14"
    assert finding.rule is RULES["LA001"]
    assert "periods=-1" in finding.message


def test_selecting_codes_runs_only_those_rules(lint) -> None:
    source = """
    frame["y"] = close.shift(-1)
    smooth = close.rolling(5, center=True).mean()
    """
    assert {finding.code for finding in lint(source)} == {"LA001", "LA003"}
    assert [finding.code for finding in lint(source, ["LA003"])] == ["LA003"]
