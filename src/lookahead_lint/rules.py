"""Rule catalogue and the AST checks that implement it.

Every rule is deliberately narrow. A check only fires on a literal, unambiguous
idiom, because a linter that reports plausible-but-wrong findings gets
uninstalled. When a pattern cannot be decided from the syntax alone, the rule
stays silent.
"""

from __future__ import annotations

import ast
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

__all__ = [
    "ALL_CODES",
    "CHECKERS",
    "RULES",
    "Finding",
    "Rule",
    "run_checks",
]


@dataclass(frozen=True)
class Rule:
    """Static description of a single check.

    Attributes:
        code: Stable identifier, for example ``LA001``. Never reused.
        name: Short kebab-case name of the idiom the rule detects.
        message: The diagnostic text reported for a violation.
        rationale: One line explaining why the idiom leaks future information.
        fix: The concrete correction a reader should apply.
    """

    code: str
    name: str
    message: str
    rationale: str
    fix: str


@dataclass(frozen=True)
class Finding:
    """A single rule violation with its source location.

    ``line``/``col`` are 1-based and always refer to the analyzed source. For
    notebooks that source is the concatenation of the code cells, so ``cell``
    and ``cell_line`` carry the location a reader can actually navigate to.
    """

    path: Path
    line: int
    col: int
    end_line: int
    code: str
    message: str
    cell: int | None = None
    cell_line: int | None = None

    @property
    def rule(self) -> Rule:
        """The :class:`Rule` this finding was produced by."""
        return RULES[self.code]

    @property
    def location(self) -> str:
        """Human-readable ``line:col`` (or ``cell N, line M:col``) location."""
        if self.cell is None:
            return f"{self.line}:{self.col}"
        return f"cell {self.cell}, line {self.cell_line}:{self.col}"


LA001 = Rule(
    code="LA001",
    name="negative-shift",
    message=(
        "negative shift pulls future rows into the present; "
        "legitimate only for label construction, which should be suppressed inline"
    ),
    rationale="A negative shift reads values that are not observable at the row's timestamp.",
    fix=(
        "Shift features backward (positive periods) instead. For a label, keep the shift and "
        "add '# lookahead-lint: ignore[LA001]' so the intent stays visible in review."
    ),
)

LA002 = Rule(
    code="LA002",
    name="backward-fill",
    message="backward fill propagates a later observation onto an earlier timestamp",
    rationale="Filling backward copies data from the future into rows that predate it.",
    fix=(
        "Use forward fill (.ffill(), method='ffill', limit_direction='forward') so gaps are "
        "filled only from already-observed values."
    ),
)

LA003 = Rule(
    code="LA003",
    name="centered-window",
    message="centered rolling window includes observations after the current bar",
    rationale="A window centered on t spans (t - w/2, t + w/2), so half of it is unobservable.",
    fix="Drop center=True; a trailing window (the pandas default) only sees the past.",
)

LA004 = Rule(
    code="LA004",
    name="fit-before-split",
    message="fit call precedes train_test_split in this scope; statistics are learned on test rows",
    rationale="Statistics fitted on the full frame encode the held-out distribution.",
    fix="Split first, fit on the training rows only, then transform the held-out rows.",
)

LA005 = Rule(
    code="LA005",
    name="shuffled-split",
    message="train_test_split is not pinned to shuffle=False, so future rows enter training",
    rationale="Random splits of a time-ordered frame place later rows before earlier ones.",
    fix="Pass shuffle=False for ordered data, or split on an explicit date boundary.",
)

LA006 = Rule(
    code="LA006",
    name="future-row-index",
    message="index offset ahead of the loop variable reads a row that has not happened yet",
    rationale="Reading position i + k inside a loop over i consumes a bar from the future.",
    fix=(
        "Read the current or a past position (i, i - 1). If you need the next bar as a label, "
        "build it once as a shifted column and suppress that line explicitly."
    ),
)

LA007 = Rule(
    code="LA007",
    name="forward-asof-merge",
    message="merge_asof with a forward/nearest direction matches rows dated after the key",
    rationale="Only direction='backward' produces a point-in-time join.",
    fix="Use direction='backward' (the default) so each row joins the most recent prior record.",
)


class _Checker:
    """Rule metadata used to build the public catalogue."""

    rule: ClassVar[Rule]


def _method_name(call: ast.Call) -> str | None:
    """Return the attribute name for ``obj.method(...)`` calls, else ``None``."""
    return call.func.attr if isinstance(call.func, ast.Attribute) else None


def _called_name(call: ast.Call) -> str | None:
    """Return the trailing callable name for both ``f(...)`` and ``mod.f(...)``."""
    func = call.func
    if isinstance(func, ast.Attribute):
        return func.attr
    if isinstance(func, ast.Name):
        return func.id
    return None


def _keyword(call: ast.Call, name: str) -> ast.expr | None:
    """Return the value node of keyword ``name``, or ``None`` if not passed literally."""
    for keyword in call.keywords:
        if keyword.arg == name:
            return keyword.value
    return None


def _has_kwargs_unpacking(call: ast.Call) -> bool:
    """True when the call forwards ``**kwargs``, making keyword analysis unreliable."""
    return any(keyword.arg is None for keyword in call.keywords)


def _string_value(node: ast.expr | None) -> str | None:
    """Return the value of a string literal node, else ``None``."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _is_true_literal(node: ast.expr | None) -> bool:
    """True only for the literal ``True`` (never for truthy variables)."""
    return isinstance(node, ast.Constant) and node.value is True


def _is_false_literal(node: ast.expr | None) -> bool:
    """True only for the literal ``False``."""
    return isinstance(node, ast.Constant) and node.value is False


def _negative_number(node: ast.expr | None) -> int | float | None:
    """Return the value of a negative numeric literal such as ``-1``, else ``None``."""
    if not isinstance(node, ast.UnaryOp) or not isinstance(node.op, ast.USub):
        return None
    operand = node.operand
    if isinstance(operand, ast.Constant) and isinstance(operand.value, int | float):
        if isinstance(operand.value, bool):
            return None
        return -operand.value
    return None


class NegativeShift(_Checker):
    """LA001: ``.shift(-n)`` / ``.shift(periods=-n)`` with a negative literal."""

    rule = LA001


class BackwardFill(_Checker):
    """LA002: fills, interpolations and reindexes that carry values backward in time."""

    rule = LA002

    _BACKWARD_METHODS = frozenset({"bfill", "backfill"})
    _METHOD_KEYWORD_CALLS = frozenset({"fillna", "reindex", "asfreq", "align"})
    _BACKWARD_DIRECTIONS = frozenset({"backward", "both"})


class CenteredWindow(_Checker):
    """LA003: ``.rolling(..., center=True)``."""

    rule = LA003


class FitBeforeSplit(_Checker):
    """LA004: a ``.fit``/``.fit_transform`` call that precedes ``train_test_split``."""

    rule = LA004

    _FIT_METHODS = frozenset({"fit", "fit_transform"})


class ShuffledSplit(_Checker):
    """LA005: ``train_test_split`` that is not pinned to ``shuffle=False``."""

    rule = LA005


class FutureRowIndex(_Checker):
    """LA006: ``series[i + k]`` inside a loop whose target is ``i``."""

    rule = LA006


def _target_names(target: ast.expr) -> frozenset[str]:
    """Collect every simple name bound by a ``for`` target, including tuple targets."""
    return frozenset(node.id for node in ast.walk(target) if isinstance(node, ast.Name))


class ForwardAsofMerge(_Checker):
    """LA007: ``merge_asof(..., direction="forward"|"nearest")``."""

    rule = LA007

    _FUTURE_DIRECTIONS = frozenset({"forward", "nearest"})


CHECKERS: tuple[type[_Checker], ...] = (
    NegativeShift,
    BackwardFill,
    CenteredWindow,
    FitBeforeSplit,
    ShuffledSplit,
    FutureRowIndex,
    ForwardAsofMerge,
)

RULES: dict[str, Rule] = {checker.rule.code: checker.rule for checker in CHECKERS}
ALL_CODES: frozenset[str] = frozenset(RULES)


def _run_combined_checks(
    tree: ast.Module,
    path: Path,
    enabled: frozenset[str],
) -> list[Finding]:
    """Evaluate every enabled rule during one iterative walk of the tree."""
    negative_shift = LA001.code in enabled
    backward_fill = LA002.code in enabled
    centered_window = LA003.code in enabled
    fit_before_split = LA004.code in enabled
    shuffled_split = LA005.code in enabled
    future_row_index = LA006.code in enabled
    forward_asof_merge = LA007.code in enabled
    findings: list[Finding] = []
    split_lines: list[int | None] = [None]
    fit_calls: list[list[ast.Call]] = [[]]
    stack: list[tuple[ast.AST, int | None, frozenset[str]]] = [
        (tree, 0 if fit_before_split else None, frozenset())
    ]

    def report(rule: Rule, node: ast.expr, detail: str | None = None) -> None:
        message = rule.message if detail is None else f"{rule.message} [{detail}]"
        findings.append(
            Finding(
                path=path,
                line=node.lineno,
                col=node.col_offset + 1,
                end_line=node.end_lineno or node.lineno,
                code=rule.code,
                message=message,
            )
        )

    while stack:
        node, fit_scope, loop_vars = stack.pop()
        if isinstance(node, ast.Call):
            need_method = negative_shift or backward_fill or centered_window or fit_before_split
            method_name = _method_name(node) if need_method else None
            need_called = fit_before_split or shuffled_split or forward_asof_merge
            called_name = _called_name(node) if need_called else None

            if fit_before_split and fit_scope is not None:
                if called_name == "train_test_split":
                    previous = split_lines[fit_scope]
                    split_lines[fit_scope] = (
                        node.lineno if previous is None else min(previous, node.lineno)
                    )
                if method_name in FitBeforeSplit._FIT_METHODS:
                    fit_calls[fit_scope].append(node)

            if negative_shift and method_name == "shift":
                argument = node.args[0] if node.args else _keyword(node, "periods")
                periods = _negative_number(argument)
                if periods is not None:
                    report(LA001, node, f"periods={periods:g}")

            if backward_fill:
                if method_name == "bfill":
                    report(LA002, node, "bfill()")
                elif method_name in BackwardFill._METHOD_KEYWORD_CALLS:
                    method = _string_value(_keyword(node, "method"))
                    if method is not None and method.lower() in BackwardFill._BACKWARD_METHODS:
                        report(LA002, node, f"{method_name}(method={method!r})")
                elif method_name == "interpolate":
                    direction = _string_value(_keyword(node, "limit_direction"))
                    if (
                        direction is not None
                        and direction.lower() in BackwardFill._BACKWARD_DIRECTIONS
                    ):
                        report(LA002, node, f"interpolate(limit_direction={direction!r})")

            if (
                centered_window
                and method_name == "rolling"
                and _is_true_literal(_keyword(node, "center"))
            ):
                report(LA003, node)

            if (
                shuffled_split
                and called_name == "train_test_split"
                and not _has_kwargs_unpacking(node)
                and not _is_false_literal(_keyword(node, "shuffle"))
            ):
                report(LA005, node)

            if forward_asof_merge and called_name == "merge_asof":
                direction = _string_value(_keyword(node, "direction"))
                if (
                    direction is not None
                    and direction.lower() in ForwardAsofMerge._FUTURE_DIRECTIONS
                ):
                    report(LA007, node, f"direction={direction!r}")

        elif future_row_index and loop_vars and isinstance(node, ast.Subscript):
            indexes = node.slice.elts if isinstance(node.slice, ast.Tuple) else (node.slice,)
            for index in indexes:
                if not isinstance(index, ast.BinOp) or not isinstance(index.op, ast.Add):
                    continue
                for operand, other in (
                    (index.left, index.right),
                    (index.right, index.left),
                ):
                    if (
                        isinstance(operand, ast.Name)
                        and operand.id in loop_vars
                        and _negative_number(other) is None
                    ):
                        report(LA006, node)
                        break
                else:
                    continue
                break

        children: list[ast.AST] = []
        for field in node._fields:
            value = getattr(node, field, None)
            if isinstance(value, ast.AST):
                children.append(value)
            elif isinstance(value, list):
                for child in value:
                    if isinstance(child, ast.AST):
                        children.append(child)
        if not children:
            continue

        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            body_ids = {id(child) for child in node.body}
            if fit_before_split:
                child_scope = len(split_lines)
                split_lines.append(None)
                fit_calls.append([])
            else:
                child_scope = None
            for child in children:
                stack.append(
                    (
                        child,
                        child_scope if id(child) in body_ids else None,
                        loop_vars,
                    )
                )
        elif isinstance(node, ast.Lambda):
            for child in children:
                stack.append((child, None, loop_vars))
        elif future_row_index and isinstance(node, ast.For | ast.AsyncFor):
            active_body = loop_vars | _target_names(node.target)
            body_ids = {id(child) for child in (*node.body, *node.orelse)}
            for child in children:
                child_loop_vars: frozenset[str]
                if child is node.target:
                    child_loop_vars = frozenset()
                elif id(child) in body_ids:
                    child_loop_vars = active_body
                else:
                    child_loop_vars = loop_vars
                stack.append((child, fit_scope, child_loop_vars))
        elif future_row_index and isinstance(
            node,
            ast.ListComp | ast.SetComp | ast.DictComp | ast.GeneratorExp,
        ):
            names: set[str] = set()
            for generator in node.generators:
                names |= _target_names(generator.target)
            active_comprehension = loop_vars | names
            for child in children:
                stack.append((child, fit_scope, active_comprehension))
        else:
            for child in children:
                stack.append((child, fit_scope, loop_vars))

    if fit_before_split:
        for first_split, calls in zip(split_lines, fit_calls, strict=True):
            if first_split is None:
                continue
            for call in calls:
                if call.lineno < first_split:
                    report(LA004, call, f"split at line {first_split}")
    return findings


def run_checks(
    tree: ast.Module,
    path: Path,
    codes: Iterable[str] | None = None,
) -> list[Finding]:
    """Run the enabled checks over a parsed module.

    Args:
        tree: Parsed module to analyze.
        path: Path recorded on every finding; not read from disk.
        codes: Rule codes to run. ``None`` runs every rule.

    Returns:
        Findings sorted by line, column and code.

    Raises:
        KeyError: If ``codes`` contains a code that is not in :data:`RULES`.
    """
    enabled = ALL_CODES if codes is None else frozenset(codes)
    unknown = enabled - ALL_CODES
    if unknown:
        raise KeyError(f"unknown rule codes: {', '.join(sorted(unknown))}")
    findings = _run_combined_checks(tree, path, enabled)
    findings.sort(key=lambda finding: (finding.line, finding.col, finding.code))
    return findings
