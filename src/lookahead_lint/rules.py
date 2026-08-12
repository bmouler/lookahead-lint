"""Rule catalogue and the AST checks that implement it.

Every rule is deliberately narrow. A check only fires on a literal, unambiguous
idiom, because a linter that reports plausible-but-wrong findings gets
uninstalled. When a pattern cannot be decided from the syntax alone, the rule
stays silent.
"""

from __future__ import annotations

import ast
from collections.abc import Iterable, Iterator, Sequence
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


class _Checker(ast.NodeVisitor):
    """Base class carrying the reporting plumbing shared by every rule."""

    rule: ClassVar[Rule]

    def __init__(self, path: Path) -> None:
        self.path = path
        self.findings: list[Finding] = []

    def _report(self, node: ast.expr, detail: str | None = None) -> None:
        message = self.rule.message if detail is None else f"{self.rule.message} [{detail}]"
        self.findings.append(
            Finding(
                path=self.path,
                line=node.lineno,
                col=node.col_offset + 1,
                end_line=node.end_lineno or node.lineno,
                code=self.rule.code,
                message=message,
            )
        )


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

    def visit_Call(self, node: ast.Call) -> None:
        """Flag method calls named ``shift`` whose period argument is a negative literal."""
        if _method_name(node) == "shift":
            argument = node.args[0] if node.args else _keyword(node, "periods")
            periods = _negative_number(argument)
            if periods is not None:
                self._report(node, f"periods={periods:g}")
        self.generic_visit(node)


class BackwardFill(_Checker):
    """LA002: fills, interpolations and reindexes that carry values backward in time."""

    rule = LA002

    _BACKWARD_METHODS = frozenset({"bfill", "backfill"})
    _METHOD_KEYWORD_CALLS = frozenset({"fillna", "reindex", "asfreq", "align"})
    _BACKWARD_DIRECTIONS = frozenset({"backward", "both"})

    def visit_Call(self, node: ast.Call) -> None:
        """Flag ``.bfill()`` plus the keyword spellings of a backward fill."""
        name = _method_name(node)
        if name == "bfill":
            self._report(node, "bfill()")
        elif name in self._METHOD_KEYWORD_CALLS:
            method = _string_value(_keyword(node, "method"))
            if method is not None and method.lower() in self._BACKWARD_METHODS:
                self._report(node, f"{name}(method={method!r})")
        elif name == "interpolate":
            direction = _string_value(_keyword(node, "limit_direction"))
            if direction is not None and direction.lower() in self._BACKWARD_DIRECTIONS:
                self._report(node, f"interpolate(limit_direction={direction!r})")
        self.generic_visit(node)


class CenteredWindow(_Checker):
    """LA003: ``.rolling(..., center=True)``."""

    rule = LA003

    def visit_Call(self, node: ast.Call) -> None:
        """Flag rolling windows explicitly centered on the current observation."""
        if _method_name(node) == "rolling" and _is_true_literal(_keyword(node, "center")):
            self._report(node)
        self.generic_visit(node)


_SCOPE_NODES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)


def _iter_scope_nodes(body: Sequence[ast.stmt]) -> Iterator[ast.AST]:
    """Yield every node under ``body`` without descending into nested scopes.

    The nested scope node itself is yielded (so callers can recurse into it
    deliberately), but its children are not, which keeps ordering checks from
    comparing line numbers across unrelated scopes.
    """
    stack: list[ast.AST] = list(body)
    while stack:
        node = stack.pop()
        yield node
        if isinstance(node, _SCOPE_NODES):
            continue
        stack.extend(ast.iter_child_nodes(node))


class FitBeforeSplit(_Checker):
    """LA004: a ``.fit``/``.fit_transform`` call that precedes ``train_test_split``."""

    rule = LA004

    _FIT_METHODS = frozenset({"fit", "fit_transform"})

    def visit_Module(self, node: ast.Module) -> None:
        """Analyze module scope, then every nested scope in turn."""
        self._check_scope(node.body)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Analyze a function body as an independent scope."""
        self._check_scope(node.body)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        """Analyze an async function body as an independent scope."""
        self._check_scope(node.body)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        """Analyze a class body as an independent scope."""
        self._check_scope(node.body)

    def _check_scope(self, body: Sequence[ast.stmt]) -> None:
        nodes = list(_iter_scope_nodes(body))
        calls = [node for node in nodes if isinstance(node, ast.Call)]
        split_lines = [call.lineno for call in calls if _called_name(call) == "train_test_split"]
        if split_lines:
            first_split = min(split_lines)
            for call in calls:
                if _method_name(call) in self._FIT_METHODS and call.lineno < first_split:
                    self._report(call, f"split at line {first_split}")
        for node in nodes:
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
                self.visit(node)


class ShuffledSplit(_Checker):
    """LA005: ``train_test_split`` that is not pinned to ``shuffle=False``."""

    rule = LA005

    def visit_Call(self, node: ast.Call) -> None:
        """Flag splits that shuffle, which is the sklearn default."""
        if _called_name(node) == "train_test_split" and not _has_kwargs_unpacking(node):
            if not _is_false_literal(_keyword(node, "shuffle")):
                self._report(node)
        self.generic_visit(node)


class FutureRowIndex(_Checker):
    """LA006: ``series[i + k]`` inside a loop whose target is ``i``."""

    rule = LA006

    def __init__(self, path: Path) -> None:
        super().__init__(path)
        self._loop_vars: list[frozenset[str]] = []

    def visit_For(self, node: ast.For) -> None:
        """Track the loop target while walking the loop body."""
        self._visit_loop(node)

    def visit_AsyncFor(self, node: ast.AsyncFor) -> None:
        """Track the loop target while walking an async loop body."""
        self._visit_loop(node)

    def visit_ListComp(self, node: ast.ListComp) -> None:
        """Comprehensions are loops too; track their targets the same way."""
        self._visit_comprehension(node, node.generators)

    def visit_SetComp(self, node: ast.SetComp) -> None:
        """Comprehensions are loops too; track their targets the same way."""
        self._visit_comprehension(node, node.generators)

    def visit_DictComp(self, node: ast.DictComp) -> None:
        """Comprehensions are loops too; track their targets the same way."""
        self._visit_comprehension(node, node.generators)

    def visit_GeneratorExp(self, node: ast.GeneratorExp) -> None:
        """Comprehensions are loops too; track their targets the same way."""
        self._visit_comprehension(node, node.generators)

    def visit_Subscript(self, node: ast.Subscript) -> None:
        """Flag subscripts offset ahead of an active loop variable."""
        for index in self._index_expressions(node.slice):
            if self._is_future_offset(index):
                self._report(node)
                break
        self.generic_visit(node)

    @staticmethod
    def _index_expressions(node: ast.expr) -> Iterator[ast.expr]:
        if isinstance(node, ast.Tuple):
            yield from node.elts
        else:
            yield node

    def _visit_loop(self, node: ast.For | ast.AsyncFor) -> None:
        self.visit(node.iter)
        self._loop_vars.append(_target_names(node.target))
        for child in (*node.body, *node.orelse):
            self.visit(child)
        self._loop_vars.pop()

    def _visit_comprehension(self, node: ast.expr, generators: list[ast.comprehension]) -> None:
        names: set[str] = set()
        for generator in generators:
            names |= _target_names(generator.target)
        self._loop_vars.append(frozenset(names))
        self.generic_visit(node)
        self._loop_vars.pop()

    def _is_future_offset(self, index: ast.expr) -> bool:
        if not isinstance(index, ast.BinOp) or not isinstance(index.op, ast.Add):
            return False
        active = {name for loop_vars in self._loop_vars for name in loop_vars}
        for operand, other in ((index.left, index.right), (index.right, index.left)):
            if isinstance(operand, ast.Name) and operand.id in active:
                # `i + -1` is a look-behind written oddly; only positive offsets look ahead.
                if _negative_number(other) is None:
                    return True
        return False


def _target_names(target: ast.expr) -> frozenset[str]:
    """Collect every simple name bound by a ``for`` target, including tuple targets."""
    return frozenset(node.id for node in ast.walk(target) if isinstance(node, ast.Name))


class ForwardAsofMerge(_Checker):
    """LA007: ``merge_asof(..., direction="forward"|"nearest")``."""

    rule = LA007

    _FUTURE_DIRECTIONS = frozenset({"forward", "nearest"})

    def visit_Call(self, node: ast.Call) -> None:
        """Flag as-of joins whose direction can match a later record."""
        if _called_name(node) == "merge_asof":
            direction = _string_value(_keyword(node, "direction"))
            if direction is not None and direction.lower() in self._FUTURE_DIRECTIONS:
                self._report(node, f"direction={direction!r}")
        self.generic_visit(node)


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
    findings: list[Finding] = []
    for checker_class in CHECKERS:
        if checker_class.rule.code not in enabled:
            continue
        checker = checker_class(path)
        checker.visit(tree)
        findings.extend(checker.findings)
    findings.sort(key=lambda finding: (finding.line, finding.col, finding.code))
    return findings
