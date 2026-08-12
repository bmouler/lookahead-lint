"""Shared fixtures for the test suite."""

from __future__ import annotations

import textwrap
from collections.abc import Callable, Iterable
from pathlib import Path

import pytest

from lookahead_lint import Finding, analyze_source

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


@pytest.fixture(scope="session")
def examples() -> Path:
    """Directory holding the shipped example fixtures."""
    return EXAMPLES


@pytest.fixture
def lint() -> Callable[..., list[Finding]]:
    """Return a helper that analyzes a source snippet and asserts it parsed."""

    def run(source: str, codes: Iterable[str] | None = None) -> list[Finding]:
        findings, errors = analyze_source(textwrap.dedent(source), Path("snippet.py"), codes)
        assert errors == [], f"snippet failed to parse: {errors}"
        return findings

    return run


@pytest.fixture
def codes_for(lint: Callable[..., list[Finding]]) -> Callable[..., list[str]]:
    """Return a helper that yields just the rule codes reported for a snippet."""

    def run(source: str) -> list[str]:
        return [finding.code for finding in lint(source)]

    return run
