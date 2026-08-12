"""Inline suppression comments.

Comments are read with :mod:`tokenize`, never with a regular expression over the
raw source. That distinction matters: a string literal containing the text
``# lookahead-lint: ignore`` is data, not a directive, and a regex over raw
source cannot tell the two apart.
"""

from __future__ import annotations

import io
import re
import tokenize
from dataclasses import dataclass, field

__all__ = ["ALL", "DIRECTIVE", "Suppressions", "collect_suppressions"]

DIRECTIVE = "lookahead-lint: ignore"
"""The literal directive text, documented so it can be referenced elsewhere."""

ALL = "*"
"""Sentinel stored for a bare directive, meaning every code on that line."""

_DIRECTIVE_RE = re.compile(
    r"#\s*lookahead-lint\s*:\s*ignore(?:\s*\[(?P<codes>[^\]]*)\])?(?=\s|$)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class Suppressions:
    """Per-line suppression directives found in a source file."""

    lines: dict[int, frozenset[str]] = field(default_factory=dict)

    def covers(self, code: str, start_line: int, end_line: int | None = None) -> bool:
        """Report whether ``code`` is suppressed for a finding spanning these lines.

        A directive suppresses a finding when it sits on any physical line of the
        offending expression, so a multi-line call can be annotated on its last
        line where the reader is actually looking.

        Args:
            code: Rule code of the finding, for example ``LA001``.
            start_line: First 1-based line of the offending expression.
            end_line: Last 1-based line of the expression; defaults to ``start_line``.

        Returns:
            True when a matching directive was found.
        """
        last = start_line if end_line is None else end_line
        wanted = code.upper()
        for line in range(start_line, last + 1):
            codes = self.lines.get(line)
            if codes is not None and (ALL in codes or wanted in codes):
                return True
        return False


def collect_suppressions(source: str) -> Suppressions:
    """Scan ``source`` for inline suppression comments.

    Args:
        source: Complete file contents. Must already be known to tokenize, which
            is guaranteed by parsing the module first.

    Returns:
        The directives found, keyed by 1-based physical line number. Two
        directives on one line are merged.
    """
    lines: dict[int, frozenset[str]] = {}
    readline = io.StringIO(source).readline
    for token in tokenize.generate_tokens(readline):
        if token.type != tokenize.COMMENT:
            continue
        match = _DIRECTIVE_RE.search(token.string)
        if match is None:
            continue
        raw_codes = match.group("codes")
        if raw_codes is None:
            codes = frozenset({ALL})
        else:
            codes = frozenset(part.strip().upper() for part in raw_codes.split(",") if part.strip())
        row = token.start[0]
        lines[row] = lines.get(row, frozenset()) | codes
    return Suppressions(lines)
