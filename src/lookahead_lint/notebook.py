"""Jupyter notebook support.

Code cells are concatenated into a single module so that scope-level checks
still see the whole story, and a line map translates every position back to
``cell N, line M`` where the reader can act on it.
"""

from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass
from pathlib import Path

__all__ = ["NotebookError", "NotebookSource", "load_notebook", "notebook_source"]

_MAGIC_RE = re.compile(r"^\s*(?:[%!]|[A-Za-z_][\w.]*\s*=\s*[%!])")


class NotebookError(ValueError):
    """Raised when a file is not a readable Jupyter notebook."""


@dataclass(frozen=True)
class NotebookSource:
    """Concatenated notebook code with a map back to cell coordinates.

    Attributes:
        source: Python source built from the code cells, joined by blank lines.
        line_map: One ``(cell, line_in_cell)`` pair per line of ``source``, both
            1-based. Code cells are numbered in document order; markdown and raw
            cells are skipped and do not consume a number.
    """

    source: str
    line_map: tuple[tuple[int, int], ...]

    def locate(self, line: int) -> tuple[int, int]:
        """Translate a 1-based line of :attr:`source` into ``(cell, line_in_cell)``.

        Args:
            line: 1-based line number in the concatenated source.

        Returns:
            The cell number and the 1-based line within that cell. Positions past
            the end of the source (as a trailing syntax error can report) clamp to
            the last known line.

        Raises:
            ValueError: If the notebook has no code cells, or ``line`` is below 1.
        """
        if not self.line_map:
            raise ValueError("notebook contains no code cells")
        if line < 1:
            raise ValueError(f"line must be 1-based, got {line}")
        index = min(line, len(self.line_map)) - 1
        return self.line_map[index]


def load_notebook(path: Path) -> NotebookSource:
    """Read a ``.ipynb`` file and return its concatenated code cells.

    Args:
        path: Path to a Jupyter notebook.

    Returns:
        The concatenated source and its line map.

    Raises:
        NotebookError: If the file is not valid JSON or is not shaped like a
            notebook document.
    """
    try:
        raw = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as error:
        raise NotebookError(f"{path}: not valid UTF-8: {error}") from error
    return notebook_source(raw, name=str(path))


def notebook_source(text: str, name: str = "<notebook>") -> NotebookSource:
    """Build a :class:`NotebookSource` from notebook JSON text.

    Args:
        text: Contents of a ``.ipynb`` document.
        name: Label used in error messages.

    Returns:
        The concatenated source and its line map.

    Raises:
        NotebookError: If the JSON is malformed or has no ``cells`` list.
    """
    try:
        document = json.loads(text)
    except json.JSONDecodeError as error:
        raise NotebookError(
            f"{name}: invalid notebook JSON: {error.msg} at line {error.lineno}"
        ) from error
    if not isinstance(document, dict) or not isinstance(document.get("cells"), list):
        raise NotebookError(f"{name}: not a notebook document (no 'cells' list)")

    lines: list[str] = []
    line_map: list[tuple[int, int]] = []
    cell_number = 0
    for index, cell in enumerate(document["cells"]):
        if not isinstance(cell, dict):
            raise NotebookError(f"{name}: cell {index} is not an object")
        if cell.get("cell_type") != "code":
            continue
        cell_number += 1
        cell_lines = _clean_cell(_cell_text(cell, name, index))
        if line_map:
            previous_cell, previous_line = line_map[-1]
            lines.append("")
            line_map.append((previous_cell, previous_line + 1))
        for offset, text_line in enumerate(cell_lines, start=1):
            lines.append(text_line)
            line_map.append((cell_number, offset))
    return NotebookSource("\n".join(lines) + "\n", tuple(line_map))


def _cell_text(cell: dict[str, object], name: str, index: int) -> str:
    source = cell.get("source", "")
    if isinstance(source, str):
        return source
    if isinstance(source, list) and all(isinstance(part, str) for part in source):
        return "".join(source)
    raise NotebookError(f"{name}: cell {index} has an unreadable 'source' field")


def _clean_cell(text: str) -> list[str]:
    """Return the cell's lines, blanking IPython magics only when they break parsing.

    A cell that is already valid Python is never modified, so an expression such
    as ``a\n% b`` keeps its meaning. Only cells that fail to parse are stripped of
    ``%magic``, ``!shell`` and ``name = !shell`` lines, each replaced by an empty
    line so numbering is preserved.
    """
    lines = text.splitlines()
    if _parses("\n".join(lines)):
        return lines
    return ["" if _MAGIC_RE.match(line) else line for line in lines]


def _parses(source: str) -> bool:
    try:
        ast.parse(source)
    except SyntaxError:
        return False
    return True
