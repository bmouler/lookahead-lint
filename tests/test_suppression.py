"""Inline suppression, including the case a regex over raw source gets wrong."""

from __future__ import annotations

from lookahead_lint import collect_suppressions
from lookahead_lint.suppression import ALL


def test_specific_code_is_suppressed(codes_for) -> None:
    assert codes_for("target = close.shift(-1)  # lookahead-lint: ignore[LA001]") == []


def test_bare_directive_suppresses_every_code_on_the_line(codes_for) -> None:
    source = "smooth = close.shift(-1).rolling(5, center=True)  # lookahead-lint: ignore"
    assert codes_for(source) == []


def test_directive_only_suppresses_the_listed_codes(codes_for) -> None:
    source = "smooth = close.shift(-1).rolling(5, center=True)  # lookahead-lint: ignore[LA003]"
    assert codes_for(source) == ["LA001"]


def test_comma_separated_codes_are_all_suppressed(codes_for) -> None:
    source = (
        "smooth = close.shift(-1).rolling(5, center=True)  # lookahead-lint: ignore[LA001, LA003]"
    )
    assert codes_for(source) == []


def test_codes_are_case_insensitive(codes_for) -> None:
    assert codes_for("target = close.shift(-1)  # lookahead-lint: ignore[la001]") == []


def test_trailing_prose_after_the_directive_is_allowed(codes_for) -> None:
    source = "target = close.shift(-1)  # lookahead-lint: ignore[LA001] this is the label"
    assert codes_for(source) == []


def test_directive_must_be_a_whole_word(codes_for) -> None:
    assert codes_for("target = close.shift(-1)  # lookahead-lint: ignoreme") == ["LA001"]


def test_empty_bracket_list_suppresses_nothing(codes_for) -> None:
    assert codes_for("target = close.shift(-1)  # lookahead-lint: ignore[]") == ["LA001"]


def test_directive_inside_a_string_literal_is_not_a_directive(codes_for) -> None:
    """The tokenize-not-regex case: the text is data on a line that really leaks."""
    source = 'NOTE = "# lookahead-lint: ignore[LA001]"; target = close.shift(-1)\n'
    assert codes_for(source) == ["LA001"]


def test_directive_inside_a_docstring_is_not_a_directive(codes_for) -> None:
    source = '''
    def build(close):
        """Explains that # lookahead-lint: ignore[LA001] would silence this."""
        return close.shift(-1)
    '''
    assert codes_for(source) == ["LA001"]


def test_directive_on_the_closing_line_of_a_multiline_call(codes_for) -> None:
    source = """
    smooth = close.rolling(
        window=20,
        center=True,
    )  # lookahead-lint: ignore[LA003]
    """
    assert codes_for(source) == []


def test_directive_on_an_unrelated_line_does_not_suppress(codes_for) -> None:
    source = """
    # lookahead-lint: ignore[LA001]
    target = close.shift(-1)
    other = close.shift(-2)
    """
    assert codes_for(source) == ["LA001", "LA001"]


def test_collect_suppressions_records_lines_and_codes() -> None:
    source = "\n".join(
        [
            "a = 1  # lookahead-lint: ignore[LA001,LA002]",
            "b = 2  # lookahead-lint: ignore",
            "c = 3  # plain comment",
        ]
    )
    suppressions = collect_suppressions(source)
    assert suppressions.lines == {1: frozenset({"LA001", "LA002"}), 2: frozenset({ALL})}
    assert suppressions.covers("LA002", 1)
    assert not suppressions.covers("LA003", 1)
    assert suppressions.covers("LA007", 2)
    assert not suppressions.covers("LA001", 3)


def test_covers_spans_the_whole_expression() -> None:
    suppressions = collect_suppressions("x = 1  # lookahead-lint: ignore[LA004]\n")
    assert suppressions.covers("LA004", 1, 1)
    assert not suppressions.covers("LA004", 2, 3)
