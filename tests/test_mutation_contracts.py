"""Focused public-contract checks for mutation-sensitive behavior."""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from lookahead_lint import (
    ALL_CODES,
    Config,
    ConfigError,
    LintError,
    Report,
    analyze_file,
    analyze_paths,
    analyze_source,
    collect_files,
    collect_suppressions,
    discover_config,
    load_notebook,
    notebook_source,
    render,
    run_checks,
)
from lookahead_lint.cli import build_parser, main
from lookahead_lint.config import load_config
from lookahead_lint.reporter import render_json, render_text


def test_cli_help_is_the_complete_public_interface() -> None:
    help_text = build_parser().format_help()
    assert help_text.startswith(
        "usage: lookahead-lint [-h] [--format {text,json,github}] [--select CODES]\n"
    )
    assert "[--ignore CODES] [--no-config] [--version]" in help_text
    assert "[PATH ...]" in help_text
    assert (
        "Flag look-ahead bias and target leakage idioms in Python and Jupyter research code. "
        "Exits 1 when findings remain, 2 on a usage or parse error."
    ) in help_text
    assert (
        "PATH                  files or directories to analyze (default: the current"
    ) in help_text
    assert "--format {text,json,github}" in help_text
    assert "output format (default: text)" in help_text
    assert "--select CODES        run only these comma-separated codes; repeatable" in help_text
    assert "--ignore CODES        disable these comma-separated codes; repeatable" in help_text
    assert "--no-config           ignore any [tool.lookahead_lint] table in" in help_text
    for code in sorted(ALL_CODES):
        assert f"  {code}  " in help_text


def test_cli_public_help_describes_the_output_format(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as raised:
        main(["--help"])
    assert raised.value.code == 0
    help_text = capsys.readouterr().out
    assert "\n  --format {text,json,github}\n" in help_text
    assert "\n                        output format (default: text)\n" in help_text


def test_cli_parser_accepts_zero_or_many_paths_and_repeated_codes() -> None:
    parser = build_parser()
    defaults = parser.parse_args([])
    assert defaults.paths == ["."]
    assert defaults.output_format == "text"
    parsed = parser.parse_args(
        ["first.py", "second.ipynb", "--select", "LA001", "--select", "LA003", "--ignore", "LA002"]
    )
    assert parsed.paths == ["first.py", "second.ipynb"]
    assert parsed.select == ["LA001", "LA003"]
    assert parsed.ignore == ["LA002"]


def test_cli_version_is_stable(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as raised:
        main(["--version"])
    assert raised.value.code == 0
    assert capsys.readouterr().out == "lookahead-lint 1.0.0\n"


def test_config_discovery_uses_lowercase_pyproject_name(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    nested = tmp_path / "work"
    nested.mkdir()
    (tmp_path / "pyproject.toml").write_text(
        '[tool.lookahead_lint]\nignore = ["LA002"]\n', encoding="utf-8"
    )
    original_is_file = Path.is_file

    def case_sensitive_is_file(path: Path) -> bool:
        if path.name == "PYPROJECT.TOML":
            return False
        return original_is_file(path)

    monkeypatch.setattr(Path, "is_file", case_sensitive_is_file)
    config = discover_config([nested])
    assert config.ignore == frozenset({"LA002"})
    assert config.source == tmp_path / "pyproject.toml"


def test_cli_discovers_config_from_an_explicit_path(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    invocation_dir = tmp_path / "invocation"
    invocation_dir.mkdir()
    target_dir = tmp_path / "project"
    target_dir.mkdir()
    target = target_dir / "model.py"
    target.write_text("target = close.shift(-1)\n", encoding="utf-8")
    (target_dir / "pyproject.toml").write_text(
        '[tool.lookahead_lint]\nignore = ["LA001"]\n',
        encoding="utf-8",
    )
    monkeypatch.chdir(invocation_dir)

    assert main([str(target)]) == 0
    assert capsys.readouterr().out == "no findings (1 file checked)\n"


def test_config_reads_explicit_utf8(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    path = tmp_path / "pyproject.toml"
    seen: list[str | None] = []

    def read_text(self: Path, *, encoding: str | None = None) -> str:
        assert self == path
        seen.append(encoding)
        return '[tool.lookahead_lint]\nselect = ["LA001"]\n'

    monkeypatch.setattr(Path, "read_text", read_text)
    assert load_config(path).select == frozenset({"LA001"})
    assert seen == ["utf-8"]


def test_config_diagnostics_are_complete(tmp_path: Path) -> None:
    path = tmp_path / "pyproject.toml"
    path.write_text("[tool.lookahead_lint]\nzebra = 1\nalpha = 2\n", encoding="utf-8")
    with pytest.raises(ConfigError) as raised:
        load_config(path)
    assert str(raised.value) == (
        f"{path}: unknown key(s) alpha, zebra; allowed: exclude, ignore, select"
    )

    path.write_text('[tool.lookahead_lint]\nselect = ["LA999", "LA888"]\n', encoding="utf-8")
    with pytest.raises(ConfigError) as raised:
        load_config(path)
    assert str(raised.value) == (
        "[tool.lookahead_lint].select: unknown rule code(s) LA888, LA999; known: "
        + ", ".join(sorted(ALL_CODES))
    )

    path.write_text('[tool.lookahead_lint]\nignore = ["LA999"]\n', encoding="utf-8")
    with pytest.raises(ConfigError, match=r"^\[tool\.lookahead_lint\]\.ignore:"):
        load_config(path)

    path.write_text("[tool.lookahead_lint]\nexclude = [1]\n", encoding="utf-8")
    with pytest.raises(ConfigError, match=rf"^{path}: \[tool\.lookahead_lint\]\.exclude"):
        load_config(path)


def test_render_defaults_to_text_and_pins_wire_format() -> None:
    report = Report(errors=(LintError(Path("z.py"), 2, 3, "bad"),), files_checked=1)
    expected = "z.py\n  2:3  error  bad\n\n0 findings in 0 files, 1 error (1 file checked)\n"
    assert render(report) == expected
    assert render_text(report) == expected
    assert render_json(Report(files_checked=1)) == (
        '{\n  "errors": [],\n  "findings": [],\n  "schema_version": 1,\n'
        '  "summary": {\n    "errors": 0,\n    "files_checked": 1,\n'
        '    "findings": 0\n  },\n  "tool": "lookahead-lint"\n}\n'
    )
    with pytest.raises(
        ValueError,
        match=r"^unknown format 'sarif'; choose from text, json, github$",
    ):
        render(Report(), "sarif")


def test_text_report_preserves_finding_fields_and_legend() -> None:
    findings, errors = analyze_source("value = close.shift(-1)\n", Path("a.py"))
    assert errors == []
    text = render_text(Report(tuple(findings), files_checked=1))
    assert text.startswith(
        "a.py\n  1:9  LA001  negative shift pulls future rows into the present; "
        "legitimate only for label construction"
    )
    assert "\nrules\n  LA001 negative-shift\n" in text


def test_github_data_escapes_carriage_returns() -> None:
    output = render(
        Report(errors=(LintError(Path("a.py"), 1, 1, "first\rsecond"),)),
        "github",
    )
    assert output.endswith("::first%0Dsecond\n")


def test_analyze_source_preserves_filename_and_syntax_position() -> None:
    findings, errors = analyze_source("def broken(:\n", Path("research/broken.py"))
    assert findings == []
    assert errors == [LintError(Path("research/broken.py"), 1, 12, "SyntaxError: invalid syntax")]


def test_analyze_source_passes_full_span_to_suppression() -> None:
    source = (
        "value = close.rolling(\n    5,\n    center=True,\n)  # lookahead-lint: ignore[LA003]\n"
    )
    assert analyze_source(source, Path("model.py")) == ([], [])


def test_analyze_source_defaults_missing_syntax_coordinates_to_one() -> None:
    findings, errors = analyze_source("\0", Path("nul.py"))
    assert findings == []
    assert errors == [
        LintError(
            Path("nul.py"),
            1,
            1,
            "SyntaxError: source code string cannot contain null bytes",
        )
    ]


def test_analyze_source_suppresses_a_multiline_finding_from_its_first_line() -> None:
    source = (
        "value = close.rolling(  # lookahead-lint: ignore[LA003]\n    5,\n    center=True,\n)\n"
    )
    assert analyze_source(source, Path("model.py")) == ([], [])


def test_collect_files_reports_missing_paths_and_preserves_explicit_files(
    tmp_path: Path,
) -> None:
    included = tmp_path / "included.py"
    included.write_text("x = 1\n", encoding="utf-8")
    explicit = tmp_path / "explicit.txt"
    explicit.write_text("not Python, but explicitly requested", encoding="utf-8")
    ignored = tmp_path / "ignored.py"
    ignored.write_text("x = 2\n", encoding="utf-8")
    (tmp_path / "notes.txt").write_text("not recursively collected", encoding="utf-8")
    missing = tmp_path / "missing.py"

    files, errors = collect_files(
        [tmp_path, explicit, included, missing],
        Config(exclude=("ignored.py",)),
    )

    assert files == [included, explicit]
    assert errors == [LintError(missing, 1, 1, "no such file or directory")]


def test_analyze_file_error_records_are_exact(tmp_path: Path) -> None:
    notebook = tmp_path / "bad.ipynb"
    notebook.write_text("not json", encoding="utf-8")
    assert analyze_file(notebook) == (
        [],
        [
            LintError(
                notebook,
                1,
                1,
                f"{notebook}: invalid notebook JSON: Expecting value at line 1",
            )
        ],
    )

    unknown = tmp_path / "unknown.py"
    unknown.write_bytes(b"# coding: no-such-codec\n")
    findings, errors = analyze_file(unknown)
    assert findings == []
    assert errors == [
        LintError(
            unknown,
            1,
            1,
            f"cannot decode: unknown encoding for '{unknown}': no-such-codec",
        )
    ]

    missing = tmp_path / "missing.py"
    assert analyze_file(missing) == (
        [],
        [
            LintError(
                missing,
                1,
                1,
                f"cannot read: [Errno 2] No such file or directory: '{missing}'",
            )
        ],
    )


def test_analyze_file_honors_selected_codes_for_notebooks(tmp_path: Path) -> None:
    path = tmp_path / "model.ipynb"
    path.write_text(
        json.dumps(
            {
                "cells": [
                    {
                        "cell_type": "code",
                        "source": ("x=close.shift(-1)\ny=close.rolling(2,center=True)"),
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    findings, errors = analyze_file(path, ["LA003"])
    assert errors == []
    assert [finding.code for finding in findings] == ["LA003"]


def test_analyze_paths_sorts_findings_and_errors(tmp_path: Path) -> None:
    z = tmp_path / "z.py"
    a = tmp_path / "a.py"
    z.write_text("x=close.shift(-1)\n", encoding="utf-8")
    a.write_text("x=close.shift(-1)\n", encoding="utf-8")
    missing_z = tmp_path / "z-missing.py"
    missing_a = tmp_path / "a-missing.py"
    report = analyze_paths([z, a, missing_z, missing_a], Config())
    assert [item.path for item in report.findings] == [a, z]
    assert [item.path for item in report.errors] == [missing_a, missing_z]
    assert report.files_checked == 2


def test_collect_suppressions_merges_codes_for_repeated_tokens(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from lookahead_lint import suppression

    original = suppression.tokenize.generate_tokens
    source = "x=1 # lookahead-lint: ignore[LA001]\n"
    tokens = list(original(iter(source.splitlines(keepends=True)).__next__))
    comment = next(token for token in tokens if token.type == suppression.tokenize.COMMENT)
    monkeypatch.setattr(
        suppression.tokenize,
        "generate_tokens",
        lambda _readline: iter(
            [comment, comment._replace(string="# lookahead-lint: ignore[LA003]")]
        ),
    )
    assert collect_suppressions(source).lines == {1: frozenset({"LA001", "LA003"})}


def test_notebook_loading_uses_utf8_and_filename(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    path = tmp_path / "model.ipynb"
    seen: list[str | None] = []

    def read_text(self: Path, *, encoding: str | None = None) -> str:
        assert self == path
        seen.append(encoding)
        return '{"cells": []}'

    monkeypatch.setattr(Path, "read_text", read_text)
    assert load_notebook(path).source == "\n"
    assert seen == ["utf-8"]

    with pytest.raises(Exception) as raised:
        notebook_source("not json", name="named.ipynb")
    assert str(raised.value) == "named.ipynb: invalid notebook JSON: Expecting value at line 1"


def test_notebook_source_default_diagnostic_label_is_public() -> None:
    with pytest.raises(ValueError) as raised:
        notebook_source("not json")
    assert str(raised.value) == "<notebook>: invalid notebook JSON: Expecting value at line 1"


def test_notebook_source_and_line_map_preserve_cell_boundaries() -> None:
    text = json.dumps(
        {
            "cells": [
                {"cell_type": "code", "source": "seed = 0"},
                {"cell_type": "code", "source": ["a = 1\n", "b = 2\n"]},
            ]
        }
    )
    extracted = notebook_source(text)
    assert extracted.source == "seed = 0\n\na = 1\nb = 2\n"
    assert extracted.line_map == ((1, 1), (1, 2), (2, 1), (2, 2))
    assert extracted.locate(100) == (2, 2)


def test_notebook_cell_diagnostic_includes_name_and_index() -> None:
    with pytest.raises(Exception) as raised:
        notebook_source(
            json.dumps({"cells": [{"cell_type": "markdown"}, {"cell_type": "code", "source": 7}]}),
            name="bad.ipynb",
        )
    assert str(raised.value) == "bad.ipynb: cell 1 has an unreadable 'source' field"


def test_notebook_code_cell_without_source_is_empty() -> None:
    extracted = notebook_source(json.dumps({"cells": [{"cell_type": "code"}]}))
    assert extracted.source == "\n"
    assert extracted.line_map == ()


def test_rule_messages_and_case_insensitive_keywords_are_public() -> None:
    cases = {
        "series.bfill()": (
            "backward fill propagates a later observation onto an earlier timestamp [bfill()]"
        ),
        'series.fillna(method="BFILL")': (
            "backward fill propagates a later observation onto an earlier timestamp "
            "[fillna(method='BFILL')]"
        ),
        'series.interpolate(limit_direction="BOTH")': (
            "backward fill propagates a later observation onto an earlier timestamp "
            "[interpolate(limit_direction='BOTH')]"
        ),
        'merge_asof(left, right, direction="FORWARD")': (
            "merge_asof with a forward/nearest direction matches rows dated after "
            "the key [direction='FORWARD']"
        ),
    }
    for source, message in cases.items():
        findings, errors = analyze_source(source, Path("rules.py"))
        assert errors == []
        assert [finding.message for finding in findings] == [message]


def test_fit_on_same_line_as_split_is_not_before_split() -> None:
    findings, errors = analyze_source(
        "scaled = scaler.fit_transform(train_test_split(frame, shuffle=False)[0])\n",
        Path("rules.py"),
    )
    assert errors == []
    assert findings == []


def test_comprehension_tracks_every_generator_variable() -> None:
    findings, errors = analyze_source(
        "values = [close[i + 1] for i in range(n) for j in range(m)]\n",
        Path("rules.py"),
    )
    assert errors == []
    assert [finding.code for finding in findings] == ["LA006"]


def test_future_row_index_traverses_nested_subscripts() -> None:
    findings, errors = analyze_source(
        "values = [frame[i + 1][i + 1] for i in range(n)]\n",
        Path("rules.py"),
    )
    assert errors == []
    assert [(finding.code, finding.line, finding.col) for finding in findings] == [
        ("LA006", 1, 11),
        ("LA006", 1, 11),
    ]


def test_run_checks_unknown_code_message_lists_every_unknown() -> None:
    with pytest.raises(KeyError) as raised:
        run_checks(ast.parse("x = 1"), Path("rules.py"), ["LA999", "LA888"])
    assert raised.value.args[0] == "unknown rule codes: LA888, LA999"


@pytest.mark.parametrize("key", ["select", "ignore"])
def test_config_list_type_errors_include_source_path(tmp_path: Path, key: str) -> None:
    path = tmp_path / "pyproject.toml"
    path.write_text(f'[tool.lookahead_lint]\n{key} = "LA001"\n', encoding="utf-8")

    with pytest.raises(ConfigError) as raised:
        load_config(path)

    assert str(raised.value) == (f"{path}: [tool.lookahead_lint].{key} must be a list of strings")


def test_text_report_sorts_findings_by_line_then_column() -> None:
    findings, errors = analyze_source(
        "wrapped = consume(close.shift(-1))\nleft = close.shift(-1); right = close.shift(-1)\n",
        Path("ordered.py"),
    )
    assert errors == []
    assert [(finding.line, finding.col) for finding in findings] == [
        (1, 19),
        (2, 8),
        (2, 33),
    ]

    report = Report(findings=(findings[2], findings[1], findings[0]), files_checked=1)
    file_block = render_text(report).split("\n\n", maxsplit=1)[0]
    assert [line.split()[0] for line in file_block.splitlines()[1:]] == [
        "1:19",
        "2:8",
        "2:33",
    ]


def test_text_report_sorts_errors_by_line_then_column() -> None:
    path = Path("ordered.py")
    report = Report(
        errors=(
            LintError(path, 2, 30, "later column"),
            LintError(path, 2, 2, "earlier column"),
            LintError(path, 1, 20, "earlier line"),
        )
    )

    file_block = render_text(report).split("\n\n", maxsplit=1)[0]
    assert [line.split(maxsplit=2) for line in file_block.splitlines()[1:]] == [
        ["1:20", "error", "earlier line"],
        ["2:2", "error", "earlier column"],
        ["2:30", "error", "later column"],
    ]
