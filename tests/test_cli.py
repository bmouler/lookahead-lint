"""End-to-end command-line behaviour, including exit codes and failure handling."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from lookahead_lint.cli import EXIT_ERROR, EXIT_FINDINGS, EXIT_OK, main

LEAK = "target = close.shift(-1)\nsmooth = close.rolling(5, center=True).mean()\n"


def _run(capsys: pytest.CaptureFixture[str], *argv: str) -> tuple[int, str, str]:
    code = main(list(argv))
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def test_clean_example_exits_zero(capsys: pytest.CaptureFixture[str], examples: Path) -> None:
    code, out, err = _run(capsys, str(examples / "clean_research.py"), "--no-config")
    assert code == EXIT_OK
    assert out == "no findings (1 file checked)\n"
    assert err == ""


def test_leaky_example_exits_one(capsys: pytest.CaptureFixture[str], examples: Path) -> None:
    code, out, _ = _run(capsys, str(examples / "leaky_research.py"), "--no-config")
    assert code == EXIT_FINDINGS
    assert "9 findings in 1 file (1 file checked)" in out


def test_directory_recursion_covers_python_and_notebooks(
    capsys: pytest.CaptureFixture[str], examples: Path
) -> None:
    code, out, _ = _run(capsys, str(examples), "--no-config", "--format", "json")
    assert code == EXIT_FINDINGS
    payload = json.loads(out)
    assert payload["summary"]["files_checked"] == 3
    assert payload["summary"]["findings"] == 12
    assert {Path(item["path"]).name for item in payload["findings"]} == {
        "leaky_research.py",
        "leaky_notebook.ipynb",
    }


def test_select_limits_the_rules(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    (tmp_path / "model.py").write_text(LEAK, encoding="utf-8")
    code, out, _ = _run(capsys, str(tmp_path), "--no-config", "--select", "LA003")
    assert code == EXIT_FINDINGS
    assert "LA003" in out
    assert "LA001" not in out


def test_ignore_accepts_repeated_and_comma_separated_values(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    (tmp_path / "model.py").write_text(LEAK, encoding="utf-8")
    code, out, _ = _run(capsys, str(tmp_path), "--no-config", "--ignore", "LA001,LA003")
    assert code == EXIT_OK
    assert out == "no findings (1 file checked)\n"


def test_unknown_code_is_a_usage_error(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    code, out, err = _run(capsys, str(tmp_path), "--no-config", "--select", "LA042")
    assert code == EXIT_ERROR
    assert out == ""
    assert "unknown rule code(s) LA042" in err


def test_missing_path_is_reported_not_ignored(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    code, out, _ = _run(capsys, str(tmp_path / "absent.py"), "--no-config")
    assert code == EXIT_ERROR
    assert "no such file or directory" in out


def test_unparseable_file_is_reported_and_the_run_continues(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    (tmp_path / "broken.py").write_text("def build(:\n    return 1\n", encoding="utf-8")
    (tmp_path / "model.py").write_text(LEAK, encoding="utf-8")
    code, out, _ = _run(capsys, str(tmp_path), "--no-config")
    assert code == EXIT_ERROR
    assert "broken.py" in out
    assert "error  SyntaxError:" in out
    assert "LA001" in out
    assert "2 findings in 1 file, 1 error (2 files checked)" in out


def test_file_with_a_null_byte_is_reported_not_crashed(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    (tmp_path / "binary.py").write_bytes(b"x = 1\n\x00\n")
    code, out, _ = _run(capsys, str(tmp_path), "--no-config")
    assert code == EXIT_ERROR
    assert "null bytes" in out


def test_undecodable_file_is_reported(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    (tmp_path / "latin.py").write_bytes(b"# -*- coding: utf-8 -*-\nname = '\xff\xfe'\n")
    code, out, _ = _run(capsys, str(tmp_path), "--no-config")
    assert code == EXIT_ERROR
    assert "cannot read" in out or "cannot decode" in out


def test_github_format_emits_annotations(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    target = tmp_path / "model.py"
    target.write_text(LEAK, encoding="utf-8")
    code, out, _ = _run(capsys, str(target), "--no-config", "--format", "github")
    assert code == EXIT_FINDINGS
    lines = out.splitlines()
    assert len(lines) == 2
    assert lines[0].startswith(f"::warning file={target},line=1,col=10,title=LA001")


def test_json_format_is_valid_json(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    (tmp_path / "model.py").write_text(LEAK, encoding="utf-8")
    code, out, _ = _run(capsys, str(tmp_path), "--no-config", "--format", "json")
    assert code == EXIT_FINDINGS
    payload = json.loads(out)
    assert [item["code"] for item in payload["findings"]] == ["LA001", "LA003"]


def test_configuration_is_honoured_and_can_be_disabled(
    capsys: pytest.CaptureFixture[str], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "pyproject.toml").write_text(
        '[tool.lookahead_lint]\nignore = ["LA001", "LA003"]\n', encoding="utf-8"
    )
    (tmp_path / "model.py").write_text(LEAK, encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    assert _run(capsys, "model.py")[0] == EXIT_OK
    assert _run(capsys, "model.py", "--no-config")[0] == EXIT_FINDINGS


def test_default_path_is_the_current_directory(
    capsys: pytest.CaptureFixture[str], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "model.py").write_text(LEAK, encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    code, out, _ = _run(capsys, "--no-config")
    assert code == EXIT_FINDINGS
    assert "model.py" in out


def test_suppression_survives_the_full_pipeline(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    source = "target = close.shift(-1)  # lookahead-lint: ignore[LA001]\n"
    (tmp_path / "model.py").write_text(source, encoding="utf-8")
    assert _run(capsys, str(tmp_path), "--no-config")[0] == EXIT_OK


def test_help_lists_every_rule(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exit_info:
        main(["--help"])
    assert exit_info.value.code == EXIT_OK
    out = capsys.readouterr().out
    assert "LA001  negative-shift" in out
    assert "LA007  forward-asof-merge" in out


def test_invalid_format_is_a_usage_error() -> None:
    with pytest.raises(SystemExit) as exit_info:
        main(["--format", "sarif"])
    assert exit_info.value.code == EXIT_ERROR
