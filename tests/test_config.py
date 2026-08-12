"""Loading and applying the optional [tool.lookahead_lint] table."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from lookahead_lint import ALL_CODES, Config, ConfigError, analyze_paths, discover_config
from lookahead_lint.config import load_config


def _write_pyproject(directory: Path, body: str) -> Path:
    path = directory / "pyproject.toml"
    path.write_text(textwrap.dedent(body), encoding="utf-8")
    return path


def test_missing_table_yields_defaults(tmp_path: Path) -> None:
    path = _write_pyproject(tmp_path, '[project]\nname = "thing"\n')
    config = load_config(path)
    assert config == Config()
    assert config.source is None
    assert config.enabled_codes() == ALL_CODES


def test_select_and_ignore_are_loaded(tmp_path: Path) -> None:
    path = _write_pyproject(
        tmp_path,
        """
        [tool.lookahead_lint]
        select = ["LA001", "LA002", "LA003"]
        ignore = ["la002"]
        exclude = ["notebooks", "*_generated.py"]
        """,
    )
    config = load_config(path)
    assert config.select == frozenset({"LA001", "LA002", "LA003"})
    assert config.ignore == frozenset({"LA002"})
    assert config.exclude == ("notebooks", "*_generated.py")
    assert config.enabled_codes() == frozenset({"LA001", "LA003"})
    assert config.source == path


def test_ignore_alone_disables_only_those_rules(tmp_path: Path) -> None:
    path = _write_pyproject(tmp_path, '[tool.lookahead_lint]\nignore = ["LA005"]\n')
    assert load_config(path).enabled_codes() == ALL_CODES - {"LA005"}


def test_select_and_ignore_cannot_cancel_out(tmp_path: Path) -> None:
    path = _write_pyproject(
        tmp_path,
        """
        [tool.lookahead_lint]
        select = ["LA001"]
        ignore = ["LA001"]
        """,
    )
    with pytest.raises(ConfigError, match="no rules left to run"):
        load_config(path).enabled_codes()


@pytest.mark.parametrize(
    ("body", "match"),
    [
        ('[tool.lookahead_lint]\nselcet = ["LA001"]\n', "unknown key"),
        ('[tool.lookahead_lint]\nselect = ["LA042"]\n', "unknown rule code"),
        ("[tool.lookahead_lint]\nselect = 3\n", "must be a list of strings"),
        ('[tool.lookahead_lint]\nexclude = "notebooks"\n', "must be a list of strings"),
        ("[tool.lookahead_lint]\nignore = [1, 2]\n", "must be a list of strings"),
    ],
)
def test_malformed_tables_are_rejected(tmp_path: Path, body: str, match: str) -> None:
    path = _write_pyproject(tmp_path, body)
    with pytest.raises(ConfigError, match=match):
        load_config(path)


def test_invalid_toml_is_rejected(tmp_path: Path) -> None:
    path = _write_pyproject(tmp_path, "[tool.lookahead_lint\n")
    with pytest.raises(ConfigError, match="invalid TOML"):
        load_config(path)


def test_unreadable_pyproject_is_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "pyproject.toml"
    path.write_text("[tool.lookahead_lint]\n", encoding="utf-8")

    def deny_read(*args: object, **kwargs: object) -> str:
        raise OSError("permission denied")

    monkeypatch.setattr(Path, "read_text", deny_read)

    with pytest.raises(ConfigError, match="cannot read: permission denied"):
        load_config(path)


def test_lookahead_lint_value_must_be_a_table(tmp_path: Path) -> None:
    path = _write_pyproject(tmp_path, 'tool = { lookahead_lint = "LA001" }\n')

    with pytest.raises(ConfigError, match=r"\[tool\.lookahead_lint\] must be a table"):
        load_config(path)


def test_empty_ignore_list_is_loaded(tmp_path: Path) -> None:
    path = _write_pyproject(tmp_path, "[tool.lookahead_lint]\nignore = []\n")

    config = load_config(path)

    assert config.ignore == frozenset()
    assert config.source == path


def test_omitted_ignore_uses_the_default_empty_set(tmp_path: Path) -> None:
    path = _write_pyproject(tmp_path, '[tool.lookahead_lint]\nselect = ["LA001"]\n')

    config = load_config(path)

    assert config.ignore == frozenset()
    assert config.enabled_codes() == frozenset({"LA001"})


def test_discovery_walks_up_to_the_nearest_table(tmp_path: Path) -> None:
    _write_pyproject(tmp_path, '[tool.lookahead_lint]\nignore = ["LA005"]\n')
    nested = tmp_path / "research" / "signals"
    nested.mkdir(parents=True)
    _write_pyproject(nested.parent, '[project]\nname = "inner"\n')
    (nested / "model.py").write_text("x = 1\n", encoding="utf-8")

    config = discover_config([nested / "model.py"])
    assert config.ignore == frozenset({"LA005"})


def test_discovery_returns_defaults_when_no_table_exists(tmp_path: Path) -> None:
    directory = tmp_path / "empty"
    directory.mkdir()
    assert discover_config([directory]).source is None


def test_command_line_overrides_replace_configured_codes() -> None:
    config = Config(select=frozenset({"LA001"}), ignore=frozenset({"LA002"}))
    overridden = config.with_overrides(select=["la003", "LA004"], ignore=[])
    assert overridden.select == frozenset({"LA003", "LA004"})
    assert overridden.ignore == frozenset()
    assert config.with_overrides().select == frozenset({"LA001"})


def test_command_line_overrides_reject_unknown_codes() -> None:
    with pytest.raises(ConfigError, match=r"--select: unknown rule code\(s\) LA900"):
        Config().with_overrides(select=["LA900"])


def test_exclude_patterns_match_names_directories_and_paths() -> None:
    config = Config(exclude=("notebooks", "*_generated.py", "research/vendor/*"))
    assert config.is_excluded(Path("notebooks/explore.py"))
    assert config.is_excluded(Path("src/features_generated.py"))
    assert config.is_excluded(Path("research/vendor/thirdparty.py"))
    assert not config.is_excluded(Path("research/signals/momentum.py"))


def test_always_excluded_directories_are_skipped_without_configuration() -> None:
    config = Config()
    assert config.is_excluded(Path("project/.venv/lib/pandas.py"))
    assert config.is_excluded(Path("project/__pycache__/module.py"))
    assert not config.is_excluded(Path("project/src/module.py"))


def test_exclusion_applies_to_directory_recursion(tmp_path: Path) -> None:
    (tmp_path / "notebooks").mkdir()
    (tmp_path / "src").mkdir()
    leak = "target = close.shift(-1)\n"
    (tmp_path / "notebooks" / "explore.py").write_text(leak, encoding="utf-8")
    (tmp_path / "src" / "features.py").write_text(leak, encoding="utf-8")

    report = analyze_paths([tmp_path], Config(exclude=("notebooks",)))
    assert report.files_checked == 1
    assert [str(finding.path.name) for finding in report.findings] == ["features.py"]


def test_selected_codes_reach_the_analysis(tmp_path: Path) -> None:
    source = "target = close.shift(-1)\nsmooth = close.rolling(5, center=True).mean()\n"
    (tmp_path / "model.py").write_text(source, encoding="utf-8")
    report = analyze_paths([tmp_path], Config(select=frozenset({"LA003"})))
    assert [finding.code for finding in report.findings] == ["LA003"]
