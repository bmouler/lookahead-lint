"""Command-line interface."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from .analyzer import analyze_paths
from .config import Config, ConfigError, discover_config
from .reporter import FORMATS, render
from .rules import RULES

__all__ = ["EXIT_ERROR", "EXIT_FINDINGS", "EXIT_OK", "build_parser", "main"]

EXIT_OK = 0
"""No findings and no errors."""

EXIT_FINDINGS = 1
"""At least one finding survived suppression."""

EXIT_ERROR = 2
"""Usage error, bad configuration, or a file that could not be analyzed."""

_EPILOG = "rules:\n" + "\n".join(
    f"  {code}  {RULES[code].name:<18} {RULES[code].rationale}" for code in sorted(RULES)
)


def build_parser() -> argparse.ArgumentParser:
    """Construct the argument parser.

    Returns:
        A parser whose ``--help`` output lists every rule and its rationale.
    """
    parser = argparse.ArgumentParser(
        prog="lookahead-lint",
        description=(
            "Flag look-ahead bias and target leakage idioms in Python and Jupyter "
            "research code. Exits 1 when findings remain, 2 on a usage or parse error."
        ),
        epilog=_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "paths",
        nargs="*",
        default=["."],
        metavar="PATH",
        help="files or directories to analyze (default: the current directory)",
    )
    parser.add_argument(
        "--format",
        dest="output_format",
        choices=FORMATS,
        default="text",
        help="output format (default: text)",
    )
    parser.add_argument(
        "--select",
        action="append",
        metavar="CODES",
        help="run only these comma-separated codes; repeatable",
    )
    parser.add_argument(
        "--ignore",
        action="append",
        metavar="CODES",
        help="disable these comma-separated codes; repeatable",
    )
    parser.add_argument(
        "--no-config",
        action="store_true",
        help="ignore any [tool.lookahead_lint] table in pyproject.toml",
    )
    parser.add_argument("--version", action="version", version="lookahead-lint 0.1.0")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the linter.

    Args:
        argv: Command-line arguments, excluding the program name. Defaults to
            :data:`sys.argv`.

    Returns:
        :data:`EXIT_OK`, :data:`EXIT_FINDINGS`, or :data:`EXIT_ERROR`.
    """
    args = build_parser().parse_args(argv)
    paths = [Path(path) for path in args.paths]
    try:
        base = Config() if args.no_config else discover_config(paths)
        config = base.with_overrides(
            select=_split_codes(args.select),
            ignore=_split_codes(args.ignore),
        )
        report = analyze_paths(paths, config)
    except ConfigError as error:
        print(f"lookahead-lint: error: {error}", file=sys.stderr)
        return EXIT_ERROR
    output = render(report, args.output_format)
    if output:
        print(output, end="")
    return report.exit_code


def _split_codes(values: list[str] | None) -> list[str] | None:
    """Flatten repeated, comma-separated ``--select``/``--ignore`` values."""
    if values is None:
        return None
    return [code for value in values for code in value.split(",")]


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
