"""Deterministic end-to-end benchmark for source analysis and text rendering."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import statistics
import time
from pathlib import Path

from lookahead_lint.analyzer import Report, analyze_source
from lookahead_lint.reporter import render

MODULE_COUNT = 160
EXPECTED_FINDINGS = MODULE_COUNT * 9
EXPECTED_SHA256 = "b19761066efde446a2a16b53dea978ad74c2138c75dc84d974296d86064f474b"
PATH = Path("research/portfolio_research.py")


def build_source(module_count: int = MODULE_COUNT) -> str:
    """Build a realistic multi-sleeve research module outside the timed region."""
    blocks = [
        "from pandas import merge_asof\nfrom sklearn.model_selection import train_test_split\n\n"
    ]
    for sleeve in range(module_count):
        blocks.append(
            f"def research_sleeve_{sleeve}(frame, quotes, model):\n"
            "    label = frame.close.shift(-1)\n"
            "    intentional_label = frame.close.shift(-2)  # lookahead-lint: ignore[LA001]\n"
            "    centered = frame.close.rolling(21, center=True).mean()\n"
            "    prices = frame.price.bfill()\n"
            "    volume = frame.volume.fillna(method='backfill')\n"
            "    signal = frame.signal.interpolate(limit_direction='both')\n"
            "    joined = merge_asof(frame, quotes, direction='forward')\n"
            "    future = [prices[i + 1] for i in range(len(prices) - 1)]\n"
            "    scaled = model.fit_transform(frame)\n"
            "    train, test = train_test_split(frame)\n"
            "    return label, intentional_label, centered, volume, signal, "
            "joined, future, scaled\n\n"
        )
    return "".join(blocks)


def analyze_and_render(source: str) -> tuple[Report, str]:
    """Exercise parsing, all rules, finding materialization, and default rendering."""
    findings, errors = analyze_source(source, PATH)
    report = Report(tuple(findings), tuple(errors), files_checked=1)
    return report, render(report, "text")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=15)
    parser.add_argument("--warmups", type=int, default=3)
    args = parser.parse_args()
    if args.samples < 11:
        parser.error("--samples must be at least 11")

    source = build_source()
    reference_report, reference_render = analyze_and_render(source)
    if reference_report.errors:
        raise RuntimeError(f"benchmark source produced parse errors: {reference_report.errors}")
    if len(reference_report.findings) != EXPECTED_FINDINGS:
        raise RuntimeError(
            f"expected {EXPECTED_FINDINGS} findings, got {len(reference_report.findings)}"
        )
    digest = hashlib.sha256(reference_render.encode()).hexdigest()
    if EXPECTED_SHA256 and digest != EXPECTED_SHA256:
        raise RuntimeError(f"expected render digest {EXPECTED_SHA256}, got {digest}")

    for _ in range(args.warmups):
        report, rendered = analyze_and_render(source)
        if report != reference_report or rendered != reference_render:
            raise RuntimeError("analysis result changed during warmup")

    samples: list[float] = []
    gc.disable()
    try:
        for _ in range(args.samples):
            started = time.perf_counter_ns()
            report, rendered = analyze_and_render(source)
            elapsed = time.perf_counter_ns() - started
            if report != reference_report or rendered != reference_render:
                raise RuntimeError("analysis result changed between timed samples")
            samples.append(elapsed / 1_000_000)
    finally:
        gc.enable()

    print(
        json.dumps(
            {
                "benchmark": "analyze_source+render_text",
                "digest_sha256": digest,
                "findings": len(reference_report.findings),
                "median_ms": statistics.median(samples),
                "min_ms": min(samples),
                "max_ms": max(samples),
                "samples": len(samples),
                "source_bytes": len(source.encode()),
                "source_lines": source.count("\n"),
                "sleeves": MODULE_COUNT,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
