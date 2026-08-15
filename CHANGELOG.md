# Changelog

## [Unreleased]

- Combined the seven analyzer rule walks into one iterative AST traversal while preserving finding order, suppression, and scope semantics.
- Added a deterministic end-to-end analysis and text-render benchmark with exact output checksums.


## [1.0.0] - 2026-08-12

First stable release.

- Static analyzer that flags look-ahead bias and target leakage idioms in quantitative research code.
- Added deterministic property-based tests covering clean and leaky programs, analyzer purity, and all report formats.
- Killed 1,109 of 1,122 generated mutants (98.84%); reviewed the 13 survivors as runtime-equivalent.
- Adopted strict mypy checking.
- Expanded CI to Linux and macOS across Python 3.11–3.13.
