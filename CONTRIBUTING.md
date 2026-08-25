# Contributing

Contributions are welcome when they improve reproducibility, correct provenance, or clarify the evidence boundary.

1. Open an issue before changing headline metrics, class order, artifact hashes, or scientific claims.
2. Branch from `main` and keep commits narrowly scoped.
3. Run `python scripts/validate_repository.py --strict`.
4. Build the paper with `make paper` when LaTeX or bibliography files change.
5. Do not commit raw images, full evidence bundles, checkpoints, logits, credentials, or local absolute paths.
6. A claim correction must identify the affected evidence file, paper location, and expected downstream changes.

Metric changes require a synchronized update to the registry, claim matrix, affected CSVs, manuscript, validation tests, and release manifest.
