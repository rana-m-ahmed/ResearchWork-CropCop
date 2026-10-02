# CropCop × EAAI journal extension

## Manuscript

**Separating External-Source Behavior from Runtime Fidelity in Plant-Health Recognition**

This directory is the paper-facing bridge between the EAAI manuscript/Supplement release and the canonical research evidence under `journal_extension/`.

> **Start here for evidence navigation:** [`docs/PAPER_EVIDENCE_AUTHORITY.md`](../../docs/PAPER_EVIDENCE_AUTHORITY.md)

## Scientific freeze

- final benchmark: **109,107 images / 120 classes**;
- selected family: **R07 / ConvNeXt-Tiny**;
- Track-B external states: **R07 S1/S2/S3**;
- Track-C deployment representative: **R07-CNXTT-CONTEXT-S1**;
- native 120-way external inference with seven mapped labels;
- exact checkpoint, class-map, manifest, FP32 and INT8 artifact identities;
- layered interpretation: **artifact transformation ≠ device execution ≠ raw-input processing ≠ runtime–hardware compatibility**.

## Part 1

The QA-enhanced Part-1 delivery contains:

- the 19-page EAAI main manuscript;
- clean Overleaf and flat Editorial Manager LaTeX source;
- exact approved Figures 1–4;
- scientific-regression, reference, typography, rendered-PDF and build-reproduction audits.

See [`PART1_RELEASE_STATUS.md`](PART1_RELEASE_STATUS.md) for the manuscript production state.

## Part 2

The QA-enhanced Part-2 delivery contains:

- readable, standalone Supplement S1–S13 PDF;
- clean Overleaf/LaTeX source;
- exhaustive machine-readable Supplement source data;
- reproducibility release with identity/replay, quantization/artifact and Pixel/POCO evidence;
- refreshed claim/evidence and Part-1↔Part-2 consistency audits;
- explicit public/restricted release boundaries.

The Supplement intentionally keeps exhaustive inventories in machine-readable companion files rather than shrinking them into unreadable PDF tables.

Generated PDFs/ZIPs and redistribution-controlled evidence are distributed through the reviewed handoff/archive rather than committed into ordinary Git history.

## Canonical evidence

See:

- [`../../docs/PAPER_EVIDENCE_AUTHORITY.md`](../../docs/PAPER_EVIDENCE_AUTHORITY.md) — current paper-to-evidence authority map;
- [`source_data/README.md`](source_data/README.md) — Supplement section-to-evidence map;
- [`../../docs/REPRODUCIBILITY.md`](../../docs/REPRODUCIBILITY.md) — reproduction routes;
- [`../../docs/EVIDENCE_BOUNDARIES.md`](../../docs/EVIDENCE_BOUNDARIES.md) — claim boundaries.

## Availability boundary

The complete private Track-B evidence archive, consolidated benchmark images, model checkpoint, PTE binaries and raw logits are not stored in ordinary public Git history. Their cryptographic identities and authorized verification routes are documented instead.

## Status

See [`PART1_RELEASE_STATUS.md`](PART1_RELEASE_STATUS.md), [`PART2_RELEASE_STATUS.md`](PART2_RELEASE_STATUS.md), and the repository-level [`MANUSCRIPT_STATUS.md`](../../MANUSCRIPT_STATUS.md).
