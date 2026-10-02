# CropCop × EAAI journal extension

## Manuscript

**Separating External-Source Behavior from Runtime Fidelity in Plant-Health Recognition**

This directory is the paper-facing bridge between the EAAI manuscript/Supplement release and the canonical research evidence under \`journal_extension/\`.

## Scientific freeze

- final benchmark: 109,107 images / 120 classes;
- selected family: R07 / ConvNeXt-Tiny;
- Track-B external states: R07 S1/S2/S3;
- Track-C deployment representative: \`R07-CNXTT-CONTEXT-S1\`;
- exact checkpoint, class-map, manifest, FP32 and INT8 artifact identities;
- native 120-way external inference with seven mapped labels;
- layered runtime interpretation: artifact transformation ≠ device execution ≠ raw-input processing ≠ runtime–hardware compatibility.

## Part 2

The production Part-2 package contains:

- standalone Supplement S1–S13 PDF;
- clean Overleaf/LaTeX source;
- machine-readable Supplement source data;
- reproducibility release with identity/replay, quantization/artifact and Pixel/POCO evidence;
- claim/evidence and Part-1↔Part-2 consistency audits;
- release/licensing boundaries.

The complete generated publication bundle is distributed as the reviewed Part-2 handoff rather than committing generated PDFs/ZIPs or restricted evidence into ordinary Git history.

## Canonical evidence

See [\`source_data/README.md\`](source_data/README.md) for the paper-facing map from Supplement sections to canonical repository evidence.

## Availability boundary

The complete private Track-B evidence archive, consolidated benchmark images, model checkpoint, PTE binaries and raw logits are not stored in ordinary public Git history. Their cryptographic identities and authorized verification routes are documented instead.

## Status

See [\`PART2_RELEASE_STATUS.md\`](PART2_RELEASE_STATUS.md) and the repository-level [\`MANUSCRIPT_STATUS.md\`](../../MANUSCRIPT_STATUS.md).
