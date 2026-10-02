# Paper evidence authority — EAAI journal extension

This page is the shortest route from the current journal manuscript to the repository evidence that supports it. Historical development material remains versioned in Git, but it should not be mistaken for current paper authority.

## Current paper

**Separating External-Source Behavior from Runtime Fidelity in Plant-Health Recognition**

The public arXiv v1 preprint (`2608.25539`) is a distinct historical paper version. The EAAI journal extension adds controlled candidate-family selection, prediction-blind external evaluation and physical-device/runtime compatibility evidence.

## Authority hierarchy

| Layer | Current authority | Purpose |
|---|---|---|
| Benchmark identity | `data_card/`, `evidence/public/dataset_lineage/` | 109,107-image / 120-class frozen benchmark, provenance and audit boundary |
| Track A | `journal_extension/amendments/track_a_strengthening_v1/`, `journal_extension/evidence/public/track_a/` | candidate pool, chronology, selector and auxiliary analyses |
| Track B | `journal_extension/track_b_r07/` | prediction-blind external-source authority, mappings, protocol and results lock |
| Track C foundation | `journal_extension/track_c_r07/shared_foundation/` | exact R07-S1 state, reconstruction and validation replay |
| Track C runtime | `journal_extension/track_c_r07/results/public/`, `journal_extension/track_c_r07/reports/` | Pixel fidelity/timing and POCO blocker/compatibility evidence |
| Paper-facing status | `paper/eaai/` | Supplement/reproducibility release map and availability boundary |
| Claim boundary | `docs/EVIDENCE_BOUNDARIES.md` | what is established, bounded and not established |
| Reproduction guide | `docs/REPRODUCIBILITY.md` | how to verify public evidence and where restricted bytes are required |

## Locked identities

- selected family: **R07 / ConvNeXt-Tiny**
- deployment state: **R07-CNXTT-CONTEXT-S1**
- checkpoint SHA-256: `dc7fea2e8db91bf1fc023cb5e792b23b67659edec22e10a7c1d46b4010db3974`
- class map SHA-256: `46f7811726c19c42bd7213b2d8178b19a5a182a1b763f60a94ee2c0e5f6688d2`
- frozen manifest SHA-256: `bdb82211ccc2059153724eea178a1680893a6b38ecc243fae484baa91dbf68e2`
- FP32 PTE SHA-256: `61556330cd9fc4725b4ff4759aef3835696b81865b2cbe030f815a4c0e8d5aec`
- INT8 PTE SHA-256: `2e0c54a1b5bb7c0018d0159a642e49e4f4bd79fcc1f940e215a40734a2695bb8`
- accepted Track-B evidence ZIP SHA-256: `22a6c865ead6f28319a9dc8a4168f3ff7fb60108339710dcbd99e1aa047981a3`

## Interpretation guardrails

The journal evidence should be read as separate layers:

1. **selection isolation** — downstream evidence does not reopen the selected family;
2. **external-source behavior** — source/task transfer of the fixed selected family;
3. **artifact transformation** — FP32 host versus INT8 host;
4. **device execution** — each artifact versus its own host reference;
5. **raw-input processing** — raw input versus paired canonical input;
6. **runtime–hardware compatibility** — executability under a specified runtime/device configuration.

Do not collapse these into one “deployment accuracy” claim.

## Restricted evidence

Ordinary public Git does not redistribute the consolidated benchmark corpus, R07-S1 checkpoint, produced PTE binaries, raw logits or complete private Track-B row-level archive. Hash identities and verification routes are provided so authorized reviewers can verify exact bytes when those objects are made available.

## Release rule

For a journal submission or archival citation, prefer an immutable release/tag created from the final journal-facing commit over a moving branch name.
