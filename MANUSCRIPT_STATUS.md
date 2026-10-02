# CropCop manuscript status — EAAI journal-extension release candidate

## Current research record

CropCop has two versioned paper states.

### arXiv v1 — public historical preprint

**CropCop: An Auditable 120-Class Plant-Health Model from Benchmark Reconstruction to a Quantised Runtime Artifact**  
Rana Muhammad Ahmed, Sabahat Abbas  
arXiv:2608.25539 · DOI: 10.48550/arXiv.2608.25539

This preprint reports the earlier internal benchmark/model-retention/software-runtime lineage. It does not establish physical Android or source-independent field generalization.

### EAAI journal extension — current manuscript

**Separating External-Source Behavior from Runtime Fidelity in Plant-Health Recognition**

Status: **submission-production stage**. Part 1 (main manuscript) and Part 2 (Supplement + reproducibility package) have both completed dedicated QA-enhancement passes and are scientifically cross-checked. Final Editorial Manager metadata, declarations and author-controlled archival decisions remain outside this repository release candidate.

The journal extension adds controlled four-family Track-A selection, prediction-blind external-cohort selection, three-state R07 external evaluation, exact-content sensitivity, exact R07-S1 reconstruction/replay, a single frozen PT2E/XNNPACK export route, Pixel physical execution, and POCO runtime-compatibility evidence.

## Scientific freeze

The journal manuscript's central upstream decision is R07 / ConvNeXt-Tiny. External and device outcomes do not reopen that selection. Track C is bound to the preselected `R07-CNXTT-CONTEXT-S1` state.

Key identities:

- R07-S1 checkpoint SHA-256: `dc7fea2e8db91bf1fc023cb5e792b23b67659edec22e10a7c1d46b4010db3974`
- class-map SHA-256: `46f7811726c19c42bd7213b2d8178b19a5a182a1b763f60a94ee2c0e5f6688d2`
- frozen manifest SHA-256: `bdb82211ccc2059153724eea178a1680893a6b38ecc243fae484baa91dbf68e2`
- FP32 PTE SHA-256: `61556330cd9fc4725b4ff4759aef3835696b81865b2cbe030f815a4c0e8d5aec`
- INT8 PTE SHA-256: `2e0c54a1b5bb7c0018d0159a642e49e4f4bd79fcc1f940e215a40734a2695bb8`
- accepted Track-B evidence ZIP SHA-256: `22a6c865ead6f28319a9dc8a4168f3ff7fb60108339710dcbd99e1aa047981a3`

## Public/restricted boundary

Public Git contains research code, protocols, locks, aggregate/derived evidence, runtime measurement tables, validators and paper-facing documentation. Ordinary public Git does not redistribute the consolidated source-image corpus, final model checkpoint, generated PTE binaries, raw logits or the complete accepted private Track-B evidence archive.

## Remaining author-controlled actions

Before actual EAAI submission/release:

1. approve final author order, affiliations, corresponding author, ORCIDs and CRediT roles;
2. approve funding, acknowledgments, competing-interest and AI-assistance declarations;
3. decide whether checkpoint/PTE redistribution is legally permitted;
4. decide whether the private Track-B evidence archive may be publicly/reviewer archived;
5. create the final immutable archival release/tag and optional Zenodo DOI after the paper-facing repository commit is frozen.

No further training or claim-producing model experiment is required for these administrative/release actions.
