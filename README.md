<div align="center">

# CropCop

### Evidence-separated evaluation for plant-health recognition, external-source transfer, and mobile runtime fidelity

[![arXiv v1](https://img.shields.io/badge/arXiv-2608.25539-B31B1B?logo=arxiv&logoColor=white)](https://arxiv.org/abs/2608.25539)
[![Validate Evidence](https://github.com/rana-m-ahmed/ResearchWork-CropCop/actions/workflows/validate-artifacts.yml/badge.svg)](https://github.com/rana-m-ahmed/ResearchWork-CropCop/actions/workflows/validate-artifacts.yml)
[![ExecuTorch](https://img.shields.io/badge/runtime-ExecuTorch%201.3.1%20%2B%20XNNPACK-EE4C2C)](https://pytorch.org/executorch/)
[![License: mixed](https://img.shields.io/badge/license-MIT%20%2B%20CC%20BY%204.0-4c1)](LICENSES.md)

**EAAI journal extension in submission preparation**  
**109,107 frozen images · 120 operational classes · prediction-blind external evaluation · exact runtime-artifact lineage · two-device compatibility evidence**

[Paper status](#paper-lineage) · [Evidence chain](#journal-extension-evidence-chain) · [Results](#journal-extension-results) · [Reproducibility](#reproducibility) · [Repository map](#repository-map) · [Limitations](#scope-and-limitations)

</div>

---

## Paper lineage

CropCop now has two clearly separated research records.

### Public preprint (v1)

**CropCop: An Auditable 120-Class Plant-Health Model from Benchmark Reconstruction to a Quantised Runtime Artifact**  
Rana Muhammad Ahmed, Sabahat Abbas  
**arXiv:2608.25539** (2026)

The preprint establishes leakage-controlled internal recognition and software-runtime fidelity. It uses the historical DINOv3 → MobileNetV4 → ExecuTorch/PTE lineage and explicitly does **not** claim physical Android or source-independent field generalization.

### EAAI journal extension

**Separating External-Source Behavior from Runtime Fidelity in Plant-Health Recognition**

The journal extension does not silently replace the preprint. It adds a new, evidence-separated study design around the same reconstructed 120-class benchmark:

1. freeze a controlled candidate-family decision before downstream evidence is opened;
2. preserve all selected-family states for prediction-blind external evaluation;
3. bind the deployment branch to a preselected R07-S1 state;
4. distinguish external-source behavior from FP32→INT8 artifact transformation;
5. distinguish artifact transformation from physical-device execution;
6. isolate raw-input preprocessing effects;
7. test runtime–hardware compatibility on a second smartphone without changing the model artifact.

The journal extension is the current research state of this repository. The arXiv preprint remains a versioned historical record.

## Why this repository exists

A high benchmark score does not identify where a deployed classifier begins to fail. Failure can enter through benchmark leakage, candidate selection, acquisition-source shift, artifact conversion, raw-image preprocessing, device execution, or runtime assumptions about target hardware.

CropCop therefore treats **evidence lineage and failure attribution** as first-class research objects. The repository is organized so a reviewer can trace a claim to the dataset/model/runtime identity that generated it without allowing downstream outcomes to rewrite the upstream scientific decision.

## Journal-extension evidence chain

\`\`\`mermaid
flowchart LR
    A[117,546 audited source images] --> B[109,107-image / 120-class frozen benchmark]
    B --> C[Track-A controlled candidate-family selection]
    C --> D[R07 ConvNeXt-Tiny selected family]
    D --> E[Track-B: S1/S2/S3 external-source evaluation]
    D --> F[Track-C: preselected R07-S1 runtime lineage]
    F --> G[FP32 host]
    G --> H[INT8 host: 252/256 vs FP32]
    G --> I[Pixel FP32: 256/256 vs FP32 host]
    H --> J[Pixel INT8: 256/256 vs INT8 host]
    H --> K[Raw-input path: 254/256 vs INT8 canonical]
    H --> L[POCO original runtime: SIGILL]
    H --> M[POCO compatible runtime: executes]
\`\`\`

The central methodological rule is **selection isolation**: external predictions and device outcomes cannot reopen the upstream family decision.

## Journal-extension results

### Benchmark and selection

| Item | Result |
| --- | ---: |
| Audited source images | 117,546 |
| Final frozen benchmark | 109,107 |
| Operational classes | 120 |
| Train / validation / historical test | 76,376 / 16,368 / 16,363 |
| Corrected trusted duplicate relations | 8,573 |
| Direct-cover duplicate rows removed | 8,355 |
| Final trusted leakage-group crossings | 0 |
| Selected family | R07 / ConvNeXt-Tiny |
| R07 mean validation macro-F1 | 0.96680861 |

The frozen selector independently reproduces the R07 family decision from four candidate systems under mean/worst-state macro-F1, class-tail performance, corruption degradation, and model-state size constraints.

### Prediction-blind external evaluation

The selected R07 family is evaluated unchanged in three fixed states on two public external cohorts while retaining the native 120-way output space.

| Cohort | Valid rows | Family mean macro-F1 | OOS rate | One-per-SHA macro-F1 |
| --- | ---: | ---: | ---: | ---: |
| GVLiD v5 | 3,477 | 0.3315 ± 0.0162 | 0.3637 | 0.2507 |
| Irish Potato v01 | 58,705 | 0.4160 ± 0.0326 | 0.6582 | 0.3646 |

\`OOS\` means **out of mapped scope** under native 120-way inference; it is not an open-set-recognition score. Exact-content deduplication is a composition-sensitivity analysis, not the uniquely “true” performance.

### Research-to-runtime evidence

The deployment branch is bound to **R07-CNXTT-CONTEXT-S1**, selected epoch 27.

| Evidence layer | Comparison | Result |
| --- | --- | ---: |
| Validation replay | checkpoint vs frozen validation metrics | all within 1e-6 |
| Artifact transformation | FP32 host vs INT8 host | 252/256 |
| FP32 device execution | FP32 host vs Pixel FP32 | 256/256 |
| INT8 device execution | INT8 host vs Pixel INT8 | 256/256 |
| Raw-input processing | raw INT8 vs paired canonical INT8 | 254/256 |
| Original POCO runtime | same INT8 artifact + locked inputs | SIGILL on first forward |
| Compatible POCO runtime | same artifact + inputs, separately versioned runtime | executes; canonical 255/256; raw 253/256 |

The four FP32→INT8 top-1 changes occur at canonical indices **11, 27, 155, and 255**. Pixel execution adds no additional canonical disagreement relative to either artifact's own host reference. The raw-input path differs at **11 and 255**.

The original POCO result is a runtime–hardware compatibility failure, not a model failure. The compatible-runtime campaign changes the runtime configuration while keeping the model artifact and locked inputs fixed; it is **not** a Pixel-versus-POCO speed ranking.

## Artifact identities

| Object | Identity |
| --- | --- |
| R07-S1 checkpoint | SHA-256 \`dc7fea2e8db91bf1fc023cb5e792b23b67659edec22e10a7c1d46b4010db3974\` |
| Frozen class map | SHA-256 \`46f7811726c19c42bd7213b2d8178b19a5a182a1b763f60a94ee2c0e5f6688d2\` |
| Frozen manifest | SHA-256 \`bdb82211ccc2059153724eea178a1680893a6b38ecc243fae484baa91dbf68e2\` |
| FP32 ExecuTorch artifact | 111,741,536 B · SHA-256 \`61556330cd9fc4725b4ff4759aef3835696b81865b2cbe030f815a4c0e8d5aec\` |
| INT8 ExecuTorch artifact | 28,555,872 B · SHA-256 \`2e0c54a1b5bb7c0018d0159a642e49e4f4bd79fcc1f940e215a40734a2695bb8\` |
| Accepted Track-B evidence ZIP | SHA-256 \`22a6c865ead6f28319a9dc8a4168f3ff7fb60108339710dcbd99e1aa047981a3\` |

The checkpoint and PTE binaries are identity-bound but are not distributed in ordinary public Git history pending redistribution review.

## Reproducibility

### Public verification

\`\`\`bash
git clone https://github.com/rana-m-ahmed/ResearchWork-CropCop.git
cd ResearchWork-CropCop
python scripts/validate_repository.py --strict
python -m pytest -q
\`\`\`

For the journal extension, see:

- [\`paper/eaai/README.md\`](paper/eaai/README.md) — paper-facing release map;
- [\`docs/REPRODUCIBILITY.md\`](docs/REPRODUCIBILITY.md) — current reproduction routes;
- [\`docs/EVIDENCE_BOUNDARIES.md\`](docs/EVIDENCE_BOUNDARIES.md) — public/restricted and claim boundaries;
- [\`journal_extension/\`](journal_extension/) — Track-A/B/C protocols, locks, scripts, evidence, and runtime measurements.

### What is reproducible without restricted research bytes

- benchmark/source accounting and ontology identities;
- candidate-family selector reproduction;
- external-cohort mappings and aggregate/state-level result checks;
- deterministic calibration-selection algorithm;
- checkpoint/class-map/manifest lineage verification;
- FP32→INT8 canonical top-1 transformation from retained host references;
- Pixel host/device and raw-input fidelity checks;
- POCO failure/compatibility evidence validation.

### What requires restricted or redistribution-controlled artifacts

- full benchmark-image reruns;
- exact checkpoint replay without authorized checkpoint access;
- regeneration of produced PTE bytes when checkpoint access is unavailable;
- the complete row-level Track-B evidence archive;
- raw logits and private forensic bundles.

The repository publishes hashes and fail-closed verification routes for these objects rather than pretending that local preservation equals public availability.

## Repository map

| Path | Purpose |
| --- | --- |
| [\`data_card/\`](data_card/) | Frozen benchmark identity, provenance, ontology, audit history, and distribution boundaries |
| [\`journal_extension/\`](journal_extension/) | Current EAAI Track-A/B/C scientific protocols, locks, scripts, evidence, and physical-device results |
| [\`evidence/public/\`](evidence/public/) | Public claim/evidence records and benchmark-derived evidence |
| [\`evidence/restricted/\`](evidence/restricted/) | Documentation for artifacts deliberately excluded from public Git |
| [\`paper/eaai/\`](paper/eaai/) | Current journal-extension status and paper-facing release metadata |
| [\`paper/\`](paper/) | Paper-version boundary, including the public preprint lineage |
| [\`docs/\`](docs/) | Reproducibility, evidence boundaries, limitations, intended use, and release policy |
| [\`scripts/\`](scripts/) | Repository-contract and evidence-validation utilities |
| [\`tests/\`](tests/) | Automated research/repository-contract tests |

## Scope and limitations

> [!IMPORTANT]
> CropCop is a research classifier and evaluation study, not an autonomous agronomic diagnostic or treatment system.

The EAAI journal extension establishes bounded evidence about a reconstructed benchmark, controlled candidate selection, source/task transfer on two prespecified external cohorts, deterministic artifact transformation, one identified Pixel 7 execution configuration, and a POCO M3 runtime-compatibility counterexample.

It does **not** establish universal field generalization, open-set recognition, production readiness, Android-wide equivalence, universal real-time behavior, energy efficiency, a Pixel-versus-POCO performance ranking, or causal architecture-only superiority.

## Citation

For the publicly archived preprint, cite:

\`\`\`bibtex
@misc{ahmed2026cropcop,
  author        = {Rana Muhammad Ahmed and Sabahat Abbas},
  title         = {CropCop: An Auditable 120-Class Plant-Health Model from Benchmark Reconstruction to a Quantised Runtime Artifact},
  year          = {2026},
  eprint        = {2608.25539},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CV},
  doi           = {10.48550/arXiv.2608.25539}
}
\`\`\`

The EAAI journal extension should be cited only after a public identifier is assigned to that version. Repository citation metadata intentionally keeps these paper versions distinct.

## Authors and journal metadata

The public arXiv v1 authors are **Rana Muhammad Ahmed** and **Sabahat Abbas**. Final EAAI author order, affiliations, corresponding-author metadata, ORCIDs, CRediT roles, funding, acknowledgments, competing interests, and final AI-assistance declaration remain author-controlled until explicitly approved.

## Licence and redistribution

- code, tests and validation scripts: **MIT**;
- original documentation, manuscript source, diagrams and derived tables: **CC BY 4.0**;
- third-party data/model assets: governed by upstream terms;
- consolidated images, checkpoints, PTE binaries and private evidence bundles: **not redistributed unless explicitly cleared**.

See [\`LICENSES.md\`](LICENSES.md).

---

<div align="center">

**Freeze the scientific decision first. Then measure where the evidence changes.**

</div>
