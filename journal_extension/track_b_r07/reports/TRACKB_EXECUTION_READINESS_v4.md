# Track B R07 — Execution Readiness v4

**Gate:** IMPLEMENTATION READY / HOLD FOR REAL KAGGLE MATERIALIZATION

## Scope

This v4 changes the Track-B operator/orchestration layer only. The scientific authority remains `EAAI-JE-TRACKBC-R07-DOWNSTREAM-v3` and the scientific runner remains unchanged.

## Supported notebooks

1. `journal_extension/kaggle/TrackB_00_Readiness_Materialization.ipynb`
2. `journal_extension/kaggle/TrackB_01_Final_Execution.ipynb`

All prior Track-B notebooks are historical/recovery surfaces and are not supported for new v4 execution.

## Frozen scientific invariants

Unchanged:

- R07 ConvNeXt-Tiny S1/S2/S3;
- GVLiD v5 + Irish Potato Version 01;
- native 120-way scoring;
- CTC-v2 preprocessing;
- V1 train+validation historical comparison only (92,744 images);
- EXT-S ceiling for the executable post-closure route;
- pHash/dHash/DINO/ORB thresholds;
- 50-family minimum support;
- 5,000 bootstrap replicates, seed 409883112;
- external-family order seed 1936263114;
- no new training/adaptation/reselection;
- consumed V1 test remains closed.

## v4 architecture

### Readiness notebook

Inputs are attached Kaggle datasets, not API-downloaded internal sources.

The readiness notebook creates two paired private Kaggle datasets:

- `cropcop-trackb-r07-infrastructure-v4`
- `cropcop-trackb-r07-external-v4`

Both are bound by the same `materialization_id`, repository source SHA, scientific execution-lock identity, code-attestation identity, and four role-manifest hashes.

Readiness is prediction-blind and must finish with:

- `PASS_TRACKB_INPUT_MATERIALIZATION`;
- protected external prediction count = 0;
- V1-test access = false;
- full private-Kaggle content round-trip verification.

### Final execution notebook

Attach only the exact two dataset versions produced by readiness.

Qualification:
- no secrets;
- no Git clone;
- no Mendeley/Zenodo acquisition;
- no Kaggle input downloading;
- exact paired-bundle validation;
- independent Q3 verification;
- zero protected external predictions.

Claim:
- same exact two attached versions;
- exact reviewed `qualification_science_sha256`;
- `KAGGLE_API_TOKEN` only;
- durable attempt ledger before first protected forward pass;
- private restricted-evidence round-trip verification;
- GitHub excluded from the scientific critical path.

## Frozen v4 source

The notebooks execute against embedded/orchestration snapshot:

`066584af76704272c4ce05d0cf6795388b350f94`

The load-bearing orchestration scripts were introduced at:

`066584af76704272c4ce05d0cf6795388b350f94`

## Remaining real-world gates

1. Run Notebook 00 on Kaggle and obtain `PASS_TRACKB_INPUT_MATERIALIZATION`.
2. Verify the two published private dataset versions and their shared materialization ID.
3. Run Notebook 01 in qualification mode.
4. Review `qualification_science_sha256`, candidate seals, grades, support, historical-link evidence, and Q3 receipt.
5. Only then run Notebook 01 in claim mode.
6. After `TRACK_B_CLOSED`, commit public-safe evidence to the repository through a reviewed branch/PR.

Until steps 1–4 pass, protected external inference remains forbidden.

## DINO factory source-root remediation

A real Notebook-00 run passed source qualification, immutable-core construction, and entered the 92,744-image historical-comparison stage before the DINO loader failed with `teacher factory source file missing: historical_dino_tiny.py`. The sealed teacher factory manifest stores `historical_dino_tiny.py` relative to its dedicated source root, while the Track-B core package had incorrectly recorded the embedded repository root as `dino_factory_source_root`.

Remediation:

- core packaging now records `repository/journal_extension/teacher_factory` as the DINO factory source root;
- Stage 0.5 validates the sealed factory bundle against that exact root;
- Stage 0.5 instantiates the exact DINO teacher checkpoint/factory before any expensive historical build;
- regression tests prove repository-root validation fails and the dedicated factory root passes;
- the v3 code attestation/execution lock and v4 locks were rebound to the repaired builder.

This is an operational path-binding correction only. DINO checkpoint SHA, factory-manifest SHA, model architecture, historical surface, audit thresholds, and scientific evidence policy are unchanged.
