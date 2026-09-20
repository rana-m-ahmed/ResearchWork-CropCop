# CropCop Track B — R07 External Validation

## Current operator architecture: v4

Track B keeps the frozen v3 scientific design and replaces only the operator/orchestration layer.

**Supported operator notebooks:**

1. `journal_extension/kaggle/TrackB_00_Readiness_Materialization.ipynb`
2. `journal_extension/kaggle/TrackB_01_Final_Execution.ipynb`

All older Track-B notebooks are historical/recovery surfaces and are **not supported for new v4 execution**.

## Scientific boundary

The v4 orchestration makes **no scientific change** to:

- downstream authority `EAAI-JE-TRACKBC-R07-DOWNSTREAM-v3`;
- R07 ConvNeXt-Tiny S1/S2/S3;
- native 120-way predictions;
- CTC-v2 preprocessing;
- GVLiD v5 and Irish Potato Version 01;
- frozen source→CropCop mappings;
- V1 train+validation historical comparison only (92,744 images);
- EXT-S ceiling for the executable post-closure historical route;
- pHash/dHash/DINO/ORB audit thresholds;
- 50-family minimum support;
- 5,000 bootstrap replicates, seed `409883112`;
- external-family order seed `1936263114`;
- consumed V1 test closure;
- no training/adaptation/reselection.

The scientific runner remains `journal_extension/scripts/run_trackb_r07.py`.

## Notebook 00 — Readiness & Materialization

Use Kaggle T4 x2 with Internet ON and secret `KAGGLE_API_TOKEN`.

Attach exactly the five frozen internal source datasets listed in
`TRACKB_INPUT_MATERIALIZATION_LOCK_v1.json`.

The readiness notebook:

1. checks out the exact frozen v4 orchestration source;
2. repairs/verifies the exact Track-B runtime;
3. reads the five internal datasets from attached `/kaggle/input` mounts rather than downloading them;
4. builds the immutable core package;
5. builds the safe 92,744-image historical comparison package;
6. acquires and verifies the exact public GVLiD v5 and Irish Potato Version 01 cohorts;
7. prepares the two candidate packages;
8. creates a common `materialization_id` binding all four role manifests, repository source, v3 execution lock and v3 scientific attestation;
9. publishes exactly two private Kaggle datasets:
   - `cropcop-trackb-r07-infrastructure-v4`
   - `cropcop-trackb-r07-external-v4`
10. performs full private-Kaggle round-trip content verification.

Readiness is prediction-blind and must finish with
`PASS_TRACKB_INPUT_MATERIALIZATION`, zero protected external predictions, and no V1-test access.

Heavy one-time materialization lives under `/kaggle/tmp`; only the readiness receipt is retained in `/kaggle/working`.

## Notebook 01 — Final Execution

Attach only the exact two private dataset versions produced by Notebook 00.

The final notebook performs no Mendeley/Zenodo acquisition, no Kaggle dataset downloading, and no Git clone. It executes from the repository snapshot embedded in the attached `core` package.

### Qualification mode

Default:

`RUN_MODE = 'qualification'`

Secrets required: **none**.

Qualification:

- validates the two paired bundle receipts;
- validates exactly four Track-B input roles;
- repairs/verifies the exact runtime;
- launches the scientific controller in a fresh subprocess;
- replays R07 S1/S2/S3 on the frozen V1 validation surface;
- runs both complete prediction-blind candidate audits;
- freezes grades, families and seals;
- writes `TRACKB_PREINFERENCE_QUALIFICATION.json`;
- runs independent pre-inference QA;
- asserts zero protected external predictions and V1-test access false;
- stops.

The key handoff is `qualification_science_sha256`.

### Claim mode

Only after independent review of qualification:

`RUN_MODE = 'claim'`

and set the exact reviewed
`AUTHORIZED_QUALIFICATION_SCIENCE_SHA256`.

Claim mode must use the exact same two attached dataset versions.

Secret required: `KAGGLE_API_TOKEN` only.

The runner recomputes the prediction-blind science identity and refuses protected inference unless it matches the reviewed digest exactly. Before the first protected R07 forward pass, it writes the durable attempt ledger. After terminal QA it builds the complete/public evidence packages and archives restricted evidence to private Kaggle with round-trip verification.

GitHub credentials are intentionally excluded from the scientific run.

## Publication boundary

GitHub is a post-closure dissemination layer, not part of the estimator.

After `TRACK_B_CLOSED`, use the public-safe evidence archive produced by Notebook 01 and commit it to:

`journal_extension/evidence/public/track_b/<science-sha-prefix>/`

through a reviewed repository branch/PR.

A GitHub publication failure must never trigger a scientific rerun.

## v4 operational locks

- `TRACKB_INPUT_MATERIALIZATION_LOCK_v1.json`
- `TRACKB_ORCHESTRATION_LOCK_v4.json`

These operational locks sit above the unchanged scientific:

- `TRACKB_R07_EXECUTION_LOCK_v3.json`
- `TRACKB_CODE_ATTESTATION_v3.json`
- `TRACKB_EXTERNAL_LINEAGE_REVIEW_v1.json`

## Current gate

Until Notebook 00 completes successfully on real Kaggle and the two published bundles are independently verified:

**HOLD — REAL MATERIALIZATION REQUIRED**

Until Notebook 01 qualification passes and its `qualification_science_sha256` is reviewed:

**HOLD — PROTECTED EXTERNAL INFERENCE FORBIDDEN**
