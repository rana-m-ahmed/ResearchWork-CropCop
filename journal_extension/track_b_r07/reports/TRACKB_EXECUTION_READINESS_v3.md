# Track B R07 — Execution Readiness v3

**Current gate:** **IMPLEMENTATION PASS / HOLD FOR REAL KAGGLE QUALIFICATION**

The remediation implementation has passed exact-head repository QA at `64f6202dfab9d4972c14557f6506311ef5b4b614`. This report does **not** claim that Track B is closed or that protected external inference has run.

## Current authority

- Downstream authority: `EAAI-JE-TRACKBC-R07-DOWNSTREAM-v3`
- Execution lock: `TRACKB_R07_EXECUTION_LOCK_v3`
- Code attestation: `TRACKB_CODE_ATTESTATION_v3`
- External lineage review: `TRACKB_EXTERNAL_LINEAGE_REVIEW_v1`
- Track-A closure parent: `604aafd51e20e70098ce4af647e90c8ff558a9e8`

The old `TRACKB_EXECUTION_READINESS_v2.*` files are retained only as historical records and are superseded for current operation.

## Frozen science

Track B remains exactly:

- GVLiD v5 grape-4;
- Irish Potato Version 01 potato-3;
- R07 S1/S2/S3;
- native 120-way predictions;
- no mapped-subset renormalization;
- 50-family minimum support per mapped class;
- 5,000 family-bootstrap replicates, seed `409883112`;
- deterministic external-family order seed `1936263114`;
- historical comparison over V1 train + validation only (92,744);
- maximum executable evidence grade `EXT-S`;
- no new training or adaptation;
- consumed V1 test closed.

## Supported Kaggle operator

Use only `journal_extension/kaggle/trackb_r07_master.ipynb`.

Required setup:

- GPU T4 x2;
- Internet ON;
- `KAGGLE_API_TOKEN`;
- `CROPCOP_GITHUB_TOKEN`;
- no manual Track-B dataset attachments;
- no manual pip installation.

Persistent evidence uses `/kaggle/working/trackb_master`; heavy/raw/transient material uses `/kaggle/tmp/cropcop_trackb_r07`.

## Repository QA

All relevant workflows passed at the audited implementation code head:

- Track B Infrastructure QA PR — **35492138674**
- Track B Infrastructure QA push — **35492136557**
- Validate secondary Track-A wave — **35492138642**
- Validate public evidence / complete CPU-safe suite — **35492138673**

## What remains

Protected external inference is **not yet authorized**.

Required next sequence:

1. **Q1** real Kaggle platform/credential/source/publication smoke.
2. **Q2** complete real prediction-blind candidate acquisition + duplicate/historical-overlap audit + seals, stopping before R07 external inference.
3. **Q3** independent verification of Q2.
4. Freeze the final exact execution head/notebook.
5. Run one clean protected claim execution.

Until Q3 passes, `TRACK_B_CLOSED` and manuscript result use remain unauthorized.
