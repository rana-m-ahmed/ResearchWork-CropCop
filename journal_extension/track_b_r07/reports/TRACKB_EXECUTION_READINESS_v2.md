# Track B R07 — Execution Readiness v2

**Gate:** `PASS — TRACK B IMPLEMENTATION + OPERATOR QA COMPLETE; INPUT MATERIALIZATION MAY BEGIN`

**Important:** this is an execution-readiness gate, not a Track-B result. No protected external R07 inference has been run, no acquired external candidate has been sealed yet, and `TRACK_B_CLOSED` has not been issued.

## Audited implementation state

The audited implementation head is `1a2c91e0d5e7e20d3e8f252a92bbe7ee64cb3c9b` on `trackb-r07-infrastructure-20260919`, rooted directly in formal Track-A closure `604aafd51e20e70098ce4af647e90c8ff558a9e8`. The branch was 34 commits ahead and zero behind at the audited head. GitHub Actions run **35461390715** completed successfully at that exact head.

The clean CI gate passed:
- Python compilation;
- **19/19** targeted Track-B scientific/loophole tests;
- canonical regeneration of all Track-B notebooks with zero diff;
- the static infrastructure validator.

The active code attestation is `TRACKB_CODE_ATTESTATION_v1`, with SHA-256 `06f91abc1bfe430f21fcf1ed3fa77d6432eb0a7028227298a059fb17efa634b9`, covering **29 load-bearing files**.

## What is now locked

The implementation binds:
- all three R07 checkpoints;
- all three authoritative replay run records;
- the exact DINO audit checkpoint;
- the exact DINO factory manifest;
- frozen Final-V1 manifest and 120-way class map;
- CTC-v2 evaluation preprocessing;
- the downstream R07 authority amendment;
- fixed external mappings, audit thresholds, support floor, family representative rule, and bootstrap;
- native 120-way scoring with no mapped-subset logit renormalization;
- prediction-blind audit and candidate sealing before protected inference;
- no new training and no consumed V1-test access.

## Consumed-test hardening

The earlier raw 117,546-image historical-index path was found to be unsafe after V1-test closure because rebuilding it now could reopen image bytes that later became the consumed test.

That path is now fail-closed.

The executable historical-comparison route is:
- **76,376 train + 16,368 validation = 92,744 images**;
- coverage `V1_TRAIN_VAL_ONLY`;
- consumed test bytes accessed: **false**;
- maximum evidence grade: **EXT-S**.

The full 117,546-image `EXT-I` route is dormant. It may be activated only if a complete comparison representation created before V1-test closure is genuinely recovered, cryptographically verified, and formally rebound. Row count alone can never authorize it.

## Deterministic core operator

Use:

`journal_extension/kaggle/trackb_build_core_package.ipynb`

It is bound to the already-existing source datasets:
- Final V1: `ranamuhammadahmed6/cropcop-finalized-v8-11-2026-1`
- R07 S1: `sabahatabbas/sec-je-r07-cnxtt-context-s1-8904b100d223-a01`
- R07 S2: `sabahatabbas/cropcop-r07-cnxtt-context-s2-abce1197-56023042`
- R07 S3: `sabahatabbas/cropcop-r07-cnxtt-context-s3-f13ca687-56023042`
- DINO bundle: `ranamuhammadahmed6/cropcop-secondary-g1-8904b100`, **version 2**

The builder finds checkpoint/run-record/factory files by authoritative SHA-256 rather than trusting filenames and copies only the **16,368-image validation surface** into the final core package.

## External candidates

### Irish Potato

Frozen source: Zenodo Version 01, DOI `10.5281/zenodo.8286529`.

Expected originals:
- `earlyblt`: 17,772
- `healthy`: 20,438
- `lateblt`: 20,499
- total: **58,709**

The source identity/count contract is ready. The exact bytes still need to be acquired, provenance/license-access metadata archived, verified with `prepare_trackb_candidate_input.py`, and published as an immutable `role = irish_potato` package.

### Agri-Vision Bangladesh

Frozen source: Mendeley Data v2, DOI `10.17632/8t6k37ztxc.2`.

Only `Original_Images` is eligible; expected total is **5,266**. The frozen mapped classes are Tomato Healthy, Tomato Mosaic, and Papaya Healthy Leaf.

Before packaging, the prediction-blind `Tomato Mosaic -> tomato_mosaic_virus` semantic record must be frozen from source-semantic evidence. Failure of that gate yields `EXT-X`; no post-hoc remap is permitted.

## Exact execution order

1. Run the core-package builder and publish immutable `role = core`.
2. Run the safe historical-comparison builder and publish immutable `role = historical_compare`.
3. Acquire/verify/package Irish Potato Version 01.
4. Acquire/verify/package Agri-Vision v2 `Original_Images` and freeze its semantic record.
5. Attach exactly those four immutable roles to `trackb_r07_end_to_end.ipynb`.
6. Kaggle T4x2, scientific device `cuda:0`, one clean **Save & Run All**.
7. Accept protected inference only after B0 replay passes, both candidate audits are terminal, and eligible seals verify.
8. Independently audit `TRACKB_FINAL_QA.json` and `TRACKB_FINAL_CLOSURE.json` before manuscript use.

## Current verdict

The **repository is ready to begin Track-B execution** at input materialization.

It is **not yet scientifically valid to claim external performance**, because the immutable candidate inputs and safe historical-comparison package have not yet been materialized and the protected inference firewall has not yet opened.
