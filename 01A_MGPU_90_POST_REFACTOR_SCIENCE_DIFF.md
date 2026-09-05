# Stage 01A-MGPU — Post-Refactor Science-Diff

- **Artifact:** `01A_MGPU_90_POST_REFACTOR_SCIENCE_DIFF.md`
- **Post-refactor execution-code candidate:** `3dfed1074b89648eaee435567a18eddcf5a08a05`
- **Reference sentinel:** `01A_MGPU_01_SCIENCE_DIFF_SENTINEL.json`
- **Exact-head CI:** Actions run #77 / `33975850338`
- **CI result:** PASS
- **MGPU contract tests:** 66 / 66 PASS
- **Complete CPU-safe suite:** 193 / 193 PASS
- **Verdict:** **PASS — NO SCIENTIFIC DRIFT**

## 1. Scientific authority

| Object | Pre-refactor | Post-refactor | Result |
|---|---|---|---|
| Stage-03R authority ID | `EAAI-JE-SDL-v2.1-QA` | same | PASS |
| Stage-03R SHA-256 | `aab17b65b0873dcb1ecedb061eb02ff60ccb09f8b830184f5e2231a600278f74` | same | PASS |
| Stage-04 base ID | `EAAI-JE-REA-v2.2-LEAN` | same | PASS |
| Stage-04 base SHA-256 | `a1bc1f8bb2d7a133a46c3f59f83dd2ba8ca3bf0fa8d77ae323553276931bc079` | same | PASS |

The additive execution amendment is separately bound as:

- ID: `EAAI-JE-MGPU-A1`
- SHA-256: `3f08f2dbe7e83143c7e6f1fcf6732c7d88086295a600a74b323b89b46caed3f6`

It does not replace or alter Stage-03R or the Stage-04 base hash.

## 2. Protected Git blobs

The executable science-diff sentinel compared the current exact repository bytes against the pre-refactor frozen Git blob identities.

| Protected surface | Pre-refactor Git blob | Post-refactor Git blob | Result |
|---|---|---|---|
| `journal_extension/src/cropcop_je/train.py` | `a259f7c825942e7f3031b20e29f33fc55f1d9d7f` | `a259f7c825942e7f3031b20e29f33fc55f1d9d7f` | BYTE-IDENTICAL |
| `journal_extension/src/cropcop_je/data.py` | `9871ece382e0345bcab2e832d3116611855c3de7` | `9871ece382e0345bcab2e832d3116611855c3de7` | BYTE-IDENTICAL |
| `journal_extension/src/cropcop_je/models.py` | `c7db1155de1621e751c4bdbad7e9c4096ab06746` | `c7db1155de1621e751c4bdbad7e9c4096ab06746` | BYTE-IDENTICAL |
| `journal_extension/configs/common/ctc_v2.json` | `517d6fcc38e3a8e480fb1edcd97bac87907c5b5c` | `517d6fcc38e3a8e480fb1edcd97bac87907c5b5c` | BYTE-IDENTICAL |
| R04 S1 | `4f6e985c08210f425b0c651c8bc10e61ad3db5f8` | same | BYTE-IDENTICAL |
| R04 S2 | `9d6f85ac6656588397ef96e76e694cf09e2203d1` | same | BYTE-IDENTICAL |
| R04 S3 | `d08f78619c22ae47115865bed2da52ac67c74625` | same | BYTE-IDENTICAL |
| R05 S1 | `2607f803d7d0ee9ded564dd52f9d91d4d0ef35a3` | same | BYTE-IDENTICAL |
| R05 S2 | `79cb7d5ecbd4674a7ce70718dae07df5de2a1b9b` | same | BYTE-IDENTICAL |
| R05 S3 | `6df147239c652d9c64368208e240b63c9ad5f61f` | same | BYTE-IDENTICAL |
| experiment registry | `ac83f61eba78a24e0a40e387a817a27bcdef2bed` | same | BYTE-IDENTICAL |

Therefore `train.py`, `data.py`, `models.py`, CTC-v2, all six R04/R05 configs, and the scientific experiment registry received **zero byte changes**.

## 3. Locked training semantics

The executable sentinel revalidated:

```text
micro_batch_size      = 16
gradient_accumulation = 4
effective_batch_size  = 64
epochs                = 30
optimizer              = AdamW
mixed precision        = FP16 autocast + GradScaler
gradient_clip_norm     = 1.0
EMA                    = off
drop_last              = false
sampling               = uniform row shuffle
early stopping         = false
```

All values are unchanged.

## 4. Seed and pair identity

| Seed | Numeric value | Pair ID | Result |
|---|---:|---|---|
| S1 | `21270083` | `MNV4-PAIR-S1` | unchanged |
| S2 | `606135704` | `MNV4-PAIR-S2` | unchanged |
| S3 | `1153870846` | `MNV4-PAIR-S3` | unchanged |

The direct and teacher members of each seed pair remain bound to the same pair identity.

## 5. Model and objective identity

Unchanged student:

`timm==1.0.26 :: mobilenetv4_conv_medium.e500_r256_in1k`

Unchanged historical teacher:

`74b4701b8931976c9227845ead50788ae47e3596f575f2817b7352a715f53b79`

Unchanged objectives:

- R04 direct: `1.00 L_CE`
- R05 teacher: `0.50 L_CE + 0.35 L_KD + 0.15 L_FEATURE`

No DDP, DataParallel, SyncBatchNorm, FSDP, model-parallel world size, execution-envelope ID, physical GPU slot, or GPU UUID was introduced into the principal scientific computation or checkpoint scientific identity.

## 6. Validation/selection and protected surfaces

Unchanged selection order:

1. validation macro-F1 descending;
2. validation balanced accuracy descending;
3. validation NLL ascending;
4. epoch ascending.

R04/R05 remain restricted to:

- `DS-V1-TRAIN`;
- `DS-V1-VAL`.

No V1-test or sealed external surface was added to the training/selection contract.

## 7. What did change

Only execution/control-plane surfaces were changed or added, including:

- additive Stage-04A execution authority metadata;
- one canonical terminal Smoke-B validator;
- clean Batch qualification gates;
- child worker-count/orchestration plumbing;
- CPU-only execution-envelope supervisor;
- isolated child GPU preflight evidence;
- dual-GPU synthetic technical smoke;
- predeclared G2 and P1/P2/P3 envelopes;
- child mutable-root and Git-token isolation;
- central G2 summary collection;
- continuation identity/state;
- durable locator/access preflight;
- parent GPU/process telemetry;
- parent-serialized evidence publication;
- science-diff/static/CI/failure-injection validation.

These changes do not alter frozen scientific computation.

## 8. Exact-head verification evidence

Actions run #77 at exact head `3dfed1074b89648eaee435567a18eddcf5a08a05` reported:

- expected SHA = actual SHA;
- Python compile PASS;
- strict repository validator PASS;
- JE static validator PASS;
- science-diff sentinel PASS;
- amendment hash PASS;
- canonical notebook code compile PASS;
- tracked forbidden-model-artifact scan PASS;
- live-looking GitHub secret scan PASS;
- MGPU contract suite: **66 tests PASS**;
- complete CPU-safe suite: **193 tests PASS**.

No real Kaggle Smoke A/B, dual-GPU smoke, G1, G2, R04 or R05 result is claimed by this repository test run.

## 9. Post-refactor verdict

# **PASS — STAGE-03R, CTC-v2, R04/R05 CONFIGS, SEEDS, MODEL IDENTITIES, OBJECTIVES AND TRAINING SEMANTICS ARE UNCHANGED**

The next operation is source freeze / wrapper binding. Any later modification to a protected science surface invalidates this verdict and requires a new science-diff review.
