# Stage 01A-MGPU — Prompt 1 Science-Diff Sentinel

- **Artifact:** `01A_MGPU_01_SCIENCE_DIFF_SENTINEL.md`
- **Machine snapshot:** `01A_MGPU_01_SCIENCE_DIFF_SENTINEL.json`
- **Snapshot source:** `ee8c6ac68bc415bd94358d077a2d0762c6d2f4b9`
- **Status:** **LOCKED PRE-REFACTOR SENTINEL**
- **Purpose:** prove that the dual-T4 remediation changes execution packaging without changing Stage-03R scientific computation.

## 1. Authority bound by the sentinel

| Object | Identity |
|---|---|
| Stage-03R authority | `EAAI-JE-SDL-v2.1-QA` |
| Stage-03R SHA-256 | `aab17b65b0873dcb1ecedb061eb02ff60ccb09f8b830184f5e2231a600278f74` |
| Stage-04 base architecture | `EAAI-JE-REA-v2.2-LEAN` |
| Stage-04 SHA-256 | `a1bc1f8bb2d7a133a46c3f59f83dd2ba8ca3bf0fa8d77ae323553276931bc079` |

The Stage-03R SHA was independently recomputed from the accessible canonical authority bytes. The repository lock carries the same identity. No earlier/stale companion digest is accepted.

## 2. Protected repository surfaces

The machine snapshot freezes the Git blob identity of:

- `journal_extension/src/cropcop_je/train.py` → `a259f7c825942e7f3031b20e29f33fc55f1d9d7f`
- `journal_extension/src/cropcop_je/data.py` → `9871ece382e0345bcab2e832d3116611855c3de7`
- `journal_extension/src/cropcop_je/models.py` → `c7db1155de1621e751c4bdbad7e9c4096ab06746`
- `journal_extension/configs/common/ctc_v2.json` → `517d6fcc38e3a8e480fb1edcd97bac87907c5b5c`
- all three R04 direct configs
- all three R05 teacher configs
- `journal_extension/locks/experiment_registry.json`

Git blob identity is used here because these are repository-resident frozen objects and is deterministic over the exact Git bytes. The externally locked Stage-03R document is protected by its SHA-256 rather than by a repository blob.

## 3. Frozen batch/training semantics

The sentinel records and later CI must preserve:

```text
micro_batch_size      = 16
gradient_accumulation = 4
effective_batch_size  = 64
epochs                = 30
EMA                   = off
drop_last             = false
mixed precision       = FP16 autocast + GradScaler
gradient_clip_norm    = 1.0
sampling              = uniform row shuffle
early stopping        = false
```

Validation checkpoint selection remains exactly:

1. macro-F1 descending;
2. balanced accuracy descending;
3. NLL ascending;
4. earlier epoch.

No dual-GPU implementation may change these values or reinterpret them as a global multi-GPU batch.

## 4. Frozen paired identities

| Seed | Numeric seed | Pair | Direct | Teacher |
|---|---:|---|---|---|
| S1 | `21270083` | `MNV4-PAIR-S1` | `R04-MNV4-DIRECT-S1` | `R05-MNV4-TEACHER-S1` |
| S2 | `606135704` | `MNV4-PAIR-S2` | `R04-MNV4-DIRECT-S2` | `R05-MNV4-TEACHER-S2` |
| S3 | `1153870846` | `MNV4-PAIR-S3` | `R04-MNV4-DIRECT-S3` | `R05-MNV4-TEACHER-S3` |

Each same-seed direct/teacher pair continues to consume the identical sealed student initialization.

## 5. Frozen model/objective identities

Student:

`timm==1.0.26 :: mobilenetv4_conv_medium.e500_r256_in1k`

Historical teacher checkpoint SHA-256:

`74b4701b8931976c9227845ead50788ae47e3596f575f2817b7352a715f53b79`

Objectives:

- R04 direct: `1.00 L_CE`
- R05 full teacher: `0.50 L_CE + 0.35 L_KD + 0.15 L_FEATURE`

No model-parallel world size, physical GPU slot, GPU UUID, envelope ID, or parent orchestration state is permitted to become part of the scientific checkpoint identity.

## 6. Frozen surfaces

R04/R05 may use:

- `DS-V1-TRAIN`
- `DS-V1-VAL`

They may not use:

- `DS-V1-TEST-CONSUMED`
- sealed external surfaces
- the historical comparison surface for principal training/selection

Selection remains validation-only.

## 7. Refactor rule

Strong requirement:

> `train.py`, `data.py`, `models.py`, CTC-v2 and all R04/R05 configs remain byte-identical through the MGPU refactor.

The experiment registry is also frozen here because its seed/experiment identity is scientific. If execution metadata must be added elsewhere, do not repurpose this registry.

A post-refactor check must compare the current Git blobs against `01A_MGPU_01_SCIENCE_DIFF_SENTINEL.json`. Any mismatch in a protected blob is a stop condition until an exact diff proves the change is non-scientific. The preferred outcome is **zero protected-blob drift**.

## 8. Prompt-1 gate

`PASS — SCIENCE SNAPSHOT FROZEN; NO PROTECTED SURFACE REQUIRES MODIFICATION`

Proceed to Prompt 2. No code mutation is authorized until the architecture decision record is complete.
