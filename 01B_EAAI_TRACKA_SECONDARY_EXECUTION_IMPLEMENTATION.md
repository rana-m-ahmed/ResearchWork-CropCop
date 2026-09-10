# Stage-01B — Track-A Secondary Execution Implementation

**Status:** IMPLEMENTED — PENDING EXACT-HEAD CI AND REAL KAGGLE QUALIFICATION  
**Date:** 2026-09-10  
**Scientific authority:** `EAAI-JE-SDL-v2.1-QA`  
**Frozen principal science source:** `f171309fc7e9dc22241ecc137ebbb8e4bcdc5433`  
**Parent implementation head:** `f89a87d2d95e565691f41be79f9f5251e0d56cc6`  

## Purpose

This stage adds the already-authorized secondary Track-A states and a separate inference-only evidence-backfill path. It does not alter or rerun R04/R05 principal training, CTC-v2, seeds, data surfaces, objectives, selection rules, or protected-surface policy.

## Fixed secondary states

- `R12-MNV4-LOGITS-S1`: S1 MobileNetV4 initialization, historical teacher, `0.65 CE + 0.35 KD`.
- `R12-MNV4-FEATURE-S1`: S1 MobileNetV4 initialization, historical teacher, `0.85 CE + 0.15 feature`.
- `R06-EFFB0-CONTEXT-S1`: TorchVision EfficientNet-B0 / `EfficientNet_B0_Weights.IMAGENET1K_V1`, direct CE.
- `R07-CNXTT-CONTEXT-S1`: TorchVision ConvNeXt-Tiny / `ConvNeXt_Tiny_Weights.IMAGENET1K_V1`, direct CE.

All use seed `21270083`, CTC-v2, `DS-V1-TRAIN` for fitting, `DS-V1-VAL` for selection, and deny `DS-V1-TEST-CONSUMED`, protected external surfaces, and historical comparison surfaces.

## Six-GPU execution fabric

The initial high-utilization schedule uses all three Kaggle accounts and both T4s on each account without inventing additional scientific states:

| Kaggle lane | GPU 0 | GPU 1 | Purpose |
|---|---|---|---|
| K1 | R12 logits | R12 feature | Mechanism pair |
| K2 | R06 EfficientNet-B0 | R07 ConvNeXt-Tiny | Context pair |
| K3 | R04-S3 validation backfill | R05-S3 validation backfill | Principal row-level evidence |

When K1 finishes its secondary pair, K1 runs the S1 principal validation-backfill pair. When K2 finishes its secondary pair, K2 runs the S2 principal validation-backfill pair. This account-local ordering avoids assuming cross-account access to the principal private durable checkpoint datasets.

## Runtime isolation

Each scientific child receives exactly one visible T4. The parent process validates a two-T4 inventory and launches two independent one-GPU process groups. No DDP, DataParallel, SyncBatchNorm, cross-GPU gradient synchronization, or effective-batch change is introduced. GPU telemetry is recorded by the parent while the children run.

## Resume and durability

Secondary scientific run IDs are deterministic from experiment ID + immutable secondary source SHA. Each child gets an independent Kaggle durable dataset locator. Before each scientific child starts, the parent inspects the owned private durable target. A target containing `checkpoint_index.json` is forced into `resume-mode=required`; a pristine first-version target is forced into `resume-mode=never`; a later-version target without a checkpoint index is rejected as ambiguous. Restore exceptions are fatal and cannot fall through into fresh science. A notebook boundary therefore resumes the same run ID and verifies the complete checkpoint identity before restoring optimizer/scheduler/scaler/RNG/data-order/selection state. A completed durable scientific checkpoint is recognized without retraining.

## Secondary G1

Secondary G1 is additive and anchored to the verified principal G1. It:

- reuses the exact principal S1 MobileNetV4 initialization bytes;
- reuses the exact historical DINO teacher and teacher evidence;
- independently verifies official TorchVision EfficientNet-B0 and ConvNeXt-Tiny V1 pretrained artifacts;
- creates deterministic 120-way S1 baseline initialization artifacts;
- binds the new immutable source, dependency lock, frozen dataset/class map, fresh Smoke-B, and fresh dual-T4 smoke;
- is published once to a private Kaggle dataset and round-trip revalidated.

The same immutable secondary G1 private dataset must be mounted by K1 and K2. It must not be independently regenerated on each account.

## Secondary G2

The secondary G2 envelope uses both T4s as a four-profile queue. `CAL-MNV4-DIRECT` and `CAL-MNV4-TEACHER` launch first; `CAL-EFFB0` and `CAL-CNXTT` consume the first freed slots. Every profile is scheduling-only but performs a real checkpoint save → Kaggle-private durable sync → deletion of local checkpoint state → durable restore → resumed progress qualification. The parent separately verifies selected-checkpoint handling and identity-mismatch rejection, and double-publishes an immutable probe to confirm publication idempotency. One terminal secondary G2 barrier requires all four calibration summaries and binds them to one source, software stack, dependency lock, secondary G1 seal, and exact baseline pretrained identities.

## Principal validation evidence backfill

The backfill path is inference-only. For each of the six frozen principal runs it:

1. fetches the existing public run record;
2. restores the exact selected checkpoint from the owning account's private Kaggle durable dataset;
3. verifies selected checkpoint SHA and complete checkpoint identity;
4. runs deterministic `DS-V1-VAL` inference only;
5. emits `stable_row_id`, true class, predicted class, and per-row true-class NLL;
6. recomputes accuracy, macro-F1, balanced accuracy, and NLL;
7. fails if replay differs from the frozen selected metrics beyond tolerance;
8. publishes only public-safe row-level evidence to the existing run-evidence branch.

It never invokes the training loop, never advances optimizer state, never accesses V1-test, and never publishes checkpoint material.

## Operational optimization

Scientific micro-batch size, gradient accumulation and effective batch remain frozen. Infrastructure-only worker count is configurable through `CROPCOP_NUM_WORKERS_PER_CHILD`; when unset it is selected conservatively from available CPU cores. Existing pinned-memory, persistent-worker and prefetch behavior is retained. This is an operational throughput knob, not a scientific hyperparameter.

## Source policy

The principal source `f171309...` and principal canonical notebook remain frozen. Secondary code is isolated in additive modules/scripts/envelopes. Before any real secondary G1/G2/scientific execution, the new source must pass exact-head CI plus the secondary science-diff contract. Real source-qualified Smoke A/B and dual-T4 smoke are then required before secondary G1 sealing.

## Gate

`PENDING — CORE SOURCE EXACT-HEAD CI, THEN SOURCE-BOUND WRAPPER + REAL NEW-SOURCE SMOKE QUALIFICATION REQUIRED BEFORE SECONDARY G1`
