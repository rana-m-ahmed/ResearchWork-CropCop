# 03B — EAAI Track-A Strengthening Amendment v1

**Status:** `POST-RESULTS AMENDMENT — PRE-EXECUTION LOCK`  
**Date:** 2026-09-12  
**Amendment ID:** `EAAI-JE-TRACKA-STRENGTHENING-A1`  
**Base Track-A pre-lock head:** `27a25d54b482806f924b6030e55c37bcc9e1f106`  
**Original authority:** `EAAI-JE-SDL-v2.1-QA`  
**Original authority SHA-256:** `aab17b65b0873dcb1ecedb061eb02ff60ccb09f8b830184f5e2231a600278f74`  
**Track-A final lock:** `FORBIDDEN UNTIL THIS AMENDMENT CLOSES`

## 1. Why this amendment exists

Stage 02 and Stage 03A are scientifically valid and remain immutable historical evidence. The second QA pass found no need to repeat the principal R04/R05 experiment. The remaining reviewer-visible weakness is asymmetry: the principal teacher/direct question has three frozen seeds, whereas the architecture-context and R12 mechanism evidence currently have only S1 and are therefore explicitly descriptive.

Current EAAI scope emphasizes real engineering applications, reproducibility on public data, verification/validation, safety/reliability, real-time AI architecture, and benchmarking. Recent plant-health work in EAAI also emphasizes laboratory-to-field generalization, reliable recognition, and embedded/lightweight deployment. This amendment addresses those evidence dimensions without changing the original result after seeing it.

This document is a **post-results amendment**. It must never be represented as part of the original prospective Stage-03R design.

## 2. Original results are immutable

The amendment may add evidence but may not:
- change the original R04/R05 or Stage-02 conclusion;
- replace or delete original R06/R07/R12 S1 evidence;
- change existing selected checkpoints;
- change the historical teacher;
- alter CTC-v2;
- introduce new model families;
- introduce new seeds;
- tune objectives after seeing amendment results;
- use `DS-V1-TEST-CONSUMED`;
- use sealed external surfaces for model selection.

## 3. New training states — exactly eight

Seeds are reused from the original principal experiment:
- S2 = `606135704`;
- S3 = `1153870846`.

Authorized new states:
1. R06 EfficientNet-B0 S2
2. R07 ConvNeXt-Tiny S2
3. R12 MobileNetV4 logits KD S2
4. R12 MobileNetV4 feature distillation S2
5. R06 EfficientNet-B0 S3
6. R07 ConvNeXt-Tiny S3
7. R12 MobileNetV4 logits KD S3
8. R12 MobileNetV4 feature distillation S3

Every state uses unchanged CTC-v2, 30 epochs, validation-only selection, and the exact S1 family objective. No hyperparameter search is authorized.

R12 S2/S3 must reuse the already frozen principal `PAIR_INIT_S2` / `PAIR_INIT_S3` student identities. EfficientNet-B0 and ConvNeXt-Tiny S2/S3 require a new **Amendment-G1A initialization seal** produced deterministically from their already frozen official pretrained objects plus the specified seed before any science run begins.

## 4. New inference-only evidence

### 4.1 Robustness suite
After all architecture states are terminal, evaluate the three direct model families across S1/S2/S3 on the exact fixed corruption suite in `robustness_protocol.json`. No adaptation, threshold tuning, severity tuning, or corruption cherry-picking is allowed.

### 4.2 120-class error analysis
Use row-level selected-checkpoint predictions only. Report per-class metrics, bottom-tail behavior and fixed confusion summaries under `analysis_protocol.json`. No 120-way post-hoc significance testing is authorized.

### 4.3 Efficiency context
Report exact parameter counts and state-tensor bytes for the three architecture families. Theoretical MAC/FLOP values are optional and must be omitted if one pinned profiler cannot produce comparable documented coverage across all three models. Real device latency belongs to Track C.

## 5. External validity is separate

Track B remains the external-validity authority. This amendment does not turn synthetic corruption into field validation and does not authorize a convenient reused source as an external cohort. Candidate independence must be established before predictions are opened.

## 6. Claims authorized after successful closure

If all eight states complete with intact identities, the manuscript may report:
- three-seed mean and sample SD for MNV4 direct, EfficientNet-B0 and ConvNeXt-Tiny;
- paired seedwise descriptive architecture deltas against MNV4;
- three-seed mean/SD for R12 logits and feature variants;
- paired logits-minus-feature descriptive deltas;
- clean-to-corruption degradation under the fixed internal robustness protocol;
- 120-class tail/confusion evidence;
- exact parameter/state-size context.

With n=3, the amendment does not automatically authorize null-hypothesis significance tests.

## 7. Claims still forbidden

Even after closure, this amendment alone cannot establish:
- field generalization;
- universal architecture superiority;
- universal device-speed superiority;
- clinical/agronomic autonomy;
- equivalence between teacher and direct learning;
- robustness to arbitrary unseen crops/diseases;
- production readiness.

## 8. Parallel execution strategy

Use three qualified Kaggle T4x2 accounts in parallel. Wave 1 executes six states concurrently; Wave 2 executes the two remaining R12-S3 states on the first qualified account that becomes available. Each T4 remains an independent single-GPU child. No DDP/DataParallel/FSDP.

The exact schedule is frozen in `parallel_execution_plan.json`.

## 9. Gates

- **A1-G0 Amendment lock:** this document + registries/protocols committed and hashed.
- **A1-G1A Identity:** new baseline S2/S3 initialization objects sealed; existing pair/teacher/pretrained identities verified.
- **A1-G2 Execution readiness:** implementation source exact-head CI PASS; no science diff outside authorized amendment registry/config plumbing.
- **A1-G3 Training:** all eight states terminal PASS or transparently reported failure.
- **A1-G4 Replay:** selected checkpoints replay full 16,368-row DS-V1-VAL with aggregate tolerance <= 1e-6.
- **A1-G5 Robustness/error/efficiency:** fixed inference-only analyses complete.
- **A1-G6 Cross-track reconciliation:** Track B/C claim boundaries checked.
- **A1-G7 Final review:** only then may a separate pass consider final Track-A lock.

## 10. Stop rules

No new experiment may be added after A1-G0 merely because amendment results are disappointing. A new experiment requires a separate named amendment committed before its outputs. Technical retries preserve the same scientific identity. Protected test access remains forbidden.

# `GO — IMPLEMENT AMENDMENT INFRASTRUCTURE; DO NOT START SCIENCE UNTIL A1-G1A/A1-G2 PASS`
