# 02 — EAAI Principal Model-Science Results

**Status:** `GO — PRINCIPAL MODEL-SCIENCE TERMINAL`  
**Closure date:** 2026-09-12  
**Authority:** `EAAI-JE-SDL-v2.1-QA`  
**Principal scientific source:** `f171309fc7e9dc22241ecc137ebbb8e4bcdc5433`  
**Validation-evidence execution source:** `8904b100d223e4319776199c87ab397db23600ce`  
**Manifest SHA-256:** `bdb82211ccc2059153724eea178a1680893a6b38ecc243fae484baa91dbf68e2`  
**Class-map SHA-256:** `46f7811726c19c42bd7213b2d8178b19a5a182a1b763f60a94ee2c0e5f6688d2`

## 1. Scope and terminal ruling

This artifact closes Stage 02 for the six prospectively locked principal states:

- `R04-MNV4-DIRECT-S1/S2/S3` — matched direct-supervision controls;
- `R05-MNV4-TEACHER-S1/S2/S3` — matched full-teacher conditions.

All six scientific runs are terminal `PASS`. The immutable selected checkpoints were subsequently replayed on the complete `DS-V1-VAL` surface in an inference-only evidence pass. Each replay covered exactly **16,368** validation rows and reproduced the frozen selected-checkpoint accuracy, balanced accuracy, macro-F1 and NLL with maximum absolute difference **0.0**, well inside the frozen `1e-6` replay tolerance.

The evidence pass performed no training, did not relaunch scientific execution, did not advance optimizer state, did not access the consumed V1 test, and did not access a protected external surface.

## 2. Selected validation results

| Seed | Condition | Selected epoch | Accuracy | Balanced accuracy | Macro-F1 | NLL |
|---|---|---:|---:|---:|---:|---:|
| S1 (`21270083`) | R04 direct | 28 | 0.9808162268 | 0.9588952571 | 0.9611202956 | 0.1223255991 |
| S1 (`21270083`) | R05 full teacher | 29 | 0.9826490714 | 0.9612049433 | 0.9608684373 | 0.1033123643 |
| S2 (`606135704`) | R04 direct | 25 | 0.9803885630 | 0.9595880331 | 0.9579954271 | 0.1236968338 |
| S2 (`606135704`) | R05 full teacher | 30 | 0.9818548387 | 0.9583325119 | 0.9581742694 | 0.1088382326 |
| S3 (`1153870846`) | R04 direct | 28 | 0.9827712610 | 0.9594970699 | 0.9604158423 | 0.1157615575 |
| S3 (`1153870846`) | R05 full teacher | 28 | 0.9797165200 | 0.9518285068 | 0.9530799258 | 0.1162784909 |

## 3. Locked teacher-effect endpoint

The prospectively locked primary causal endpoint is the paired difference in selected-checkpoint validation macro-F1, **teacher minus direct**, in percentage points for the same seed.

| Seed | Teacher − direct macro-F1 (pp) | Direction |
|---|---:|---|
| S1 | −0.025186 | teacher slightly lower |
| S2 | +0.017884 | teacher slightly higher |
| S3 | −0.733592 | teacher lower |

**Mean paired delta:** `−0.246964 pp`  
**Sample SD:** `0.421981 pp`

The frozen practical-effect rule is:

- practically beneficial only if mean paired delta is `>= +0.25 pp` **and** at least 2/3 seed deltas are positive;
- practically worse only if mean paired delta is `<= −0.25 pp` **and** at least 2/3 seed deltas are negative;
- otherwise: **no practically meaningful difference under this protocol**.

The observed mean (`−0.246964 pp`) does not cross the frozen `−0.25 pp` SESOI boundary. Therefore the terminal interpretation is:

> **NO PRACTICALLY MEANINGFUL DIFFERENCE UNDER THE FROZEN PROTOCOL**

This conclusion is descriptive under the prospectively frozen engineering threshold. With only three paired seeds, **no seed-level t-test or Wilcoxon test is authorized or reported**.

## 4. Descriptive paired secondary metrics

Teacher-minus-direct descriptive differences were:

| Metric | S1 | S2 | S3 | Mean | Sample SD |
|---|---:|---:|---:|---:|---:|
| Accuracy (pp) | +0.183284 | +0.146628 | −0.305474 | +0.008146 | 0.272221 |
| Balanced accuracy (pp) | +0.230969 | −0.125552 | −0.766856 | −0.220480 | 0.505640 |
| Macro-F1 (pp) | −0.025186 | +0.017884 | −0.733592 | −0.246964 | 0.421981 |
| NLL (raw difference) | −0.019013 | −0.014859 | +0.000517 | −0.011118 | 0.010288 |

These are supporting descriptive quantities only. They do not replace the locked macro-F1 endpoint and are not used to redefine the SESOI after observing outcomes.

## 5. Recovery and retry history

The principal execution programme encountered technical session-boundary/recovery issues during the original Kaggle execution. Those events were treated strictly as infrastructure failures, not as permission to redesign the science:

- scientific source `f171309...` remained frozen;
- original A01 run identities were retained;
- checkpoint resume state was verified before continuation;
- a missing/invalid recovery checkpoint was never allowed to silently become a fresh scientific restart;
- owner-specific private durable stores were enforced;
- technical retries were distinguished from new scientific experiments;
- no new seed, objective, teacher, checkpoint-selection rule or protected-data access was introduced.

The later validation-evidence pass was inference-only and used the immutable selected checkpoints. It independently reproduced all frozen aggregate selected metrics exactly.

## 6. Evidence identities

| Experiment | Run ID | Selected checkpoint SHA-256 | Validation evidence SHA-256 | Validation predictions SHA-256 |
|---|---|---|---|---|
| R04-S1 | `JE-R04-MNV4-DIRECT-S1-f171309fc7e9-A01` | `63a5ba04a278fcc5a0a333bfc38a21ab4718df5530c9b4f99bc81ef13011251c` | `b7004401f26612f6538ec9bd50e2a361edb0e2b874165375d1a192e5d05762eb` | `0aaad16b9fd3b3e73846bb3d1955156d5ef6dedecb4bba1f91fae5bfaad796a8` |
| R05-S1 | `JE-R05-MNV4-TEACHER-S1-f171309fc7e9-A01` | `331b4eb79bd7d02fb1f2887918ff5266e3c600814d51533a6730ab8b8ce7f5fa` | `ce4da2aab89cbfad4d35a009df01a71e11820d799087a3078e372e3afdb8e583` | `22df6c1b0916bc88ad7cb2d1bb8cc4f33973afaaec0889773be709b58ed16603` |
| R04-S2 | `JE-R04-MNV4-DIRECT-S2-f171309fc7e9-A01` | `66a3e5f4a90d363c3f6b4ea342ae804af666414851f874238fd3ad83397cd32a` | `38c8a925f2e78fa844bac573f11972f71881415d62f09716b4044697a9265a1c` | `c0ddd5fefa6f48b7e8363d9e5aa033a43183a7c8404dfcab5e4affcbe6d7f7b1` |
| R05-S2 | `JE-R05-MNV4-TEACHER-S2-f171309fc7e9-A01` | `157375477b04cf36095f8b66dff557760c44839373c77691738bca7e9de8468c` | `e23064f5c8b8efdbc42b434f334e0f15e9aa2b39565c0809e90a90199d89890a` | `1de04c9d96206ca0b73653a8e9f1178b3846373586c3375856f093a048e366c6` |
| R04-S3 | `JE-R04-MNV4-DIRECT-S3-f171309fc7e9-A01` | `fa200d6add9c25856b1bdaa4d05961a0d1e4050e1204ab2a3103474b9c4a8ed2` | `efeb5666b2b0ce7874887c51ef51a06daf301f58a8d0d7f71d4706686c2069af` | `28a1404e2abf430b11c4a1cc4f5553eafbdd0bda290de0cea8b7814e4757366b` |
| R05-S3 | `JE-R05-MNV4-TEACHER-S3-f171309fc7e9-A01` | `0df653f67fe6a0f6a3c4d47fd8746ab563ef92d7182e36598602eba24b518d56` | `2b6bf5330468f60cc15f40ee70b167595c9e2ff2e9b50fdec972dab57b76a582` | `f44172fcf410d896ff95fa937f65a2a498a64d410ae0357c48a77c888286a847` |

The complete machine-readable Wave-1 closure record is stored at:

`journal_extension/evidence/public/track_a/WAVE1_PRINCIPAL_VALIDATION_CLOSURE.json`

## 7. Terminal gate

- Six principal scientific states: **PASS / terminal**
- Six selected-checkpoint validation replays: **PASS**
- Validation rows per run: **16,368**
- Maximum aggregate replay difference: **0.0**
- Training during evidence replay: **false**
- Optimizer advancement during evidence replay: **false**
- Consumed V1-test access: **false**
- Protected external access: **false**
- Scientific retuning after outcomes: **none**

# `GO — PRINCIPAL MODEL-SCIENCE TERMINAL`

Stage 02 is closed. The next Track-A evidence action is the already preauthorized secondary selected-checkpoint validation-evidence wave for `R12-MNV4-LOGITS-S1`, `R12-MNV4-FEATURE-S1`, `R06-EFFB0-CONTEXT-S1`, and `R07-CNXTT-CONTEXT-S1`.
