# Stage-01A G2 Preflight Remediation v1

**Status:** LOCKED — PRE-FREEZE CI PASS  
**Date:** 2026-09-06  
**Prior frozen execution source:** `3a90234f66ee09ed25141d5c45c7ed38971d69e5`  
**Scientific authority:** `EAAI-JE-SDL-v2.1-QA`  
**Authority SHA-256:** `aab17b65b0873dcb1ecedb061eb02ff60ccb09f8b830184f5e2231a600278f74`  
**Stage-04 execution amendment:** `EAAI-JE-MGPU-A1`  
**Dependency lock SHA-256:** `6ea5fb51a0cc39c7940e4aaeb136d0f05214f5c558edd3c47b4cb61c6b516f37`

## Scope

This remediation was opened only after terminal readiness, Smoke A, Smoke B, dual-GPU smoke, and G1 had passed for the prior frozen source. A pre-G2 audit found deterministic downstream implementation defects that would prevent a valid calibration/principal run.

No optimizer, objective, seed, pair, model, augmentation, split, selection rule, protected surface, or test-firewall semantic is changed.

## Authoritative Final-V1 evidence used

The certified Final-V1 artifacts are unchanged:

- `final_manifest.csv` SHA-256: `bdb82211ccc2059153724eea178a1680893a6b38ecc243fae484baa91dbf68e2`
- `class_to_idx.json` SHA-256: `46f7811726c19c42bd7213b2d8178b19a5a182a1b763f60a94ee2c0e5f6688d2`
- rows: 109,107
- train: 76,376
- validation: 16,368
- test: 16,363
- classes: 120

The certified manifest header contains `record_key`, `label`, `split`, and `portable_relpath`; it does **not** contain a numeric class-index column.

A full-row audit established:

- all 109,107 labels are present in the locked class map;
- the class-map values form an exact bijection over integers 0..119;
- no duplicate `record_key`;
- no duplicate `portable_relpath`;
- every portable path is consistent with its frozen split.

## R1 — Frozen manifest/class-index mismatch

### Defect

The production training and ConvNeXt calibration loaders required a numeric manifest class-index column and called `int(raw[column])`. No such column exists in the certified manifest. Any operator-selected substitute would either fail conversion or silently encode the wrong target.

### Remediation

A deterministic adapter now binds exactly:

- stable row ID: `record_key`
- image path: `portable_relpath`
- split: `split`
- label: `label`
- numeric target: derived only from the hash-locked `class_to_idx.json`

The adapter rejects:

- any historical `relative_path` substitution;
- any non-empty numeric class-index override;
- class maps that are not a direct 120-entry bijection over 0..119;
- missing labels;
- duplicate row IDs/portable paths;
- unsafe or split-inconsistent portable paths.

The manifest bytes and hashes are not modified.

## R2 — Unreachable terminal envelope publication

### Defect

The first assignment to `envelope_branch` in `run_envelope.py` was indented inside the prior-continuation mutation guard after a `raise`, making the assignment unreachable. A normal first calibration envelope could complete child work and then fail with an unbound terminal publication variable.

### Remediation

The publication assignment now executes after the continuation-integrity guard and before the publication branch is recorded in terminal evidence.

## R3 — ConvNeXt calibration control identity not centrally bound

### Defect

The design lock specifies TorchVision `ConvNeXt_Tiny_Weights.IMAGENET1K_V1`, but the calibration path accepted an arbitrary ConvNeXt-Tiny state file and the central G2 barrier did not bind the observed ConvNeXt pretrained SHA.

### Remediation

`CAL-CNXTT` now requires:

- basename `convnext_tiny-983f1562.pth`;
- SHA-256 prefix `983f1562`, matching TorchVision's published IMAGENET1K_V1 hash prefix.

The full observed SHA-256 is retained in the calibration summary and is now included and validated in the central G2 barrier.

This calibration remains scheduling/context evidence only and does not become an R04/R05 scientific checkpoint.

## R4 — Durable Kaggle recovery privacy not fail-closed

### Defect

The production Kaggle durable-store preflight required owner identity and authenticated readability but did not prove that recovery datasets were private. A public dataset could therefore pass the old preflight and later receive checkpoint/recovery material.

### Remediation

Every current-envelope Kaggle durable locator must now pass the same authoritative private-target metadata/ownership/status preflight used for the sealed G1 target, plus the existing CLI read probe.

Owner mismatch fails before any network/API request.

## Scientific non-change statement

The following remain byte/semantically frozen except for the explicitly re-frozen loader implementation required to consume the certified manifest correctly:

- CTC-v2 preprocessing and augmentation;
- AdamW schedule and hyperparameters;
- micro/effective batch sizes;
- three training seeds;
- paired S1/S2/S3 identities;
- R04 direct objective;
- R05 teacher objective;
- MobileNetV4 model identity;
- historical DINO teacher identity;
- V1 train/validation surface contract;
- consumed V1 test firewall;
- external sealed-surface firewall;
- principal selection rules.

The implementation change realizes the already-locked requirement that class ordering comes from the final `class_to_idx.json`; it does not relabel, resplit, mutate, or reinterpret the benchmark.

## Qualification consequence

Because the execution source SHA changes and G1/G2 envelopes intentionally require exact source equality, the prior qualification evidence remains valid historical evidence but cannot authorize the repaired source.

After terminal CI and source freeze, the repaired source must replay the source-bound chain:

1. readiness;
2. Smoke A;
3. Smoke B;
4. dual-GPU smoke;
5. G1;
6. G2 calibration-dual;
7. principal envelopes.

This replay is required by the integrity architecture, not because the prior passed artifacts were scientifically invalid.

## Runtime-change moratorium after re-freeze

After the repaired source is frozen, no further source change is authorized unless a new execution produces a concrete earliest-causal defect attributable to the frozen source. Warnings, style changes, speculative hardening, and convenience changes are insufficient grounds for another source SHA.


## Pre-freeze validation closure

The consolidated repaired runtime state at `1d6ce71c62411132e0b32643fd4f9bcc2169abc1` passed workflow **Validate public evidence #257**:

- repository/static contract: PASS;
- science-diff + amendment: PASS;
- canonical notebook compile/parity: PASS;
- QA1 Waves A/B/C: PASS;
- MGPU envelope contract: PASS;
- complete CPU-safe suite: **337/337 PASS**;
- exact Kaggle 2.2.4 API contract: PASS;
- offline DINOv3/Transformers compatibility: PASS.

The next commit containing this closure is documentation/governance only and may be selected as the immutable repaired execution source. No further runtime change is authorized before replay qualification.
