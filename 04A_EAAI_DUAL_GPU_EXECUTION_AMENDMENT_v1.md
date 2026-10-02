# CropCop → EAAI Journal Extension
## Stage 04A — Dual-GPU Execution Amendment

**Authority ID:** `EAAI-JE-MGPU-A1`  
**Status:** LOCKED EXECUTION AMENDMENT — PRE-RESULTS  
**Date:** 2026-09-05  
**Base scientific authority:** `EAAI-JE-SDL-v2.1-QA`  
**Base scientific authority SHA-256:** `aab17b65b0873dcb1ecedb061eb02ff60ccb09f8b830184f5e2231a600278f74`  
**Base Stage-04 authority:** `EAAI-JE-REA-v2.2-LEAN`  
**Base Stage-04 SHA-256:** `a1bc1f8bb2d7a133a46c3f59f83dd2ba8ca3bf0fa8d77ae323553276931bc079`

---

# 1. Scope

This document is an **additive execution-packaging amendment**.

It does not edit, replace, reinterpret, or reopen Stage-03R scientific design. It does not authorize a new experiment, model, loss, seed, dataset surface, endpoint, statistical rule, batch definition, training duration, or checkpoint-selection rule.

Its sole purpose is to allow one clean Kaggle T4×2 Saved Version to supervise up to two **isolated existing single-GPU child runs**.

# 2. Preserved scientific objects

The following remain exactly governed by `EAAI-JE-SDL-v2.1-QA`:

- R04/R05 experiment definitions;
- S1/S2/S3 seeds;
- direct↔teacher same-seed pairing;
- one sealed student initialization per pair;
- MobileNetV4 identity;
- historical teacher identity;
- CTC-v2;
- micro-batch `16`;
- gradient accumulation `4`;
- effective batch `64`;
- 30 epochs;
- FP16 autocast + GradScaler;
- gradient clipping `1.0`;
- EMA off;
- `drop_last=false`;
- uniform row shuffle and deterministic row/epoch augmentation;
- direct and teacher objectives;
- validation-only checkpoint selection;
- protected-surface prohibitions;
- outcome-blind retry/governance rules.

Each individual R04/R05 run remains a normal single-GPU Stage-03R run.

# 3. Superseded Stage-04 execution-packaging sentence

Stage-04 statement:

> `One Kaggle job = one run ID.`

is superseded only by:

> **One Kaggle Saved Version = one execution envelope. One execution envelope may supervise one or two isolated child run IDs. Each child receives exactly one visible CUDA device and remains one existing single-GPU scientific/calibration state.**

No other Stage-04 scientific or evidence-governance rule is superseded.

# 4. Preserved run identity rules

- one run ID = one scientific or calibration state;
- one run ID = one checkpoint lineage;
- one run ID has one durable locator;
- one run ID has one terminal/continuation state;
- parallel children must have unique run IDs and mutable roots;
- parallel children may share only immutable/read-only inputs;
- no concurrent child publication into one mutable Git integration target;
- physical GPU slot and GPU UUID are execution metadata, not scientific checkpoint identity.

# 5. New vocabulary

## 5.1 Execution envelope

One clean Kaggle Saved Version orchestration container that may supervise one or two isolated child processes.

The envelope is infrastructure state, not a scientific experiment.

## 5.2 Child run

An existing scientific or calibration run ID executed through the existing single-GPU JE child path.

## 5.3 Logical lane

Existing K1/K2/K3 routing semantics. Logical lane remains part of existing run identity and is distinct from physical GPU placement.

## 5.4 Physical GPU slot

Host placement slot `0` or `1` assigned by the envelope before a child Python interpreter starts.

Physical slot may change on a valid later continuation.

## 5.5 Envelope state

Parent-only technical state such as:

- `PLANNED`;
- `PREFLIGHT_PASS`;
- `RUNNING`;
- `PARTIAL_TERMINAL`;
- `CONTINUATION_REQUIRED`;
- `PASS`;
- `FAIL_TECHNICAL`.

Envelope state is not a model result.

## 5.6 Envelope evidence

Machine-readable technical evidence binding source, dependency lock, run type, host inventory, child placement, output/durable isolation, timing/overlap, terminal states, publication state, and science-diff status.

Envelope evidence is not a scientific endpoint.

# 6. Execution isolation contract

Every child receives a private:

- `CROPCOP_OUTPUT_ROOT`;
- calibration-local G2-summary root when applicable;
- terminal-evidence root;
- stdout/stderr log;
- envelope child-state directory;
- unique durable locator derived from its run ID.

Children may share read-only:

- source checkout;
- V1 train/validation manifest and image root;
- class map;
- pretrained student object;
- completed G1 bundle;
- teacher object/factory/lineage evidence;
- completed central G2 summaries for principal execution.

# 7. GPU binding contract

Before child Python starts:

- child on physical slot 0 receives `CUDA_VISIBLE_DEVICES=0`;
- child on physical slot 1 receives `CUDA_VISIBLE_DEVICES=1`.

In dual-envelope mode every child must verify:

- CUDA is available;
- `torch.cuda.device_count() == 1`.

The parent must remain CPU-only and use `nvidia-smi` for physical inventory and telemetry.

No DDP, DataParallel, FSDP, model parallelism or teacher/student cross-GPU split is authorized for R04/R05.

# 8. Common session clock

All children inherit the same notebook-global:

`CROPCOP_NOTEBOOK_STARTED_MONOTONIC`

No child may establish a fresh 12-hour budget.

The envelope must refuse a new child start if insufficient common safe time remains for useful execution plus checkpoint/durable finalization.

# 9. Child Git policy

Child environments must remove:

- `CROPCOP_GITHUB_TOKEN`;
- `GITHUB_TOKEN`.

Children therefore cannot publish.

The parent retains credentials and serializes approved public-safe evidence publication after child output is stable.

A publication failure does not alter scientific results, but the evidence chain remains incomplete until repaired.

# 10. Initial resource profile

Initial dual-T4 qualification:

`CROPCOP_NUM_WORKERS_PER_CHILD=2`

Child thread bounds:

```text
OMP_NUM_THREADS=1
MKL_NUM_THREADS=1
OPENBLAS_NUM_THREADS=1
NUMEXPR_NUM_THREADS=1
TOKENIZERS_PARALLELISM=false
```

This is an execution resource profile, not a model hyperparameter.

A different loader-worker count may be adopted only for technical resource/throughput reasons before principal launch, and the affected G2 profile must then be rerun. Validation/model-quality outcomes may not drive that change.

# 11. Clean Saved-Version qualification

Terminal qualification phases require clean Kaggle Batch / Save-&-Run-All execution:

- `smoke-write`;
- `smoke-restore`;
- `dual-gpu-smoke`;
- `g1`;
- `calibration-dual`;
- `principal-dual`.

Interactive execution may be used only as `DIAGNOSTIC_NOT_QUALIFYING`.

An Interactive run cannot be promoted after the fact into a terminal qualification.

# 12. Required pre-G1 technical chain

After this amendment changes execution source:

1. freeze the new exact-head execution source under CI;
2. bind the thin notebook wrapper to that frozen source;
3. execute new-source `smoke-write` as clean Batch;
4. execute new-source `smoke-restore` in a fresh clean Batch with exact A output attached read-only;
5. execute `dual-gpu-smoke` as clean Batch;
6. independently audit all three real evidence bundles;
7. only then authorize G1.

The historical Interactive Smoke-A remains diagnostic hardware evidence only.

# 13. Canonical Smoke-B contract

One canonical terminal Smoke-B validator must be used by every downstream consumer.

It must validate the actual producer schema, including:

- supported schema;
- PASS;
- RESTORE;
- non-scientific;
- synthetic-only;
- exact execution source;
- exact dependency lock;
- clean Batch run type;
- restore/recover/resume PASS;
- exact ordered sequence `READ_A, VERIFY_A, RESTORE_A, RECOVER_A, LOAD_A, RESUME, CHECKPOINT_B`;
- expected Smoke-A checkpoint SHA equals restored SHA;
- resumed optimizer step is greater than restored step;
- attached Smoke-A input unchanged;
- G1/G2/R04/R05 not executed;
- public-safe evidence publication PASS;
- valid Smoke-A manifest/evidence hash references.

No consumer may invent an extra producer field merely to pass a gate.

# 14. Dual-GPU synthetic qualification

`dual-gpu-smoke` is technical, synthetic and non-scientific.

It must prove:

- intended homogeneous T4×2 host;
- one visible CUDA device per child;
- distinct physical GPU UUIDs;
- temporal overlap;
- optimizer-step advancement in both children;
- independently verified checkpoints;
- no output collision;
- no child Git publication;
- common session clock;
- clean parent finalization.

It does not use CropCop images, teacher bytes, G1, G2, or R04/R05.

# 15. G2 scheduling amendment

Existing calibration identities and step counts remain unchanged:

- `CAL-MNV4-DIRECT`: 200 optimizer steps;
- `CAL-MNV4-TEACHER`: 100 optimizer steps;
- `CAL-CNXTT`: 100 optimizer steps.

Envelope `MGPU-G2-T4X2-V1` schedules:

1. direct on GPU0;
2. teacher on GPU1;
3. ConvNeXt-Tiny on the first technically free slot.

The third start is triggered only by technical slot availability, never by observed metrics.

Each child writes its calibration summary into a private temporary G2 root. Parent validates and atomically collects summaries into the central G2 directory, then constructs the single existing G2 barrier once all three are present.

# 16. Principal envelopes

Predeclared pair envelopes are:

## P1 — `MGPU-P1-S1PAIR-V1`

- GPU0 → `R04-MNV4-DIRECT-S1`
- GPU1 → `R05-MNV4-TEACHER-S1`

## P2 — `MGPU-P2-S2PAIR-V1`

- GPU0 → `R04-MNV4-DIRECT-S2`
- GPU1 → `R05-MNV4-TEACHER-S2`

## P3 — `MGPU-P3-S3PAIR-V1`

- GPU0 → `R04-MNV4-DIRECT-S3`
- GPU1 → `R05-MNV4-TEACHER-S3`

Each pair uses the existing sealed same-seed initialization and the final common G2 barrier.

No outcome-driven queue mutation is permitted.

# 17. Continuation

A continuation Saved Version may receive:

`CROPCOP_ENVELOPE_INPUT_ROOT`

The parent must verify prior envelope ID, amendment identity, source SHA, G1, G2 and child run IDs.

A child already terminal PASS must not be relaunched.

Only a nonterminal/continuation-required child may resume the same run ID and durable checkpoint lineage.

Physical slot may change without creating a new experiment.

# 18. Persistence

Production G2/principal durability remains separate from API-free smoke.

Before G2/principal launch, the existing production backend must prove:

- required Kaggle credentials;
- unique run-specific durable locators;
- write/read permission;
- no collision across calibration and six principal run IDs.

This amendment does not authorize silently degrading production persistence to ephemeral notebook storage.

# 19. Failure semantics

Child-local technical failure:

- preserve a valid sibling;
- record technical evidence;
- retry only the affected child under existing machine-verifiable, science-unchanged retry rules.

Host-global invariant failure:

- signal both children to finalize safely;
- preserve valid checkpoints;
- mark continuation/failure explicitly.

Metric outcome is never a retry/schedule-edit reason.

# 20. Evidence files

Each envelope writes outside Git:

- `ENVELOPE_MANIFEST.json`;
- `ENVELOPE_EVIDENCE.json`;
- `ENVELOPE_STATE.json`;
- per-child logs;
- parent GPU telemetry JSONL.

The manifest self-hashes and binds amendment identity/hash, source, dependency lock, Kaggle run type, common clock, host inventory, child run IDs/experiments/lanes/slots, visible-device observations, mutable roots, durable locators, timing/overlap, status, publication, no-Git-child policy, protected-surface state, and science-diff result.

# 21. Source freeze

The implementation must use the existing anti-self-reference pattern:

1. final execution implementation commit;
2. exact-head CI;
3. freeze that SHA as the execution source;
4. later wrapper-only commit binds canonical notebook/generator/operator text to that source;
5. exact-head wrapper CI;
6. no scientific code change between those commits.

# 22. Scientific authority preservation statement

This amendment does not change the existing Stage-03R authority ID/hash or Stage-04 base authority ID/hash.

The authority registry may add only:

- `stage04_execution_amendment_id`;
- `stage04_execution_amendment_sha256`.

No new scientific experiment ID is introduced.

# 23. Pre-results declaration

At amendment lock time:

- no new G1 result has been produced under this amendment;
- no G2 calibration has been produced under this amendment;
- no R04/R05 result has been produced under this amendment;
- no model-quality outcome was used to choose this architecture.

# 24. Amendment verdict

# **LOCKED — DUAL-INDEPENDENT-SINGLE-GPU EXECUTION ENVELOPE AUTHORIZED**

Any implementation that changes the protected science-diff sentinel requires an explicit stop and scientific review rather than silent acceptance.
