# Stage 01A-MGPU — EAAI Dual-T4 Refactor Implementation Report

- **Artifact:** `01A_MGPU_EAAI_DUAL_T4_REFACTOR_IMPLEMENTATION_REPORT.md`
- **Date:** 2026-09-05
- **Repository:** `rana-m-ahmed/ResearchWork-CropCop`
- **PR:** #8 — `je-stage01a-core-20260905`
- **Status:** **REPOSITORY IMPLEMENTATION CLOSED — REAL NEW-SOURCE KAGGLE QUALIFICATION PENDING**

## 1. Authority stack

### 1.1 Scientific authority

- **Stage-03R ID:** `EAAI-JE-SDL-v2.1-QA`
- **Stage-03R SHA-256:** `aab17b65b0873dcb1ecedb061eb02ff60ccb09f8b830184f5e2231a600278f74`

Stage-03R remains the scientific authority. It was not replaced, reinterpreted, or re-locked by the MGPU remediation.

### 1.2 Base execution architecture

- **Stage-04 ID:** `EAAI-JE-REA-v2.2-LEAN`
- **Stage-04 SHA-256:** `a1bc1f8bb2d7a133a46c3f59f83dd2ba8ca3bf0fa8d77ae323553276931bc079`

Stage-04 remains the base repository/execution architecture.

### 1.3 Additive MGPU execution amendment

- **Stage-04A ID:** `EAAI-JE-MGPU-A1`
- **Stage-04A SHA-256:** `3f08f2dbe7e83143c7e6f1fcf6732c7d88086295a600a74b323b89b46caed3f6`
- **Scope:** execution packaging only

The amendment authorizes one Kaggle T4×2 Saved Version to supervise isolated existing single-GPU child states. It does not create a new scientific experiment or change frozen Stage-03R computation.

## 2. Mandatory pre-coding audit chain

The master-prompt audit sequence was completed before implementation:

1. `01A_MGPU_00_LIVE_AUTHORITY_AUDIT.md`
2. `01A_MGPU_01_SCIENCE_DIFF_SENTINEL.json`
3. `01A_MGPU_01_SCIENCE_DIFF_SENTINEL.md`
4. `01A_MGPU_02_ARCHITECTURE_DECISION_RECORD.md`

Architecture verdict:

> **ACCEPT — dual-independent-single-GPU execution envelope**

Rejected for principal scientific execution:

- `nn.DataParallel`;
- DDP;
- DDP + SyncBatchNorm;
- teacher/student cross-GPU split.

No scientific re-lock was required.

## 3. Frozen execution source

```text
MGPU_EXECUTION_SOURCE_SHA = ba5dd4661b97d072593af4646b76552686953a2d
```

This SHA is the frozen Stage-01A-MGPU execution source.

It contains the completed execution implementation plus the mandatory post-refactor science-diff artifact.

It is not superseded by later notebook-wrapper or documentation commits.

### 3.1 Exact-head source CI

- **GitHub Actions run:** #78
- **Run ID:** `33975913927`
- **Exact source SHA:** `ba5dd4661b97d072593af4646b76552686953a2d`
- **Conclusion:** SUCCESS

Verified by run #78:

- expected checkout SHA == actual checkout SHA;
- Python compile PASS;
- strict repository validator PASS;
- JE static validator PASS;
- MGPU science-diff PASS;
- Stage-04A amendment hash PASS;
- canonical notebook code compile PASS;
- forbidden tracked model/checkpoint artifact scan PASS;
- live-looking GitHub secret scan PASS;
- MGPU suite: **66 / 66 PASS**;
- complete CPU-safe suite: **193 / 193 PASS**.

Run #78 is the execution-source proof and must be preserved as such.

## 4. Frozen scientific computation remained unchanged

The post-refactor science-diff artifact is:

`01A_MGPU_90_POST_REFACTOR_SCIENCE_DIFF.md`

Verdict:

> **PASS — NO SCIENTIFIC DRIFT**

The following frozen repository surfaces remained byte-identical through the execution refactor:

- `journal_extension/src/cropcop_je/train.py`;
- `journal_extension/src/cropcop_je/data.py`;
- `journal_extension/src/cropcop_je/models.py`;
- `journal_extension/configs/common/ctc_v2.json`;
- all R04 direct configs;
- all R05 teacher configs;
- `journal_extension/locks/experiment_registry.json`.

Unchanged scientific semantics include:

- micro-batch = 16;
- gradient accumulation = 4;
- effective batch = 64;
- epochs = 30;
- AdamW;
- FP16 autocast + GradScaler;
- gradient clipping = 1.0;
- EMA off;
- `drop_last=false`;
- uniform row shuffle and deterministic row/epoch augmentation;
- validation-only checkpoint selection;
- S1/S2/S3 seeds;
- same-seed R04↔R05 pair identities;
- MobileNetV4 identity;
- historical teacher identity;
- R04 and R05 objectives;
- protected-surface prohibitions.

Physical GPU slot, GPU UUID and execution-envelope ID were not added to the scientific checkpoint identity.

## 5. Implemented MGPU execution controls

Repository implementation includes:

- canonical terminal Smoke-B producer/consumer contract;
- clean Kaggle Saved-Version/Batch qualification;
- Interactive runs rejected as terminal qualification;
- CPU-only parent envelope supervisor;
- two isolated single-GPU children;
- pre-Python `CUDA_VISIBLE_DEVICES` pinning;
- child verification of exactly one visible CUDA device;
- child Git-token stripping;
- private child mutable roots;
- common notebook-global session clock;
- process-group termination/finalization handling;
- parent GPU and compute-process telemetry;
- durable locator collision protection;
- durable access preflight;
- continuation identity and terminal-child skip semantics;
- parent-serialized public-safe evidence publication;
- dual-GPU synthetic technical smoke;
- central G2 summary validation/collection;
- predeclared P1/P2/P3 same-seed principal envelopes;
- machine-enforced science-diff CI.

## 6. Final canonical wrapper

### 6.1 Final wrapper SHA

`7e67ed785c9a3a89d892f9f82d28bd8dc471bea4`

This is the final green wrapper commit.

It is **not** the execution source.

Its source parent chain remains bound to:

`ba5dd4661b97d072593af4646b76552686953a2d`

The only differences from the frozen execution source are wrapper/validation surfaces:

- `.github/workflows/validate-artifacts.yml`;
- `journal_extension/kaggle/generate_canonical_notebook.py`;
- `journal_extension/kaggle/canonical_lane.ipynb`;
- `journal_extension/src/cropcop_je/validate.py`;
- wrapper-related tests.

No frozen execution/scientific implementation file changed after source freeze.

### 6.2 Generator/source binding

The final generator records:

```text
MGPU_EXECUTION_SOURCE_SHA = ba5dd4661b97d072593af4646b76552686953a2d
AUTHORIZED_SOURCE_SHA = ba5dd4661b97d072593af4646b76552686953a2d
```

The generated canonical notebook contains the same literal authorized source SHA.

The notebook is nbformat 4 with one thin orchestration code cell and real multiline Python source.

CI executes the generator and proves the committed notebook exactly matches generator output.

## 7. Final operator-facing phase vocabulary

The canonical notebook exposes exactly:

- `smoke-write`;
- `smoke-restore`;
- `dual-gpu-smoke`;
- `g1`;
- `calibration-dual`;
- `principal-dual`.

Legacy operator-facing production aliases:

- `calibration`;
- `principal`;

are not routed by the final notebook.

There is no silent phase fallthrough.

## 8. Principal envelope selection

For `principal-dual`, the non-secret operator setting is:

`CROPCOP_PRINCIPAL_ENVELOPE`

Allowed values are exactly:

- `P1`;
- `P2`;
- `P3`.

The notebook does not define arbitrary scientific combinations. It passes the validated envelope selection to the existing repository envelope runner.

Repository predeclared mappings remain:

- P1 → `R04-MNV4-DIRECT-S1` + `R05-MNV4-TEACHER-S1`;
- P2 → `R04-MNV4-DIRECT-S2` + `R05-MNV4-TEACHER-S2`;
- P3 → `R04-MNV4-DIRECT-S3` + `R05-MNV4-TEACHER-S3`.

## 9. Dual calibration and dual smoke wrapper routing

`calibration-dual` delegates directly to:

`journal_extension/kaggle/run_envelope.py`

The notebook does not duplicate G2 queue/scheduler logic.

`dual-gpu-smoke` delegates directly to:

`journal_extension/scripts/smoke_dual_gpu.py`

The dual-GPU smoke remains:

- synthetic;
- non-scientific;
- no CropCop data;
- no G1;
- no G2;
- no R04/R05.

## 10. Smoke A/B wrapper contract

`smoke-write` and `smoke-restore` retain the existing cross-Saved-Version API-free workflow.

Smoke phases require only:

`CROPCOP_GITHUB_TOKEN`

Kaggle API credentials are not reintroduced into Smoke A/B.

Smoke B still requires an explicit attached Smoke-A input root.

The terminal Smoke-B validator remains the frozen execution-source implementation.

## 11. Later-phase credentials and artifacts

For G1/G2/principal routes the wrapper preserves the frozen execution source's real credential requirements.

The clean-session bootstrap requires:

- `CROPCOP_GITHUB_TOKEN`;
- `KAGGLE_USERNAME`;
- `KAGGLE_KEY`;

for `g1`, `calibration-dual`, and `principal-dual`.

Artifact/durability variables remain delegated to the frozen execution implementation. The wrapper does not weaken:

- G1 identity requirements;
- Smoke-B evidence requirement;
- G2 summary/barrier identity;
- private durable-store requirements;
- teacher/student artifact identity;
- checkpoint/resume semantics.

## 12. Notebook-global session clock

The canonical notebook establishes:

`CROPCOP_NOTEBOOK_STARTED_MONOTONIC`

before:

- repository clone;
- dependency installation;
- clean-session bootstrap;
- G1/smoke/envelope launch.

Child processes inherit the same clock.

No child receives a new twelve-hour budget.

## 13. Final wrapper exact-head CI

- **Final wrapper SHA:** `7e67ed785c9a3a89d892f9f82d28bd8dc471bea4`
- **GitHub Actions run:** #80
- **Run ID:** `33976471717`
- **Conclusion:** SUCCESS

Run #80 proved:

- expected checkout SHA == actual wrapper SHA;
- Python compile PASS;
- strict repository validator PASS;
- JE static validator PASS;
- science-diff PASS;
- Stage-04A amendment hash PASS;
- canonical notebook code compile PASS;
- **canonical notebook generation consistency PASS**;
- secret scan PASS;
- forbidden-artifact scan PASS;
- MGPU tests: **66 / 66 PASS**;
- complete CPU-safe suite: **193 / 193 PASS**.

A prior wrapper attempt, run #79, correctly failed because one historical test still required the notebook to call `run_lane.py` directly. That stale wrapper-test expectation was corrected without changing generator routing or frozen execution code. Run #80 is the terminal wrapper proof.

## 14. Historical real Smoke-A status

Historical evidence from the earlier source proved a real two-Tesla-T4 host and successful infrastructure behavior, but its observed Kaggle mode was:

`KAGGLE_KERNEL_RUN_TYPE = Interactive`

Therefore it remains:

> **HISTORICAL DIAGNOSTIC EVIDENCE ONLY**

It is not terminal qualification for the frozen Stage-01A-MGPU source.

It must not be promoted into new-source Smoke A/B evidence.

## 15. Claims explicitly not made

Repository closure does **not** claim any of the following for source `ba5dd466...`:

- real new-source Smoke A PASS;
- real new-source Smoke B PASS;
- real dual-GPU smoke PASS;
- G1 PASS;
- G2 PASS;
- R04 launched;
- R05 launched;
- any principal scientific result.

No real scientific result was fabricated, inferred from CI, or committed as part of this closure.

## 16. Repository closure gate

All repository-side requirements for the Stage-01A-MGPU source-freeze/wrapper-bind pattern are satisfied:

- frozen execution source independently green;
- wrapper hard-bound to that source;
- exact six-phase vocabulary;
- P1/P2/P3-only principal envelope selection;
- envelope/G2/dual-smoke delegation;
- Smoke A/B API-free behavior preserved;
- Batch qualification preserved;
- notebook-global clock preserved;
- generator consistency proven;
- science-diff remains green;
- final wrapper exact-head CI green;
- frozen scientific/execution implementation unchanged after source freeze.

# **REPOSITORY CLOSURE PASS — REAL NEW-SOURCE KAGGLE QUALIFICATION REMAINS NEXT**

The next qualification boundary begins with a new clean Saved-Version `smoke-write` under frozen source `ba5dd4661b97d072593af4646b76552686953a2d`.
