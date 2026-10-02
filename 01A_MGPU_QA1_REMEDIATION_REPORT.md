# Stage 01A-MGPU-QA1 — Remediation Report

- **Artifact:** `01A_MGPU_QA1_REMEDIATION_REPORT.md`
- **Date:** 2026-09-05
- **Repository:** `rana-m-ahmed/ResearchWork-CropCop`
- **PR:** #8 / `je-stage01a-core-20260905`
- **QA1 scope:** post-implementation execution, recovery, evidence chronology and operator-safety remediation
- **Scientific redesign:** NONE
- **Real Kaggle execution during QA1:** NONE

## 1. Starting live state

QA1 began from independently verified live PR head:

`e7ad831dae327c4e81b9417e46cb47c5237354af`

with:

- main: `32190dd86293caa82170df3feea505e3c7443b4b`;
- PR #8: OPEN / DRAFT;
- exact-head CI #81: SUCCESS;
- pre-QA MGPU execution source: `ba5dd4661b97d072593af4646b76552686953a2d`;
- Stage-03R: `EAAI-JE-SDL-v2.1-QA`;
- Stage-03R SHA-256: `aab17b65b0873dcb1ecedb061eb02ff60ccb09f8b830184f5e2231a600278f74`;
- Stage-04 base SHA-256: `a1bc1f8bb2d7a133a46c3f59f83dd2ba8ca3bf0fa8d77ae323553276931bc079`;
- Stage-04A: `EAAI-JE-MGPU-A1`;
- Stage-04A SHA-256: `3f08f2dbe7e83143c7e6f1fcf6732c7d88086295a600a74b323b89b46caed3f6`.

The required live confirmation artifact is:

`01A_MGPU_QA1_00_LIVE_CONFIRMATION.md`

All eight independently reported QA findings reproduced and were classified **CONFIRMED**.

## 2. Confirmed findings and closure

### S0-1 — Stage-04A chronology was not machine-enforced

**Closed.**

Created one canonical:

`validate_terminal_dual_gpu_smoke_evidence(...)`

It verifies:

- schema/status/qualification ID;
- non-scientific synthetic-only state;
- exact source;
- exact dependency lock;
- Stage-04A ID/hash;
- clean Batch mode;
- exact T4×2 inventory;
- distinct physical GPU UUIDs;
- expected child IDs and physical slots;
- one visible CUDA device per child;
- T4 identity;
- optimizer progress;
- checkpoint SHA/byte evidence;
- common notebook clock;
- positive overlap;
- no child Git credentials/publication;
- no restricted CropCop data;
- G1/G2/R04/R05 false;
- science-diff PASS;
- public-safe publication PASS;
- exact `run-evidence/DUAL-GPU-SMOKE` branch.

Dual smoke now records:

`smoke_b_evidence_sha256 = sha256_json(terminal_smoke_b_object)`

G1 launch, G1 sealing, G1 barrier validation, `calibration-dual`, and `principal-dual` all refuse missing/stale/invalid dual-smoke evidence.

The dual-smoke digest remains technical chronology evidence and is not scientific checkpoint identity.

### S1-2 — 30-second parent grace could kill a valid finalizing child

**Closed.**

Planned/common-deadline finalization is distinct from emergency termination.

Planned finalization now:

- sends SIGTERM to running children together;
- derives bounded grace from notebook finalization margin and conservative checkpoint/sync estimates;
- polls during grace;
- accepts children that exit after more than 30 seconds but within the bounded finalization window;
- SIGKILLs only genuinely hung child groups after grace;
- records termination mode and duration;
- runs normal child preflight/result collection after SIGTERM.

Emergency runtime timeout remains a separate shorter kill path.

### S1-3 — publication failure could be laundered through continuation

**Closed.**

Child state now separates:

- execution status;
- publication status;
- evidence-chain completeness.

A PASS result with publication failure is classified as **publication repair**, not a terminal skip and not a training retry.

Continuation:

1. validates the prior terminal result;
2. validates required metrics/segment evidence or calibration summary;
3. carries the exact prior result into the new envelope;
4. retries publication only;
5. preserves the same run ID/result;
6. records `training_relaunched=false`;
7. permits envelope PASS only after evidence-chain completion.

### S1-4 — completed epoch-30 checkpoint recovery edge

**Closed without editing `train.py`.**

New execution-state module:

`journal_extension/src/cropcop_je/terminal_recovery.py`

recognizes a terminal checkpoint only when:

- recovered candidate is verified `latest`;
- epoch is at/after locked final epoch;
- batch cursor is zero;
- data-order cursor is terminal;
- optimizer step is valid;
- selection history covers locked epochs;
- selected checkpoint identity verifies.

`scripts/run_training.py` checks this before the scientific training loop.

If recognized, terminal evidence is reconstructed from already-persisted metadata with:

- **0 optimizer steps advanced**;
- no scheduler advancement;
- no forward/validation recomputation;
- no invented metrics.

Otherwise normal resume remains authoritative.

### S2-5 — durability access preflight was unnecessarily global

**Closed.**

Global locator namespace validation remains over all nine reserved calibration/principal IDs.

Authenticated access probing is now scoped to the current envelope only:

- G2 → exactly 3 calibration IDs;
- P1 → S1 direct/teacher pair only;
- P2 → S2 pair only;
- P3 → S3 pair only.

Owner binding, authenticated read, run-specific locator uniqueness and global collision protection remain fail-closed.

No fake/throwaway Kaggle dataset version is created as a preflight write.

The first real durable sync remains the actual write exercise.

### S1-6 — active operator docs were stale

**Closed.**

Current operator surfaces are synchronized to the final QA1 source:

- `journal_extension/kaggle/README_SMOKE.md`;
- `journal_extension/kaggle/smoke_inputs.example.json`;
- marked current QA1 section of `journal_extension/README.md`.

A CI validator prevents drift from generator-authorized source, six-phase vocabulary and dual-smoke-before-G1 chronology.

Historical reports/chronology remain untouched and are explicitly non-active.

### S2-7 — wrapper lacked explicit cross-Saved-Version evidence handoff

**Closed.**

Canonical wrapper now exposes:

- `CROPCOP_SMOKE_A_INPUT_ROOT`;
- `CROPCOP_SMOKE_B_INPUT_ROOT`;
- `CROPCOP_DUAL_GPU_SMOKE_INPUT_ROOT`.

Each lookup is restricted below the explicit attached Notebook Output root and requires exactly one expected evidence file.

There is no global `/kaggle/input` search.

### S2-8 — continuation trusted state more than full evidence bundle

**Closed.**

Continuation requires one coherent, co-located:

- `ENVELOPE_MANIFEST.json`;
- `ENVELOPE_EVIDENCE.json`;
- `ENVELOPE_STATE.json`.

Validation covers:

- manifest self-hash;
- source/amendment/G1/G2 identity;
- evidence→manifest binding;
- child/run IDs;
- execution/publication/completeness coherence;
- actual terminal result artifact;
- required principal metrics/segments;
- calibration summary validation.

Attached prior control files are fingerprinted before and after continuation use.

## 3. Wave verification

### Wave A

- exact head: `a439277ba70dd3638a955231d39e2bbddaaa8e8d`
- Actions run: #95
- run ID: `33978283923`
- result: SUCCESS
- Wave-A tests: **18 / 18**
- MGPU: **66 / 66**
- total: **211 / 211**
- science-diff: PASS

Report:

`01A_MGPU_QA1_WAVE_A_REPORT.md`

### Wave B

- exact head: `66a5934b0c35dbd84048af84b9f927ac41b1b96d`
- Actions run: #112
- run ID: `33978769002`
- result: SUCCESS
- Wave-A: **18 / 18**
- Wave-B: **13 / 13**
- MGPU: **66 / 66**
- total: **224 / 224**
- science-diff: PASS

Report:

`01A_MGPU_QA1_WAVE_B_REPORT.md`

### Wave C / final execution-source freeze

- **final QA1 execution source:** `be9b6965d760ff6e8674623b658f66572cd57093`
- Actions run: #118
- run ID: `33978909616`
- result: SUCCESS
- Wave-A: **18 / 18**
- Wave-B: **13 / 13**
- Wave-C: **8 / 8**
- MGPU: **66 / 66**
- complete CPU-safe suite: **232 / 232**
- science-diff: PASS

Run #118 additionally proves:

- expected checkout SHA == actual source SHA;
- compile PASS;
- repository validator PASS;
- JE static PASS;
- Stage-04A hash PASS;
- notebook source compile PASS;
- generator consistency PASS;
- active docs validator PASS;
- secret scan PASS;
- forbidden-artifact scan PASS.

The former source `ba5dd4661b97d072593af4646b76552686953a2d` is now historical pre-QA source evidence.

## 4. Final source/wrapper anti-self-reference closure

### Execution source

`be9b6965d760ff6e8674623b658f66572cd57093`

This is the immutable QA1 execution source.

No execution/recovery implementation file changed after this SHA.

### Final wrapper

`2f422d4a5b6a9d042cc0d99a49b2f08c4544bce9`

Changes after source freeze were limited to:

- `journal_extension/kaggle/generate_canonical_notebook.py`;
- regenerated `journal_extension/kaggle/canonical_lane.ipynb`;
- active operator docs/example;
- source-binding static validation expectation;
- source-binding notebook test expectation.

The source→wrapper diff contains **no execution/recovery implementation file**.

The generator and notebook both hard-bind:

`AUTHORIZED_SOURCE_SHA = "be9b6965d760ff6e8674623b658f66572cd57093"`

### Wrapper CI

- Actions run: #124
- run ID: `33979031988`
- exact wrapper SHA: `2f422d4a5b6a9d042cc0d99a49b2f08c4544bce9`
- conclusion: SUCCESS

Run #124 proves:

- expected checkout SHA == actual wrapper SHA;
- compile PASS;
- repository validator PASS;
- JE static PASS;
- science-diff PASS;
- Stage-04A hash PASS;
- notebook code compile PASS;
- generator == notebook PASS;
- active operator docs PASS;
- secret scan PASS;
- forbidden-artifact scan PASS;
- Wave-A **18/18**;
- Wave-B **13/13**;
- Wave-C **8/8**;
- MGPU **66/66**;
- complete suite **232/232**.

## 5. Changed execution/remediation files

From starting QA1 head through final execution source, remediation touched:

- `.github/workflows/validate-artifacts.yml`;
- `journal_extension/kaggle/run_envelope.py`;
- `journal_extension/kaggle/run_g1.py`;
- `journal_extension/scripts/run_training.py`;
- `journal_extension/scripts/seal_g1.py`;
- `journal_extension/scripts/smoke_dual_gpu.py`;
- `journal_extension/scripts/validate_g1_barrier.py`;
- `journal_extension/scripts/validate_operator_docs.py`;
- `journal_extension/src/cropcop_je/envelope.py`;
- `journal_extension/src/cropcop_je/smoke_handoff.py`;
- `journal_extension/src/cropcop_je/terminal_recovery.py`;
- active wrapper/docs surfaces during Wave A;
- QA1 and adjusted regression tests;
- QA1 live/Wave reports.

## 6. Scientific integrity

The science-diff sentinel remained PASS through every green wave and final wrapper CI.

Unchanged protected objects include:

- `train.py`;
- `data.py`;
- `models.py`;
- CTC-v2;
- all R04 configs;
- all R05 configs;
- experiment registry;
- S1/S2/S3 seeds;
- model identities;
- R04/R05 objectives;
- batch/accumulation/effective-batch semantics;
- validation checkpoint-selection semantics;
- protected evaluation surfaces.

No scientific re-lock is required.

## 7. Residual facts intentionally not expanded

### Periodic durable sync

Still true:

- local periodic checkpoint cadence is every 250 optimizer steps;
- Layer-B durable sync is segment-boundary, not every periodic local checkpoint.

QA1 did not add aggressive Kaggle dataset versioning.

After G2 measures runtime/checkpoint/sync cost, a bounded pre-results wall-clock durable cadence may be considered for technical reasons only.

### Physical UUID attribution

Numeric physical slot pinning plus parent T4 UUID inventory and child one-visible-T4 evidence remains the accepted bounded homogeneous-T4 assumption.

No higher-churn UUID pinning mechanism was required for QA1 closure.

## 8. Active operator chronology

Current active sequence is:

```text
smoke-write
→ fresh smoke-restore
→ independent audit
→ dual-gpu-smoke
→ independent audit
→ G1
→ calibration-dual
→ principal P1/P2/P3
```

G1 machine-refuses missing/invalid terminal dual-smoke evidence.

## 9. Claims not made

QA1 repository remediation did **not** run or claim:

- new-source real Smoke A PASS;
- new-source real Smoke B PASS;
- real dual-GPU-smoke PASS;
- G1 PASS;
- G2 PASS;
- R04 launch/result;
- R05 launch/result.

The historical earlier Interactive Smoke A remains diagnostic history only and is not qualification for source `be9b6965...`.

## 10. Closure verdict

Every confirmed S0/S1 defect is closed.

The S2 operator/durability findings are closed or explicitly bounded/non-material.

Stage-03R science is unchanged.

Source and wrapper exact-head CI are green.

Active docs are current.

G1 refuses invalid/missing dual-smoke evidence.

Continuation cannot convert publication failure into retraining or unearned PASS.

A fully completed terminal checkpoint can be recovered without advancing optimizer/scheduler or recomputing model-quality evidence.

# **PASS — 01A-MGPU-QA1 CLOSED; NEW-SOURCE REAL SMOKE A/B MAY BEGIN**
