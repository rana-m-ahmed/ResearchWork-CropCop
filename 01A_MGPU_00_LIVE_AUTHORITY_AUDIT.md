# Stage 01A-MGPU — Prompt 0 Live Authority Audit

- **Artifact:** `01A_MGPU_00_LIVE_AUTHORITY_AUDIT.md`
- **Audit time:** 2026-09-05T20:21:00+05:00 (Asia/Karachi)
- **Repository:** `rana-m-ahmed/ResearchWork-CropCop`
- **Pre-refactor live `main`:** `32190dd86293caa82170df3feea505e3c7443b4b`
- **Pre-refactor PR #8 head:** `8d6ce17b21bfc4ba301c75ece8e149fe9bd06bbb`
- **PR branch:** `je-stage01a-core-20260905`
- **Status:** **AUDIT COMPLETE — EXECUTION-PACKAGING REMEDIATION MAY PROCEED**
- **Code-mutation statement:** no execution/training source was modified while Prompt 0 was performed. This report records the verified pre-refactor state.

## 1. Executive verdict

No Stage-03R scientific-authority conflict was found.

The live defect is execution/evidence packaging, not science:

1. the current terminal Smoke-B consumer in `journal_extension/kaggle/run_lane.py` requires a field the Smoke-B producer does not emit;
2. the existing real Smoke-A run is a genuine dual-T4 diagnostic but reports Kaggle `KAGGLE_KERNEL_RUN_TYPE=Interactive`, so it cannot be the final clean Saved-Version qualification for a new execution source;
3. the current notebook/run-lane architecture is single-child and has no dual-envelope supervisor;
4. child publication is currently possible whenever Git credentials are inherited, so an envelope must strip child credentials and serialize publication in the parent.

These are all remediable above the frozen Stage-03R scientific computation.

## 2. Authority reconstruction

### 2.1 Scientific authority

The accessible canonical Stage-03R authority was independently read and SHA-256 verified:

- file: `03R_EAAI_SCIENTIFIC_DESIGN_LOCK_v2.md`
- authority ID: `EAAI-JE-SDL-v2.1-QA`
- SHA-256: `aab17b65b0873dcb1ecedb061eb02ff60ccb09f8b830184f5e2231a600278f74`
- status in the authority document: LOCKED

The repository lock `journal_extension/locks/scientific_authority.json` independently carries the same authority ID and SHA.

A stale historical companion digest `ec1293d075a7880207539a8f7032ed97434017f4ae209e0f0cb7f1b2b0c65560` exists in the external file library for the earlier pre-QA Stage-03R revision. It is **not** the live authority and must not replace the v2.1-QA hash.

### 2.2 Stage-04 authority

The accessible canonical Stage-04 authority was independently read and SHA-256 verified:

- file: `04_EAAI_REPOSITORY_EXECUTION_BLUEPRINT_v2.md`
- architecture ID: `EAAI-JE-REA-v2.2-LEAN`
- SHA-256: `a1bc1f8bb2d7a133a46c3f59f83dd2ba8ca3bf0fa8d77ae323553276931bc079`
- status: LOCKED — SUPERVISOR-AMENDED LEAN EXECUTION ARCHITECTURE

The repository lock carries the same architecture ID/hash.

Stage-04 explicitly permits paired R04/R05 work to be distributed across independent GPUs/accounts while preserving each individual run as a single-GPU Stage-03R run. Its operational sentence `One Kaggle job = one run ID` is therefore amendable at the packaging layer without changing the scientific unit of analysis.

### 2.3 Stage-00 launch gate

The accessible Stage-00 rapid gate was read in full:

- `00V3_1_EAAI_RAPID_LAUNCH_GATE.md`
- gate: `GO — TRACKS EXTRACTED; START 01A/01B/01C NOW`

It authorizes R04/R05 scientifically while identifying G0/G1/G2 and restricted execution inputs as engineering prerequisites.

### 2.4 Authority-location correction

The Stage-00, Stage-03R and Stage-04 Markdown authority documents are not currently present as files in the PR #8 Git tree. They are external locked authority inputs whose exact hashes are bound into repository lock metadata. This audit independently recomputed the Stage-03R and Stage-04 canonical SHA-256 values from the accessible original bytes. The repository must not pretend those external documents are tree-resident.

## 3. Live repository and PR chronology

### 3.1 Branch relationship

- live `main`: `32190dd86293caa82170df3feea505e3c7443b4b`
- PR #8 base: `main` at the same baseline SHA
- PR #8 pre-refactor head: `8d6ce17b21bfc4ba301c75ece8e149fe9bd06bbb`
- PR state: OPEN / DRAFT
- observed pre-refactor PR size: 28 commits, 79 changed files, 10,178 additions, 3 deletions

### 3.2 Complete pre-refactor PR commit chronology

1. `8044f6f` — Implement lean EAAI JE core for locked R04/R05
2. `9d55acb` — Add Stage 01A implementation and launch report
3. `7c2c4c5` — Record successful Stage 01A CI validation
4. `cd60166` — Harden Stage 01A Kaggle execution and recovery
5. `1a23d18` — Close Stage 01A hardening review defects
6. `b3013fc` — Document Stage 01A-H hardening validation
7. `a7de96d` — Fix canonical notebook JSON serialization
8. `c1a640d` — Reconcile Stage 01A-H validation contracts
9. `ac39ebd` — Verify Stage 01A-H CI closure
10. `78089ec` — Close Stage 01A-P pre-G1 integrity gaps
11. `5897b53` — Document Stage 01A-P pre-G1 launch gate
12. `d775a4d` — Fix Stage 01A-P durable recovery validation
13. `731fcf1` — Fix Stage 01A-P integrity validation
14. `e796a59` — Implement API-free cross-session smoke
15. `045fcf5` — Align JE validator with smoke state machine
16. `112c108` — Bind canonical smoke notebook to execution source
17. `f61f7df` — Fix canonical serialization regression
18. `0bfe493` — Make notebook static validation fail closed
19. `c5b8eda` — Align notebook tests with clean source
20. `2602384` — Harden Kaggle GitHub auth preflight
21. `f12c016` — Regenerate canonical notebook
22. `1321868` — Align notebook validation with auth hardening
23. `a48d956` — Normalize tracked LF blobs
24. `939455c` — Test fresh exact-head checkout
25. `0b35321` — Bind Kaggle smoke notebook to clean execution source
26. `f735ecf` — Harden public evidence publication
27. `6737014` — Fix publication regression and freeze hardened source
28. `8d6ce17` — Bind Kaggle smoke notebook to hardened publication source

### 3.3 Complete observed PR CI history before this refactor

| Actions run # | head | result |
|---:|---|---|
| 23 | `8044f6f` | PASS |
| 24 | `9d55acb` | PASS |
| 25 | `7c2c4c5` | PASS |
| 26 | `c1a640d` | PASS |
| 27 | `ac39ebd` | PASS |
| 28 | `5897b53` | FAIL |
| 29 | `d775a4d` | FAIL |
| 30 | `731fcf1` | PASS |
| 31 | `e796a59` | FAIL |
| 32 | `045fcf5` | PASS |
| 33 | `112c108` | PASS |
| 34 | `f61f7df` | FAIL |
| 35 | `0bfe493` | FAIL |
| 36 | `c5b8eda` | PASS |
| 37 | `f12c016` | FAIL |
| 38 | `1321868` | PASS |
| 39 | `a48d956` | PASS |
| 40 | `939455c` | PASS |
| 41 | `0b35321` | PASS |
| 42 | `f735ecf` | FAIL |
| 43 | `6737014` | PASS |
| 44 | `8d6ce17` | PASS |

Actions run #44 independently showed expected SHA = actual SHA = `8d6ce17b21bfc4ba301c75ece8e149fe9bd06bbb`, successful compile, strict repository validation, JE static validation, and **127/127 CPU-safe tests PASS**.

Failed intermediate CI is retained in history and is not to be rewritten.

## 4. Current source/wrapper relationship

The pre-refactor execution source is:

`67370145c9104edd52330b788c3b41b28f5cab87`

It was exact-head validated by Actions run #43 with 127/127 tests.

The pre-refactor PR head/wrapper is:

`8d6ce17b21bfc4ba301c75ece8e149fe9bd06bbb`

Its parent is exactly `67370145...`. Commit `8d6ce17...` binds the canonical notebook/generator and related wrapper validation/report text to the already-verified execution source. It is therefore **not** the execution-source SHA. This is the existing anti-self-reference pattern that the MGPU refactor must preserve.

Because the MGPU remediation changes execution source, both `67370145...` and `8d6ce17...` become historical once a new exact-head source is frozen.

## 5. Exact locked scientific semantics

The live Stage-03R/CTC/config/registry state agrees on:

- micro-batch: **16**
- gradient accumulation: **4**
- effective batch: **64**
- epochs: **30**
- optimizer: AdamW
- mixed precision: **FP16 autocast + GradScaler**
- gradient clip norm: **1.0**
- EMA: **off**
- `drop_last`: **false**
- sampling: uniform row shuffle with deterministic row/epoch augmentation
- early stopping: disabled
- checkpoint selection order:
  1. validation macro-F1 descending
  2. validation balanced accuracy descending
  3. validation NLL ascending
  4. epoch ascending
- student: `mobilenetv4_conv_medium.e500_r256_in1k`, `timm==1.0.26`
- teacher checkpoint SHA-256: `74b4701b8931976c9227845ead50788ae47e3596f575f2817b7352a715f53b79`
- S1 seed: `21270083`
- S2 seed: `606135704`
- S3 seed: `1153870846`
- pair IDs: `MNV4-PAIR-S1`, `MNV4-PAIR-S2`, `MNV4-PAIR-S3`
- R04 objective: CE=1.0, KD=0.0, feature=0.0
- R05 objective: CE=0.5, KD=0.35, feature=0.15
- R04/R05 allowed surfaces: `DS-V1-TRAIN`, `DS-V1-VAL`
- V1 test and sealed external surfaces forbidden during these runs

No evidence observed in this audit justifies altering any of those objects.

## 6. Scientifically sensitive versus orchestration-only source

### 6.1 Scientific-computation / scientific-contract sensitive

These must remain byte-identical unless a re-lock is justified:

- `journal_extension/src/cropcop_je/train.py`
- `journal_extension/src/cropcop_je/data.py`
- `journal_extension/src/cropcop_je/models.py`
- `journal_extension/src/cropcop_je/evaluate.py`
- `journal_extension/src/cropcop_je/selection.py`
- `journal_extension/src/cropcop_je/surfaces.py`
- `journal_extension/configs/common/ctc_v2.json`
- all `journal_extension/configs/r04_direct/*.json`
- all `journal_extension/configs/r05_teacher/*.json`
- seed/experiment scientific content in `journal_extension/locks/experiment_registry.json`
- Stage-03R authority ID/hash

### 6.2 Scientific-control sensitive but execution-remediable

These protect identity/chronology but do not define model math:

- G1 validation/sealing
- G2 barrier validation
- checkpoint/recovery identity validation
- durable persistence
- protected-surface gates
- smoke handoff validation
- source-state validation

Changes here are allowed only to make an existing frozen contract fail-closed; they must not alter the frozen scientific objects.

### 6.3 Orchestration-only

- `journal_extension/kaggle/run_lane.py`
- canonical notebook/generator
- clean-session bootstrap
- session-budget plumbing
- environment capture
- Git publication orchestration
- future envelope supervisor/configs
- execution worker-count plumbing
- technical telemetry/logging
- public-safe evidence routing

## 7. Real Smoke-A evidence verification

The branch still exists:

`run-evidence/SMOKE-A-INFRA-SMOKE-67370145c910-450c18397a20`

Observed terminal public evidence commit:

`b906295661991747d1e50f504d4dc0fb6ba98980`

Verified evidence facts:

- status: PASS
- source SHA: `67370145c9104edd52330b788c3b41b28f5cab87`
- dependency lock: `767859a3cb1b37c0e50f753aa8c7a5c400424417cba3eaa59f0ab617ff18f701`
- scientific: false
- synthetic-only: true
- G1/G2/R04/R05 executed: false
- Python: 3.12.13
- torch: 2.12.1
- torchvision: 0.27.1
- timm: 1.0.26
- CUDA available: true
- CUDA device count: 2
- GPU0: Tesla T4, UUID `GPU-17939234-0ba2-76a9-9f7b-0de270a853e8`
- GPU1: Tesla T4, UUID `GPU-669cf7b6-d46e-8f66-b123-104a19baf264`
- driver: 580.159.04
- CUDA runtime: 13.0

**Exact run-mode finding: `KAGGLE_KERNEL_RUN_TYPE = Interactive`.**

Therefore this evidence is retained as historical real-host diagnostic + dual-T4 hardware proof, but it is not accepted as the terminal post-refactor clean Saved-Version qualification.

## 8. Producer/consumer evidence map

| Producer | Evidence | Current consumers | Audit result |
|---|---|---|---|
| `smoke_infrastructure.py` write mode | Smoke-A evidence + manifest + content-addressed checkpoint bundle | Smoke-B verifier | internally coherent; current real A is Interactive |
| `smoke_infrastructure.py` restore mode | Smoke-B evidence + manifest + resumed checkpoint | `run_lane.validate_smoke`, `seal_g1.py`, `cropcop_je.g1.validate_mounted_g1`, G1 barrier path | **contract mismatch exists** |
| Smoke-B evidence publisher | `git_publication_status`, public evidence branch | later audit/G1 chain | producer supports PASS/FAIL publication state |
| pretrained provenance capture | `MNV4_PRETRAINED_PROVENANCE.json` | G1 seal, mounted G1 validation | bound to exact pretrained bytes/config |
| teacher factory capture | `TEACHER_FACTORY_BUNDLE.json` | G1 seal, mounted G1 validation | bound to source bundle |
| teacher class-order verifier | `TEACHER_CLASS_ORDER_EVIDENCE.json` | G1 seal, mounted G1 validation | shape-only inference forbidden |
| `seal_g1.py` | `G1_MODEL_IDENTITY_SEAL.json`, pair-init evidence/private bytes | G1 barrier, calibration, principal trainer | single global immutable G1 |
| calibration scripts | `CAL-MNV4-DIRECT.json`, `CAL-MNV4-TEACHER.json`, `CAL-CNXTT.json` summaries | G2 barrier validator | scheduling-only; fixed step counts |
| `validate_g2_barrier.py` | `G2_CALIBRATION_BARRIER.json` | principal `run_training.py` | single final barrier required |
| `run_training.py` / `train.py` | checkpoints, run record, metrics, segments | resume, persistence, public-safe evidence | scientific checkpoint identity excludes physical GPU slot |
| persistence backend | durable checkpoint generations | later Saved-Version resume | production Kaggle private dataset path retained |
| publication module | small `run-evidence/<run_id>` branches | G2 collection / independent audit | currently callable by child if token inherited |

### Confirmed Smoke-B defect

Current `run_lane.validate_smoke()` requires:

`secret_retrieval_proved_without_value_disclosure == true`

The current Smoke-B producer does **not** emit that field. The producer instead emits the actual restore/recover/resume chain, exact A/B hashes, optimizer-step advancement, input immutability, non-scientific flags, and publication state.

This is a genuine producer/consumer mismatch. The field must not be fabricated. One canonical terminal Smoke-B validator must replace all ad-hoc consumers.

## 9. Current checkpoint/resume/persistence state

The existing scientific trainer already has:

- content-addressed checkpoint objects;
- latest/previous/selected recovery generations;
- checkpoint identity hashing;
- optimizer/scheduler/GradScaler/RNG restoration;
- mid-epoch cursor/data-order state;
- selection-state persistence;
- exact identity mismatch rejection;
- notebook-global session rollover;
- Layer-A local recovery plus durable store;
- production `KagglePrivateDatasetStore` path.

The current principal checkpoint scientific identity contains experiment/config/data/model/seed/G1/G2/lane identities but **not physical GPU index/UUID**. That is correct and should be preserved.

Production durability still requires Kaggle API credentials and a unique per-run locator. The MGPU refactor must not silently replace it with ephemeral `/kaggle/working`.

## 10. Current notebook/execution state

The canonical generator/notebook currently supports explicit:

- `smoke-write`
- `smoke-restore`
- `g1`
- `calibration`
- `principal`

but it does not yet provide:

- `dual-gpu-smoke`
- `calibration-dual`
- `principal-dual`
- envelope selection
- envelope continuation input
- fail-closed Batch qualification for every terminal phase
- dual-child isolation

The global monotonic clock is correctly established before clone/install and is inherited by subprocesses.

## 11. Historical multi-GPU references

The exact archived v5/v7.1 notebook files were not directly available in the live PR tree or discoverable as standalone notebook files in the accessible file library during this audit. Therefore their code was **not re-certified directly in Prompt 0**.

The amendment's descriptions of historical DataParallel and DDP systems remain advisory historical context only. The current architecture decision is made from the frozen JE semantics and the live trainer, so inability to re-open those historical notebooks does not block the MGPU decision.

## 12. Stale or qualified assumptions in the proposed amendment

1. **PR head** `8d6ce17...` was correct at the start of this audit; it becomes a historical pre-refactor head as soon as audit/implementation commits are added.
2. **main** `32190dd...` remains correct and unchanged at audit time.
3. **Smoke-A source** `67370145...` is correct.
4. **Smoke-A mode Interactive** is independently confirmed.
5. **dual T4 hardware** is independently confirmed from the actual public evidence branch.
6. **Smoke producer/consumer mismatch** is independently confirmed.
7. **Stage-03R and Stage-04 hashes** in the amendment are independently confirmed from canonical accessible bytes.
8. The amendment should not imply Stage-03R/Stage-04 Markdown files are repository-resident; they are externally locked, hash-bound authority inputs.
9. Historical v5/v7.1 notebook internals were not directly re-opened in this audit; any historical implementation detail from those notebooks is non-authoritative.
10. Any quoted future source SHA, test count, or wrapper SHA in the amendment is necessarily provisional and must be replaced by live post-refactor evidence.

## 13. Prompt-0 gate

`GO — NO SCIENTIFIC RE-LOCK REQUIRED; PROCEED TO SCIENCE-DIFF SENTINEL AND ADVERSARIAL ARCHITECTURE REVIEW`

The remediation must remain above the frozen scientific trainer/configuration and must fail closed if later work reveals scientific drift.
