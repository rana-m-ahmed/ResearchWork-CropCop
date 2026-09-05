# CropCop → EAAI Journal Extension
## Stage 01A-H — Kaggle Execution Hardening Report

- **Stage / continuation:** Stage 01A-H — final execution/recovery hardening before real G1/G2
- **Execution date/time:** 2026-09-05T15:49:06+05:00 (Asia/Karachi)
- **Repository:** `rana-m-ahmed/ResearchWork-CropCop`
- **Branch:** `je-stage01a-core-20260905`
- **PR:** #8
- **Current `main` HEAD:** `32190dd86293caa82170df3feea505e3c7443b4b`
- **Pre-hardening branch HEAD:** `7c2c4c54fb00eb465629588ac3152d3b4bc0bbcb`
- **Hardening implementation commit:** `1a23d18f64bfb0b42b447cb9940bae2b57253ffd`
- **Scientific authority:** `03R_EAAI_SCIENTIFIC_DESIGN_LOCK_v2.md` / `EAAI-JE-SDL-v2.1-QA`
- **Scientific authority SHA-256:** `aab17b65b0873dcb1ecedb061eb02ff60ccb09f8b830184f5e2231a600278f74`
- **Execution authority:** `04_EAAI_REPOSITORY_EXECUTION_BLUEPRINT_v2.md` / `EAAI-JE-REA-v2.2-LEAN`
- **Execution authority SHA-256:** `a1bc1f8bb2d7a133a46c3f59f83dd2ba8ca3bf0fa8d77ae323553276931bc079`
- **Status at report commit:** implementation complete; exact-head GitHub Actions verification required before GO
- **Historical V1 mutation statement:** no historical V1 artifact was modified; V1 test was not accessed.

## 1. Scope and scientific-lock preservation

This continuation changed execution/reliability machinery only. It did **not** change:

- R04/R05 definitions;
- S1/S2/S3 seeds;
- paired initialization semantics;
- CTC-v2;
- objectives;
- 30 epochs;
- optimizer/schedule/effective-batch semantics;
- validation checkpoint-selection rule;
- historical teacher scientific identity;
- V1 dataset identities;
- V1-test firewall;
- protected external boundaries.

No low, null, favorable, or unfavorable model result existed during this work and no result-dependent
branching was possible.

## 2. Issue ledger

| Severity | Finding | Resolution |
|---|---|---|
| S0 | Resume reset validation history/best state, allowing a post-interruption checkpoint to replace an earlier true best | Persist `SelectionState` history/best in checkpoints; verify selected checkpoint during recovery |
| S0 | Teacher feature-dimension probing could mutate student BatchNorm state and consume the student RNG only in R05 | Probe student/teacher in eval + no-grad; restore original modes; initialize projection inside isolated RNG fork |
| S0 | Persistent DataLoader workers could retain a stale mutable dataset epoch | Sampler now carries `(row_index, epoch)` so each worker receives the exact augmentation epoch explicitly |
| S0 | In-place/latest-only checkpoint semantics could lose the newest valid state after corruption/interruption | Atomic content-addressed objects plus generation-aware `latest/selected/previous` recovery and fallback |
| S1 | Resume identity omitted execution-source/lane/software/factory dimensions | Bind source Git SHA, lane, software-stack hash, teacher-factory source hash and all prior model/data/config identities |
| S1 | Checkpoint index and payload progress could drift silently | Recompute identity digest and verify index ↔ payload epoch/batch/optimizer-step consistency |
| S1 | JSON/state writes and output directories lacked a single fail-closed ownership path | Atomic fsync+replace writes and run-directory ownership claim |
| S1 | Kaggle interruption could leave claim-producing recovery only on ephemeral storage | Layer-A local checkpoint bundle plus required pluggable Layer-B durable store for scientific lanes |
| S1 | Calibration resume timing could contaminate scheduling telemetry | Separate save→resume qualification from measured calibration; record checkpoint load/save separately |
| S1 | Three-account calibration could race into principal science before all G2 evidence was visible | Persist three calibration summaries and require one hash-bound G2 barrier; recover summaries from evidence branches |
| S1 | Small Git evidence publication could accidentally expose secrets/private paths/restricted file types | Explicit text-file allowlist, size cap, sensitive-pattern scan, isolated `run-evidence/<run_id>` branches, no main push |
| S1 | Lane files contained apparent principal run IDs before a real launch | Removed static principal run IDs; resolve deterministic/explicit IDs only when the real lane executes |
| S2 | Hardware/software/runtime evidence was too shallow for later forensic reproduction | Capture package family, CUDA/cuDNN, GPU/driver identity and safe Kaggle context |
| S2 | Validation/dataloader costs were not represented in scheduling evidence | Record dataloader wait/throughput and bounded validation forward throughput |
| S2 | Multiple notebooks would risk logic drift | One canonical thin notebook delegates to repository code; only K1/K2/K3 JSON lane specs differ |
| S3 | Durable backend and external credentials still require real Kaggle configuration | Intentionally deferred to G1/G2 execution; no fake remote store was created |

No unresolved S0/S1 defect is intentionally accepted by this report. The exact pushed snapshot must
still prove that claim through CI.

## 3. Checkpoint and recovery architecture

A scientific checkpoint is written to a temporary file, flushed, fsynced, SHA-256 hashed, loaded back
for structural/scientific identity validation, and only then atomically renamed to a content-addressed
object. A small atomic `checkpoint_index.json` tracks:

- newest `latest`;
- previous valid `latest`;
- current `selected`;
- monotonic generation.

Recovery evaluates indexed candidates by generation and accepts only a candidate whose bytes, SHA,
scientific identity, identity digest, and epoch/batch/optimizer-step metadata validate.

The checkpoint payload persists model, optional projection, optimizer, scheduler, GradScaler, Python /
NumPy / PyTorch / CUDA RNG state, exact epoch/batch/optimizer-step position, examples seen, data-order
state, validation history, and current best-selection state.

## 4. Resume semantics

Resume is a continuation only when all bound identities match. The identity includes:

- experiment and scientific authority;
- source Git commit;
- resolved config and CTC-v2 hashes;
- V1 manifest and class-map hashes;
- seed;
- paired student-init SHA;
- pretrained SHA;
- teacher SHA when applicable;
- teacher-factory source hash when applicable;
- software-stack hash;
- lane ID.

A mismatch fails closed. A technically failed run that requires a genuine retry must receive a new
runtime run ID/attempt rather than silently changing the old run's science.

Mid-epoch continuation starts from the exact next batch on the same deterministic epoch permutation.
Checkpoints are written only at optimizer-update boundaries, so no unpersisted partial accumulation is
treated as completed progress.

## 5. Segment and session semantics

Every execution session gets a unique segment ID and append-only JSONL START/END events containing
parent run identity, source SHA, host/GPU evidence, input artifact hashes, ending checkpoint SHA,
optimizer step and termination reason.

The default session controller models a 12-hour hard limit with a one-hour finalization margin. A lane
can deliberately stop before the unsafe tail, write a verified recovery checkpoint, persist Layer-B,
mark continuation required, and exit without pretending the scientific run is terminal.

## 6. Durable persistence

Layer A is the local content-addressed checkpoint/index bundle.

Layer B is pluggable:

- `filesystem` for an independently durable mounted destination; or
- `kaggle-dataset` for a pre-created private Kaggle dataset version.

A scientific lane can require Layer-B persistence. Durable sync verifies copied hashes. Sync failure is
a technical failure and never becomes a successful scientific terminal state.

## 7. Calibration and G2 barrier

Scheduling-only calibrations remain scientifically disposable:

- `CAL-MNV4-DIRECT`: 200 optimizer steps;
- `CAL-MNV4-TEACHER`: 100 optimizer steps;
- `CAL-CNXTT`: 100 optimizer steps.

Save→resume qualification is separated from measured calibration. Summaries record accelerator,
seconds/optimizer-step, examples/s, dataloader wait/throughput, peak GPU memory, checkpoint save/load
cost and validation forward throughput.

Principal R04/R05 execution cannot pass the G2 barrier until all three summaries:

1. are terminal PASS;
2. prove save→resume success;
3. are marked non-scientific calibration;
4. share the same source Git SHA;
5. share the same locked software-stack identity;
6. provide the required telemetry.

Forecasts are scheduling-only and cannot change the scientific design.

## 8. Cross-lane and canonical Kaggle architecture

There is exactly one notebook source:

`journal_extension/kaggle/canonical_lane.ipynb`

It delegates all logic to:

`journal_extension/kaggle/run_lane.py`

K1/K2/K3 differ only through small JSON lane specifications. Principal run IDs are resolved only at
real execution time, using an explicit environment override when supplied or a deterministic
experiment/source-SHA/attempt identity. Repository lane files do not claim that a real run already
exists.

Calibration summaries can be published as audited small evidence and recovered from
`run-evidence/<calibration_id>` branches across independent Kaggle accounts. If Git credentials are
not configured, the runner does not fake success; central/manual evidence integration is then required
before G2 can open.

## 9. Git evidence boundary

Only explicitly supplied small text evidence may be published. The publisher rejects:

- checkpoint/model/runtime binary suffixes;
- unsupported suffixes;
- large files;
- suspicious restricted filenames;
- GitHub/Kaggle credentials and generic secret-like values;
- common private Kaggle/Windows absolute paths.

The publisher writes only to `run-evidence/<run_id>`; it never pushes directly to `main`.
Publication failure is persisted as a separate technical evidence-chain failure and does not alter a
scientific metric.

## 10. Local validation

The surviving local hardening workspace was **not a full Git checkout** of the repository, so Git
staged/unstaged status and repository-wide strict validation cannot honestly be reported as local Git
results.

What was reproducible in that workspace after the final S0 assertions were added:

`python -m compileall -q cropcop_je scripts kaggle tests calibrate.py calibrate_cnxtt.py run_training.py`

→ **PASS**

`PYTHONPATH=. python -m unittest discover -s tests -v`

→ **32 / 32 PASS** in the hardening/failure-injection harness.

Those 32 tests include the required interruption/resume equivalence, corrupted-newest fallback,
config/manifest/pretrained/teacher/source/seed/lane drift, output collision, V1-test and protected
external access attacks, absent Git secret, Git push failure, credential leakage, durable-store
failure, low-disk failure, planned session rollover, persistent-worker epoch propagation and
teacher-state/RNG isolation.

Repository-wide strict validation, JE static validation, original repository tests and exact hardened
test discovery are therefore delegated to the mandatory GitHub Actions run on the exact pushed PR
head; they are not falsely labeled as local results.

## 11. CI evidence design

The workflow on the final PR head performs, in order:

1. Python compile validation;
2. strict repository validator;
3. JE static validator;
4. complete `unittest discover -s tests -v` suite with a persisted log;
5. machine-readable exact-head CI context;
6. artifact upload of repository/JE validator reports, unit-test log and CI context.

The exact runtime run ID/run number/final conclusion cannot be embedded into the same Git commit it
verifies without changing that commit's SHA. To avoid a self-referential false record, the workflow
emits `ci-verification.json` as an Actions artifact, while PR #8 metadata and the Stage-01A-H final gate
record the exact run ID, run number, commit SHA and conclusion after completion.

The pre-hardening successful run is not accepted as Stage-01A-H evidence.

## 12. Remaining real G1/G2 prerequisites

Hardening does not manufacture the next-stage inputs. Real execution still requires:

1. canonical restricted V1 manifest bytes matching
   `bdb82211ccc2059153724eea178a1680893a6b38ecc243fae484baa91dbf68e2`;
2. V1 train/validation image bytes/root;
3. frozen 120-way class-map bytes matching
   `46f7811726c19c42bd7213b2d8178b19a5a182a1b763f60a94ee2c0e5f6688d2`;
4. exact MobileNetV4 ImageNet pretrained bytes for the locked `timm==1.0.26` object;
5. exact historical DINO teacher bytes matching
   `74b4701b8931976c9227845ead50788ae47e3596f575f2817b7352a715f53b79`;
6. the trusted historical-teacher loader/factory;
7. exact ConvNeXt-Tiny `torchvision==0.27.1 / IMAGENET1K_V1` pretrained bytes for CAL-CNXTT;
8. a qualified Kaggle CUDA host capable of the locked software stack;
9. configured durable-store destination/credentials before scientific lanes;
10. optional Git evidence credential if autonomous cross-account evidence-branch synchronization is desired.

## 13. No fabricated execution

As of this report:

- no real G1 verification has been claimed;
- no direct/teacher/ConvNeXt GPU calibration has been launched;
- no R04 or R05 run has been launched;
- no GPU timing/quota measurement exists;
- no scientific checkpoint hash exists;
- no Kaggle notebook/job ID exists;
- no principal scientific run ID is claimed as launched;
- no V1 test was opened;
- no protected external prediction was performed.

Calibration identifiers appearing in code are orchestration definitions, not empirical evidence.

## 14. Gate rule

This report does **not** issue GO by itself.

GO is authorized only after PR #8 points to the final documentation snapshot, GitHub Actions executes
the complete hardened validation workflow on that exact head, the run concludes successfully, test
discovery includes the hardened suites, the artifact/secret boundary remains clean, and no S0/S1
defect is found in the final reconciliation.

Until then the operative state is:

**AWAITING EXACT-HEAD CI VERIFICATION — NO REAL G1/G2 LAUNCH AUTHORIZED BY THIS REPORT.**
