# 01A — EAAI JE Core Implementation, Calibration and R04/R05 Launch Report

- **Stage / pack:** Stage 01A / Operational Prompt Pack v3.1
- **Execution date/time:** 2026-09-05, Asia/Karachi
- **Current repository `main` HEAD:** `32190dd86293caa82170df3feea505e3c7443b4b`
- **Implementation branch:** `je-stage01a-core-20260905`
- **Implementation commit:** `8044f6fb7c1c99f5f5bd211586d7bfd8f68fd32b`
- **Draft PR:** #8 — `Stage 01A: lean EAAI JE core for R04/R05`
- **Scientific authority:** `03R_EAAI_SCIENTIFIC_DESIGN_LOCK_v2.md` v2.1-QA
- **Scientific-authority SHA-256:** `aab17b65b0873dcb1ecedb061eb02ff60ccb09f8b830184f5e2231a600278f74`
- **Execution authority:** `04_EAAI_REPOSITORY_EXECUTION_BLUEPRINT_v2.md` v2.2-LEAN
- **Execution-authority SHA-256:** `a1bc1f8bb2d7a133a46c3f59f83dd2ba8ca3bf0fa8d77ae323553276931bc079`
- **Prior-stage input:** `00V3_1_EAAI_RAPID_LAUNCH_GATE.md` — `GO — TRACKS EXTRACTED; START 01A/01B/01C NOW`
- **EAAI/Elsevier policy verification:** not used/re-run in Stage 01A; Stage 00 authority is inherited
- **Status:** `BLOCKED_TECHNICAL_AFTER_G0`
- **Historical V1 mutation statement:** no historical V1 artifact was modified and no V1-test access was performed.

## 1. Repository implementation completed

The connected repository was modified on an isolated branch, not directly on `main`.

The implementation commit is exactly one commit ahead of the Stage-00 baseline and zero commits behind at the verification point. The diff contains 35 changed files.

Implemented objects include:

- JE scientific-authority identity;
- Track-A/B/C/D launch matrix;
- R04/R05 experiment registry;
- exact three paired training seeds;
- frozen CTC-v2 configuration;
- exact direct and full-teacher objectives;
- V1 train/validation-only surface bindings;
- deny-by-default protected-surface guard;
- exact-byte SHA-256 verification hooks;
- per-seed paired student-initialization creation and evidence;
- fail-closed exact historical-teacher loading through a restricted-environment factory;
- deterministic epoch permutation;
- deterministic per-row augmentation seeding;
- trainer and validation evaluator;
- validation-only checkpoint-selection ordering;
- checkpoint/resume with scientific-identity mismatch rejection;
- calibration entrypoint with save→resume qualification;
- run JSON evidence records;
- terminal-run ingestion;
- prelaunch validator;
- public-Git binary/secret guard integration;
- JE static CI validation;
- JE contract tests.

## 2. Locked principal experiment bindings

The implementation binds the principal experiment identities exactly as follows:

| Pair | Direct | Teacher | Seed |
|---|---|---|---:|
| S1 | `R04-MNV4-DIRECT-S1` | `R05-MNV4-TEACHER-S1` | 21270083 |
| S2 | `R04-MNV4-DIRECT-S2` | `R05-MNV4-TEACHER-S2` | 606135704 |
| S3 | `R04-MNV4-DIRECT-S3` | `R05-MNV4-TEACHER-S3` | 1153870846 |

Each pair is required to consume the same serialized student-initialization evidence object before either condition can launch.

No scientific run ID has been issued yet because no scientific/calibration run was actually launched. Inventing run IDs before launch would violate the evidence contract.

## 3. Protected-data controls

The training entrypoint authorizes only:

- `DS-V1-TRAIN`
- `DS-V1-VAL`

It rejects:

- `DS-V1-TEST-CONSUMED`
- sealed external surfaces;
- historical comparison surfaces;
- device-only surfaces;
- unregistered surfaces.

The manifest loader verifies the expected restricted-manifest SHA-256 before resolving rows and checks the frozen train/validation counts.

The V1 test was not accessed in this stage.

## 4. Model-identity controls

### Student

Locked object:

`timm==1.0.26 / mobilenetv4_conv_medium.e500_r256_in1k`

The implementation requires the exact pretrained object as external bytes, records its SHA-256 at G1, loads those bytes into the canonical 1000-way timm object, and only then creates the prospectively seeded 120-way head.

The pretrained SHA-256 is intentionally **not** filled with an invented value.

### Teacher

Locked historical teacher SHA-256:

`74b4701b8931976c9227845ead50788ae47e3596f575f2817b7352a715f53b79`

The public JE core verifies that exact hash and then requires a trusted restricted-environment teacher factory. It does not guess that the historical checkpoint can be reconstructed with a generic ConvNeXt constructor.

## 5. Validation evidence

### Pre-commit assembled implementation snapshot

The authored Stage-01A snapshot was Python-compiled successfully and its JE-specific contract suite returned:

`9 tests / 9 PASS`

The static JE validator returned `PASS` with no errors before repository publication.

The tested controls included:

- exact R04/R05 IDs and seeds;
- same-pair binding;
- CTC-v2 lock values;
- protected-surface rejection;
- restricted-manifest filtering;
- deterministic per-row augmentation seed;
- resume-identity drift rejection;
- V1-test rejection in run records.

### Repository publication verification

GitHub comparison after publication reported:

- base: `32190dd86293caa82170df3feea505e3c7443b4b`
- implementation head: `8044f6fb7c1c99f5f5bd211586d7bfd8f68fd32b`
- ahead by: 1
- behind by: 0
- changed files: 35

Draft PR #8 was created so repository CI can validate the committed branch.

GitHub Actions subsequently executed the committed branch snapshot as **Validate public evidence — run #24** on PR #8. The workflow completed with **conclusion: success** against branch head `9d55acb8b229df702f8048ef6be10c5d5fe2cf84`. Therefore the repository validator, JE static validator, and unit-test workflow are recorded as **PASS** for the committed snapshot.

## 6. Calibration status

Direct/teacher scientific calibration was **not executed**.

The currently available local execution host is not a qualified Stage-03R scientific host:

- Python observed: 3.13.5
- PyTorch observed: 2.10.0+cpu
- torchvision observed: 0.25.0+cpu
- CUDA: unavailable
- `timm`: unavailable in the host used for repository-side validation

The locked scientific environment requires:

- Python 3.12.13
- PyTorch 2.12.1
- torchvision 0.27.1
- timm 1.0.26
- qualified CUDA/FP16 execution

Consequently, no accelerator throughput, sec/optimizer-step, peak GPU memory, dataloader throughput, or real checkpoint-save timing is reported. Supplying synthetic measurements would violate the prompt.

## 7. Exact remaining G1/G2 blockers

The following minimum Track-A prerequisites are not accessible through the current connected execution surface:

1. canonical restricted V1 manifest bytes matching `bdb82211ccc2059153724eea178a1680893a6b38ecc243fae484baa91dbf68e2`;
2. V1 train/validation image bytes/root;
3. exact MobileNetV4 pretrained bytes for G1 hashing/verification;
4. historical DINO teacher checkpoint bytes matching `74b4701b8931976c9227845ead50788ae47e3596f575f2817b7352a715f53b79`;
5. the trusted loader/factory for that historical teacher checkpoint;
6. a qualified Kaggle GPU execution surface using the locked software stack.

There is no connected Kaggle execution tool in the current environment, so I cannot truthfully launch K1/K2/K3 from this session.

These are execution-environment / restricted-artifact blockers. They are **not** scientific-design blockers and do not justify changing R04/R05.

## 8. Principal launch state

| Lane | Required chain | State |
|---|---|---|
| K1 | R04-S1 → R05-S1 | NOT LAUNCHED — G1/G2 prerequisites unavailable |
| K2 | R04-S2 → R05-S2 | NOT LAUNCHED — G1/G2 prerequisites unavailable |
| K3 | R04-S3 → R05-S3 | NOT LAUNCHED — G1/G2 prerequisites unavailable |

No result, timing, checkpoint, or run ID has been fabricated.

## 9. Evidence destinations established

Public-safe evidence is designed to persist under:

- `journal_extension/runs/`
- `journal_extension/evidence/`
- authority/config registries under `journal_extension/locks/`

Restricted/large artifacts remain outside public Git and are represented by durable locators, SHA-256, byte size, and producing run identity once real execution occurs.

## 10. Stage decision

The minimum repository-side JE core and G0 controls have been implemented, but G1 model/data identity execution and G2 GPU calibration cannot be completed from the currently connected environment.

**TECHNICAL BLOCKER — PRINCIPAL R04/R05 RUNS NOT LAUNCHED: exact restricted V1/model artifacts and a qualified Kaggle CUDA execution surface are unavailable to this session.**
