# Track-A v1.2 Kaggle Master Operator v6

**Scientific source (immutable):** `9a72e9466a9a3e7429e0e36a028edac662f83146`  
**Distribution branch:** `ops-tracka-kaggle-master-v6-20260915`  
**Pinned master runtime:** `208656f895af5c218a0998b582f2adbb81167eaa`  
**Immutable runtime branch:** `ops-tracka-kaggle-master-runtime-v6-208656f`  
**Driver:** `master_account_driver_v6.py`  
**Authority:** `EAAI-JE-SDL-v2.1-QA`

This branch contains operator/distribution infrastructure only. The frozen scientific source is never modified by the master operator.

## Canonical interface

Normal operation uses exactly three notebooks:

- `TRACKA_V12_MASTER_K1.ipynb`
- `TRACKA_V12_MASTER_K2.ipynb`
- `TRACKA_V12_MASTER_K3.ipynb`

Each notebook is a thin orchestration shell. Every invocation clones the immutable v6 runtime into a **process-isolated** `/kaggle/working` directory, verifies exact HEAD `208656f...` and a clean worktree, then invokes `master_account_driver_v6.py`. The driver separately checks out scientific source `9a72e946...` detached and verifies it remains clean.

The prior v5 distribution/runtime is rollback/history only for new execution.

## Proven Kaggle input layout

The v6 notebooks prefer the mount layout proven by the real K1 runs:

- V1 root: `/kaggle/input/datasets/ranamuhammadahmed6/cropcop-finalized-v8-11-2026-1/CropCop_Final_v1`
- manifest: `audit/final_manifest.csv`
- class map: `audit/class_to_idx.json`
- image root: `dataset`
- K1 historical principal G1: `/kaggle/input/datasets/ranamuhammadahmed6/cropcop-g1-sealed/G1_PACKAGE`

Preferred paths are **not trusted by path name**. The runtime verifies the frozen manifest/class-map SHA-256 values, TRAIN/VAL image structure, and principal-G1 seal. If Kaggle changes the mount prefix, bounded discovery is used instead. Protected V1-test images are not opened for root qualification.

Required identities:

- manifest SHA256: `bdb82211ccc2059153724eea178a1680893a6b38ecc243fae484baa91dbf68e2`
- class map SHA256: `46f7811726c19c42bd7213b2d8178b19a5a182a1b763f60a94ee2c0e5f6688d2`
- historical principal G1 seal SHA256: `442d9e7708749efedcb82ef9f4fd131211549770eecad985177eaaf4117052cd`

## TorchVision provenance and frozen G1A compatibility

For EFFB0 and ConvNeXt-Tiny the operator uses the frozen two-stage provenance sequence:

1. `prepare_torchvision_pretrained.py` downloads and verifies the official artifact bytes.
2. `capture_torchvision_pretrained_provenance.py` performs exact key/shape/dtype/tensor comparison against the frozen official TorchVision enum and writes the complete tensor-identity record.
3. Frozen `validate_torchvision_provenance()` must PASS before G1A sealing.

The actual K1 v5 run proved both provenance records had `official_tensor_match=true`, then exposed a separate defect in the frozen G1A sealer: `seal_tracka_v12_g1a.py` references `TORCHVISION_VERSION` without importing or defining it.

v6 does **not** edit the scientific source. Static symbol analysis verifies that `TORCHVISION_VERSION` is the sealer's only unresolved global. The operator launches that unchanged frozen script through `runpy.run_path(..., init_globals=...)` and injects the exact constant from frozen `cropcop_je.secondary.TORCHVISION_VERSION` (`0.27.1`). A real subprocess compatibility test is part of CI. The detached science checkout is rechecked clean immediately after G1A.

## Duplicate-execution protection

The latest K1 log also showed duplicate stage execution in one Kaggle session, including duplicate pip installation and duplicate G1A activity. v6 treats this as a release-blocking concurrency hazard.

Before any master stage, `master_account_driver_v6.py` acquires a per-account `fcntl.flock` under `/kaggle/working` and writes an atomic session marker:

- only one K1/K2/K3 driver may enter the account critical section;
- an overlapping duplicate waits and mirrors the primary terminal result instead of rerunning stages;
- a second invocation after a same-session terminal result does not rerun the account;
- a stale `RUNNING` marker after process loss fails closed and requires a fresh Batch session;
- intended G2A GPU0/GPU1 parallelism remains inside the single primary driver and is unaffected.

The notebook bootstrap also uses a process-isolated runtime checkout, so duplicate notebook processes cannot delete or overwrite each other's operator code before the driver lock is acquired.

## Runtime stages

Every primary account execution logs:

1. `RUNTIME_AND_HARDWARE_PREFLIGHT`
2. `FROZEN_INPUT_RESOLUTION`
3. `SCIENCE_SOURCE_AND_GITHUB_PREFLIGHT`
4. `EXACT_EXECUTION_STACK`
5. `CANONICAL_G1A`
6. `ACCOUNT_G2A`
7. `CONTROL_PLANE`
8. `SCIENCE_DURABILITY_PREFLIGHT`
9. `SCIENTIFIC_QUEUE`

The science checkout is reverified clean after checkout, stack repair, G1A, G2A, control-plane construction, and before science.

The optional `CROPCOP_EXPECTED_KAGGLE_USERNAME` environment variable can bind one notebook to an expected Kaggle username and fails before heavy work on mismatch.

## Three-account behavior

- **K1:** creates/adopts one canonical G1A, runs `CAL-EFFB0` + `CAL-CNXTT`, collects all five G2A summaries, seals barrier/scheduler/durability-bound GO, then runs its scientific queue.
- **K2:** consumes exact K1 G1A, runs `CAL-MNV4-LOGITS` + `CAL-MNV4-FEATURE`, validates control, then runs its scientific queue.
- **K3:** consumes exact K1 G1A, runs `CAL-R13` on GPU0 during G2A, validates control, then uses both GPUs in science.

G2A uses five T4s because exactly five prospective profiles are frozen. Scientific execution uses six independent single-GPU workers. No DDP, DataParallel, FSDP, or cross-run gradient synchronization is allowed.

## Evidence and recovery

Public-safe text evidence is parent-published to source-bound `run-evidence/*` branches. Publication is serialized and recovery-safe:

- identical remote evidence is a successful no-op;
- partial bundles publish only missing/changed audited files;
- final full-bundle round-trip is required;
- evidence branches must descend from the frozen science SHA;
- publication failure does not invalidate completed science.

Private model/checkpoint material stays in private Kaggle datasets. GPU children do not receive Git credentials.

A controlled dependency/session/publication continuation is resumed by running the **same account notebook in a fresh Batch session**. Same-session duplicate execution is never a recovery mechanism; it is serialized/mirrored. Actual validation failures, malformed state, source drift, or scientific failures remain fail-closed.

## Kaggle requirements

All three accounts:

- accelerator: **T4 x2**;
- Internet: **ON**;
- execution: **Save Version -> Save & Run All / Batch**;
- secrets: `KAGGLE_USERNAME`, `KAGGLE_KEY`, `CROPCOP_GITHUB_TOKEN`;
- frozen CropCop V1 dataset attached.

K1 additionally attaches the complete historical principal G1 bundle.

After a real K1 G1A PASS publishes the canonical private G1A locator, give K2 and K3 **Can view** access to that private dataset once. Do not make it public.

## Safety boundary

These notebooks automate G1A, five G2A profiles, barrier/scheduler/final GO, private durability, and the 11 remaining Track-A continuation states. They do **not** open protected V1 test, Track B, or Track C. Post-training evidence/XAI/selection/comprehensive 21-state closure begins only after actual terminal scientific evidence exists.

`MASTER_RUNTIME_FREEZE.json` records operator implementation/CI qualification only; it is not a substitute for real Kaggle G1A/science artifacts.
