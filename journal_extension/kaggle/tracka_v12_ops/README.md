# Track-A v1.2 Kaggle Master Operator v5

**Scientific source (immutable):** `9a72e9466a9a3e7429e0e36a028edac662f83146`  
**Distribution branch:** `ops-tracka-kaggle-master-v3-20260913`  
**Pinned master runtime:** `902e7774dda32106a03bfb3f5917946c11ff8e2c`  
**Immutable runtime branch:** `ops-tracka-kaggle-master-runtime-v5-902e777`  
**Driver:** `master_account_driver_v5.py`  
**Authority:** `EAAI-JE-SDL-v2.1-QA`

This branch contains operator/distribution infrastructure only. The frozen scientific source is never modified by the master operator.

## Canonical interface

Normal operation uses exactly three notebooks:

- `TRACKA_V12_MASTER_K1.ipynb`
- `TRACKA_V12_MASTER_K2.ipynb`
- `TRACKA_V12_MASTER_K3.ipynb`

Each notebook is a thin orchestration shell. It clones the immutable v5 runtime, verifies exact HEAD `902e777...` and a clean worktree, then invokes `master_account_driver_v5.py`. The runtime separately checks out scientific source `9a72e946...` detached and verifies it is clean.

Older runtimes/notebooks are rollback/history only and are not canonical for new execution.

## Proven Kaggle input layout

The v5 notebooks prefer the mount layout observed in the actual K1 run:

- V1 root: `/kaggle/input/datasets/ranamuhammadahmed6/cropcop-finalized-v8-11-2026-1/CropCop_Final_v1`
- manifest: `audit/final_manifest.csv`
- class map: `audit/class_to_idx.json`
- image root: `dataset`
- K1 historical principal G1: `/kaggle/input/datasets/ranamuhammadahmed6/cropcop-g1-sealed/G1_PACKAGE`

Preferred paths are **not trusted by path name**. The runtime still verifies the frozen manifest/class-map SHA-256 values, TRAIN/VAL image structure, and principal-G1 seal. If Kaggle changes the mount prefix, bounded discovery is used instead. Protected V1-test images are not opened for root qualification.

Required V1 identities:

- manifest SHA256: `bdb82211ccc2059153724eea178a1680893a6b38ecc243fae484baa91dbf68e2`
- class map SHA256: `46f7811726c19c42bd7213b2d8178b19a5a182a1b763f60a94ee2c0e5f6688d2`
- historical principal G1 seal SHA256: `442d9e7708749efedcb82ef9f4fd131211549770eecad985177eaaf4117052cd`

## Correct TorchVision provenance contract

The master no longer treats the download receipt as pretrained provenance.

For EFFB0 and ConvNeXt-Tiny, v5 performs the frozen scientific sequence:

1. `prepare_torchvision_pretrained.py` downloads/verifies the official artifact bytes and writes a preparation receipt.
2. `capture_torchvision_pretrained_provenance.py` loads the candidate tensors and frozen official TorchVision enum, checks exact key/shape/dtype/tensor equality, computes candidate and official tensor identities, and writes the full provenance contract.
3. The frozen `validate_torchvision_provenance()` must PASS before the record can be supplied to `seal_tracka_v12_g1a.py`.

Required provenance includes `tensor_identity_algorithm`, `tensor_identity_sha256`, `official_tensor_identity_algorithm`, `official_tensor_identity_sha256`, and `official_tensor_match=true`.

## Runtime stages

Every account logs explicit stages:

1. `RUNTIME_AND_HARDWARE_PREFLIGHT`
2. `FROZEN_INPUT_RESOLUTION`
3. `SCIENCE_SOURCE_AND_GITHUB_PREFLIGHT`
4. `EXACT_EXECUTION_STACK`
5. `CANONICAL_G1A`
6. `ACCOUNT_G2A`
7. `CONTROL_PLANE`
8. `SCIENCE_DURABILITY_PREFLIGHT`
9. `SCIENTIFIC_QUEUE`

The optional `CROPCOP_EXPECTED_KAGGLE_USERNAME` environment variable can bind a notebook to one expected Kaggle account and fails before heavy work on mismatch.

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

A controlled dependency/session/publication continuation is resumed by running the **same account notebook** again. Actual validation failures, worker failures, malformed state, source drift, or scientific failures remain fail-closed.

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
