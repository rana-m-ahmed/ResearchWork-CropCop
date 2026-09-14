# Track-A v1.2 Kaggle Master Operator v6

**Scientific source (immutable):** `9a72e9466a9a3e7429e0e36a028edac662f83146`  
**Distribution branch:** `ops-tracka-kaggle-master-v3-20260913`  
**Pinned master runtime:** `280743cf619d47b093944de6ea63be62dcb8f7ae`  
**Immutable runtime branch:** `ops-tracka-kaggle-master-runtime-v6-280743c`  
**Launcher:** `master_launch_guard_v6.py`  
**Driver:** `master_account_driver_v6.py`  
**Authority:** `EAAI-JE-SDL-v2.1-QA`

This branch contains operator/distribution infrastructure only. The frozen scientific source remains unchanged.

## Canonical interface

Normal operation uses exactly three thin notebooks:

- `TRACKA_V12_MASTER_K1.ipynb`
- `TRACKA_V12_MASTER_K2.ipynb`
- `TRACKA_V12_MASTER_K3.ipynb`

Each notebook clones the immutable v6 runtime, verifies exact HEAD and a clean worktree, then invokes the **launch guard**, not the scientific driver directly. The guard serializes the entire account orchestration. One process becomes owner; duplicate launches wait and return the owner result instead of independently repeating pip repair, G1A construction, upstream downloads, durability setup, or evidence publication.

## Verified Kaggle inputs

Preferred V1 layout from the actual K1 environment:

- V1 root: `/kaggle/input/datasets/ranamuhammadahmed6/cropcop-finalized-v8-11-2026-1/CropCop_Final_v1`
- manifest: `audit/final_manifest.csv`
- class map: `audit/class_to_idx.json`
- image root: `dataset`
- K1 historical principal G1: `/kaggle/input/datasets/ranamuhammadahmed6/cropcop-g1-sealed/G1_PACKAGE`

The preferred paths are accelerators only. The runtime re-verifies the frozen manifest/class-map identities, TRAIN/VAL structure, and principal-G1 seal. Protected V1-test images are not opened for root qualification. If the mount prefix changes, bounded verified discovery is used.

Required identities:

- manifest SHA256: `bdb82211ccc2059153724eea178a1680893a6b38ecc243fae484baa91dbf68e2`
- class map SHA256: `46f7811726c19c42bd7213b2d8178b19a5a182a1b763f60a94ee2c0e5f6688d2`
- historical principal G1 seal SHA256: `442d9e7708749efedcb82ef9f4fd131211549770eecad985177eaaf4117052cd`

## TorchVision provenance

EFFB0 and ConvNeXt-Tiny use the frozen two-step path:

1. `prepare_torchvision_pretrained.py` downloads/verifies official bytes and creates the preparation receipt.
2. `capture_torchvision_pretrained_provenance.py` performs exact tensor comparison against the frozen official TorchVision enum and writes the candidate/official tensor identities.
3. Frozen `validate_torchvision_provenance()` must PASS before G1A receives the record.

A download receipt is never accepted as scientific provenance.

## Frozen G1A sealer compatibility

The real v5 K1 run established a source defect in the frozen sealer: `journal_extension/scripts/seal_tracka_v12_g1a.py` references `TORCHVISION_VERSION` when writing baseline initialization evidence but does not bind/import that name. The same immutable science tree defines the intended value as `cropcop_je.secondary.TORCHVISION_VERSION == "0.27.1"`.

v6 **does not modify the frozen sealer**. `master_g1a_sealer_compat_v6.py`:

- requires the exact frozen sealer Git blob `90a918fcdf14130b44b9c4ec24b0b60b1706bb2c`;
- AST-checks that `TORCHVISION_VERSION` is referenced but not bound;
- imports the value from frozen `cropcop_je.secondary` and requires `0.27.1`;
- executes the unchanged sealer bytes with only that missing global injected;
- refuses execution if any of those assumptions drift.

This is an execution compatibility remediation for a verified missing global, not a scientific-setting change.

## Runtime stages

Each canonical owner process logs a PID and launch identity for:

1. `RUNTIME_AND_HARDWARE_PREFLIGHT`
2. `FROZEN_INPUT_RESOLUTION`
3. `SCIENCE_SOURCE_AND_GITHUB_PREFLIGHT`
4. `EXACT_EXECUTION_STACK`
5. `CANONICAL_G1A`
6. `ACCOUNT_G2A`
7. `CONTROL_PLANE`
8. `SCIENCE_DURABILITY_PREFLIGHT`
9. `SCIENTIFIC_QUEUE`

`CROPCOP_EXPECTED_KAGGLE_USERNAME` remains an optional fail-fast account binding.

## Three-account behavior

- **K1:** creates/adopts the one canonical G1A, runs `CAL-EFFB0` + `CAL-CNXTT`, collects all five G2A summaries, seals barrier/scheduler/durability-bound GO, then runs K1 science.
- **K2:** consumes the exact K1 G1A, runs `CAL-MNV4-LOGITS` + `CAL-MNV4-FEATURE`, validates control, then runs K2 science.
- **K3:** consumes the exact K1 G1A, runs `CAL-R13` on GPU0 during G2A, validates control, then uses both GPUs during science.

G2A uses five T4s because exactly five prospective profiles are frozen. Science uses six independent single-GPU workers. DDP, DataParallel, FSDP, and cross-run gradient synchronization remain forbidden.

## Evidence and recovery

Public-safe evidence is parent-published to source-bound `run-evidence/*` branches. Publication remains ancestry-checked, serialized, idempotent, incremental for partial bundles, and full-roundtrip verified. Private model/checkpoint material remains in private Kaggle datasets, and GPU children do not receive Git credentials.

For a controlled `rc=2`, rerun the **same account notebook** in a fresh Batch session. Actual source/config/provenance/scientific failures remain fail-closed.

## Kaggle requirements

All accounts:

- **T4 x2**;
- Internet **ON**;
- **Save Version -> Save & Run All / Batch**;
- secrets: `KAGGLE_USERNAME`, `KAGGLE_KEY`, `CROPCOP_GITHUB_TOKEN`;
- frozen CropCop V1 attached.

K1 additionally attaches the complete historical principal G1 bundle. After a real K1 G1A PASS publishes the canonical private locator, give K2/K3 **Can view** access. Do not make the dataset public.

## Safety boundary

The master automates G1A, five G2A profiles, barrier/scheduler/final GO, private durability, and the 11 remaining Track-A continuation states. It does not open protected V1 test, Track B, or Track C.

`MASTER_RUNTIME_FREEZE.json` records operator implementation/CI qualification only. It is not evidence that a real G1A or scientific experiment has passed.
