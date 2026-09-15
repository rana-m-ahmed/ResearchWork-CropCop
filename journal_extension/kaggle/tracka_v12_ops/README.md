# CropCop Track-A Kaggle Master Runtime v8.2

**Science source:** `4ced2fd7c764c07fa47fb57fbea38376d2ce61a4`  
**Runtime branch:** `ops-tracka-kaggle-master-runtime-v8r2-4ced2fd-20260915`  
**Authority:** `EAAI-JE-SDL-v2.1-QA`  
**Active launcher:** `master_launch_guard_v8.py`  
**Active driver:** `master_account_driver_v8.py`

This branch is runtime infrastructure only. It does not contain canonical execution notebooks. After this runtime passes release QA, a separate distribution branch pins the exact runtime commit and contains exactly three notebooks: K1, K2 and K3.

## What v8.2 fixes

The active route is now versioned end-to-end for the pre-science R13 parity amendment:

- G1A: `seal_tracka_v12_g1a_v121.py`
- R13 parity contract: `TRACKA-A1-R13-PRETRAINED-IDENTITY-V1.2.1`
- required max absolute parity difference: `5e-5`
- historical v1.2 threshold retained as provenance: `1e-5`
- scientific account parent: `run_tracka_v12_account_v121.py`
- training runner: `run_tracka_v12_training_v121.py`
- SCIENCE_GO: `seal_tracka_v12_science_go_v124.py`

Historical v1.2 implementations remain unchanged for auditability. Active v8.2 helpers explicitly rebind every G1A consumer to the v1.2.1 validator so a hidden historical validator cannot reject the amended canonical seal.

## Exact-head release gate

Before dependency installation or G1A work, every account:

1. verifies the runtime release manifest;
2. checks out the exact science SHA detached and clean;
3. checks GitHub evidence write access;
4. downloads the exact GitHub Actions code and lock/runtime attestation artifacts;
5. verifies artifact IDs, byte SHA-256 values, source/head SHA, parity contract, required pre-science gates, exact-pretrained parity evidence, and protected-surface closure;
6. only then installs/verifies the frozen execution stack.

This prevents a stale or locally copied attestation from silently authorizing a different source.

## Proven Kaggle input layout

Known K1 mount layout from the completed historical runs:

- V1 root: `/kaggle/input/datasets/ranamuhammadahmed6/cropcop-finalized-v8-11-2026-1/CropCop_Final_v1`
- manifest: `audit/final_manifest.csv`
- class map: `audit/class_to_idx.json`
- image root: `dataset`
- historical principal G1: `/kaggle/input/datasets/ranamuhammadahmed6/cropcop-g1-sealed/G1_PACKAGE`

Paths are conveniences, not trust anchors. The runtime verifies:

- manifest SHA256 `bdb82211ccc2059153724eea178a1680893a6b38ecc243fae484baa91dbf68e2`
- class-map SHA256 `46f7811726c19c42bd7213b2d8178b19a5a182a1b763f60a94ee2c0e5f6688d2`
- principal-G1 seal SHA256 `442d9e7708749efedcb82ef9f4fd131211549770eecad985177eaaf4117052cd`
- TRAIN/VAL image-root compatibility without opening protected V1-test images.

## Account flow

All accounts require Kaggle T4 x2, Internet ON, Batch execution, `KAGGLE_USERNAME`, `KAGGLE_KEY`, `CROPCOP_GITHUB_TOKEN`, and account-local `CROPCOP_EXPECTED_KAGGLE_USERNAME` matching the authenticated Kaggle identity.

- **K1** owns/creates the one canonical private G1A, runs `CAL-EFFB0` and `CAL-CNXTT`, validates all five G2A summaries, seals the barrier/scheduler/durability-bound SCIENCE_GO, and then executes its frozen science queue.
- **K2** consumes K1 G1A, runs `CAL-MNV4-LOGITS` and `CAL-MNV4-FEATURE`, consumes the exact control plane, and executes its queue.
- **K3** consumes K1 G1A, runs `CAL-R13` on GPU0 for G2A, consumes the control plane, and executes its two-GPU science queue.

K1 must grant K2 and K3 **Can view** access to the private canonical G1A dataset before the worker accounts attempt to consume it.

## Durability and recovery

G2A and scientific checkpoints use generation-aware private Kaggle datasets. Destructive local deletion occurs only after the intended uploaded generation is confirmed and retrievable. Restores validate transaction/generation identity plus checkpoint hashes and frozen scientific identity.

A technical dependency/session rollover is recovered by running the same account notebook again. Validation failures, wrong owners, wrong source identity, corrupt/stale checkpoints, malformed summaries, scientific failures, and protected-surface access remain fail-closed.

## Safety boundary

- Exactly 11 Track-A scientific states remain.
- No scientific training occurs before `SCIENCE_GO`.
- V1 test, Track B and Track C remain closed.
- One model is assigned per GPU; no DDP, DataParallel, FSDP or cross-run gradient synchronization is allowed.
- Scientific children receive no Git credentials; the master parent publishes public-safe evidence.

A green runtime workflow means **operator/release QA PASS only**. Real G1A, G2A, SCIENCE_GO and scientific-result PASS states must still be produced by actual Kaggle execution.
