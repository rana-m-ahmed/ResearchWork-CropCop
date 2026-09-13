# Track-A v1.2 Kaggle Operator Infrastructure

**Scientific source (immutable):** `9a72e9466a9a3e7429e0e36a028edac662f83146`  
**Operator branch:** `ops-tracka-kaggle-launch-20260913`  
**Authority:** `EAAI-JE-SDL-v2.1-QA`

This directory is operational infrastructure only. It MUST NOT become the scientific source of record. Every notebook in this bundle checks out the exact scientific commit above into a separate detached worktree and refuses source drift.

## Design goals

- no hard-coded Kaggle mount slug is trusted;
- frozen manifest and class map are resolved by exact SHA-256;
- the image root is qualified against manifest-relative image paths and must resolve uniquely;
- the historical principal G1 bundle is identified by its sealed G1 identity, not by a folder name;
- EfficientNet-B0 and ConvNeXt-Tiny upstream artifacts are downloaded through the frozen TorchVision enum and provenance script;
- R13 is downloaded from the exact Hugging Face repository/commit and verified by exact byte count and SHA-256;
- Python and every frozen training dependency are validated before qualification;
- G2A children receive exactly one T4 via `CUDA_VISIBLE_DEVICES`;
- Git/publication credentials are stripped from G2A/scientific child environments;
- calibration durability uses distinct owner-matched private Kaggle datasets;
- calibration runs never enable validation metrics or authorize science;
- G2A barrier, scheduler and final GO remain separate fail-closed gates;
- scientific account notebooks consume the frozen scheduler queues and create private durable targets only for their assigned states;
- session rollover is resumable from private Kaggle durability and must never silently restart a scientific state.

## Notebook sequence

1. `01_TRACKA_V12_G1A_SEAL.ipynb`
   - inputs: frozen V1 manifest/class map/images + historical principal G1 bundle;
   - auto-downloads exact official R06/R07 upstream weights and exact R13 upstream bytes;
   - emits `TRACKA_V12_G1A_BUNDLE/`;
   - terminal success requires `TRACKA_V12_G1A_SEAL.json` with `status=PASS` and `science_authorized=false`.

2. `02_TRACKA_V12_G2A_K1.ipynb`
   - K1/GPU0: `CAL-EFFB0`
   - K1/GPU1: `CAL-CNXTT`
   - creates/validates two private durability datasets owned by the authenticated K1 Kaggle account;
   - writes two summary JSON files.

3. `03_TRACKA_V12_G2A_K2.ipynb`
   - K2/GPU0: `CAL-MNV4-LOGITS`
   - K2/GPU1: `CAL-MNV4-FEATURE`

4. `04_TRACKA_V12_G2A_K3.ipynb`
   - K3/GPU0: `CAL-R13`
   - GPU1 intentionally unused.

5. `05_TRACKA_V12_SEAL_G2A_AND_SCIENCE_GO.ipynb`
   - mount the five G2A summary JSONs, the G1A bundle/seal, and the two exact-head CI attestation JSONs;
   - seals the v1.2.2 G2A barrier and six-slot scheduler;
   - seals the durability-bound final science GO;
   - derives the exact 11-state durable-map locator plan from scheduler→account ownership;
   - emits `TRACKA_V12_EXECUTION_CONTROL_BUNDLE/`.

6. `06_TRACKA_V12_SCIENCE_K1.ipynb`
7. `07_TRACKA_V12_SCIENCE_K2.ipynb`
8. `08_TRACKA_V12_SCIENCE_K3.ipynb`
   - mount the frozen V1 dataset, full G1A bundle and execution-control bundle;
   - each notebook checks/creates only its own private scientific durability targets;
   - invokes the repository's canonical `run_tracka_v12_account.py`;
   - two independent one-GPU children are used; DDP/DataParallel/FSDP remain forbidden.

## Required Kaggle secrets

For G2A and scientific durability notebooks add:

- `KAGGLE_USERNAME`
- `KAGGLE_KEY`

The authenticated username must be the owner of every durability dataset used by that account.

## Cross-account handoff

G1A model material and execution-control artifacts are private. Do not make them public merely to move them between Kaggle accounts. Transfer them as private inputs (download/upload or explicitly shared private datasets) and keep the exact files unchanged.

The five G2A summary JSONs contain no scientific metrics, but the final barrier notebook still treats them as exact qualification evidence and validates their hashes/bindings.

## Path-mismatch policy

Automatic discovery is preferred. If the same data are mounted twice and discovery becomes ambiguous, the notebook stops and asks for an explicit override. It never guesses among multiple candidate roots.

## Source discipline

The ops branch may evolve without changing the scientific source. Any change to scientific code requires a new exact-head CI attestation cycle and is outside this operator bundle.
