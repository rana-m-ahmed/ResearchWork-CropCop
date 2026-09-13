# Track-A v1.2 Kaggle Operator Infrastructure

**Scientific source (immutable):** `9a72e9466a9a3e7429e0e36a028edac662f83146`  
**Operator branch:** `ops-tracka-kaggle-launch-20260913`  
**Authority:** `EAAI-JE-SDL-v2.1-QA`  
**Canonical operator schema:** `2.1`

This directory is operational infrastructure only. It MUST NOT become the scientific source of record. Every notebook checks out the exact scientific commit above into a separate detached worktree and refuses source drift.

`tracka_v12_kaggle_operator_v2.py` is the canonical helper surface. `tracka_v12_kaggle_operator.py` is its internal base implementation and MUST NOT be imported directly by canonical drivers. CI enforces this distinction.

## Kaggle session requirements

- Internet: **ON** for repository clone, exact upstream materialization and Kaggle private-dataset API operations.
- Accelerator: G1A may use a normal Kaggle GPU session; G2A/scientific account notebooks require **T4 x2** and fail closed otherwise.
- Python: exact `3.12.13`; the driver stops before qualification if the interpreter drifts.
- G2A/science secrets: `KAGGLE_USERNAME` and `KAGGLE_KEY` for the account executing that lane.
- Never expose GitHub publication credentials to G2A/scientific children.

## Design goals

- no hard-coded Kaggle mount slug is trusted;
- frozen manifest and class map are resolved by exact SHA-256;
- image root is qualified against manifest-relative paths and must resolve uniquely;
- historical principal G1 is identified by sealed identity, not folder name;
- R06/R07 upstream artifacts are downloaded through the frozen TorchVision enum/provenance implementation;
- R13 uses its exact Hugging Face repository/commit and exact byte/SHA identity;
- exact dependency environment is validated before qualification;
- G2A children receive exactly one T4 via `CUDA_VISIBLE_DEVICES`;
- Git/publication credentials are stripped from G2A/scientific child environments;
- calibration durability uses distinct owner-matched private Kaggle datasets;
- G2A validation metrics remain disabled and G2A cannot authorize science;
- G2A barrier, scheduler and final durability-bound GO remain separate fail-closed gates;
- scientific notebooks consume only frozen scheduler queues and their account-owned durability targets;
- session rollover resumes verified private checkpoint state and never silently starts fresh.

## Notebook sequence

1. `01_TRACKA_V12_G1A_SEAL.ipynb`
   - attach frozen V1 manifest/class map/images + complete historical principal G1 bundle;
   - exact R06/R07/R13 upstreams are materialized automatically;
   - terminal success: `TRACKA_V12_G1A_SEAL.json` has `status=PASS`, `science_authorized=false`, and `source_git_sha=9a72e946...`.

2. `02_TRACKA_V12_G2A_K1.ipynb`
   - K1/GPU0: `CAL-EFFB0`
   - K1/GPU1: `CAL-CNXTT`

3. `03_TRACKA_V12_G2A_K2.ipynb`
   - K2/GPU0: `CAL-MNV4-LOGITS`
   - K2/GPU1: `CAL-MNV4-FEATURE`

4. `04_TRACKA_V12_G2A_K3.ipynb`
   - K3/GPU0: `CAL-R13`
   - GPU1 intentionally unused during qualification.

5. `05_TRACKA_V12_SEAL_G2A_AND_SCIENCE_GO.ipynb`
   - attach five G2A summary JSONs, the full G1A bundle, and both exact-head CI attestation JSONs;
   - seals G2A v1.2.2, six-slot scheduler and canonical final authorization (`status=GO`);
   - derives the exact 11-state durable map from sealed scheduler queues and the three authenticated account owners;
   - emits `TRACKA_V12_EXECUTION_CONTROL_BUNDLE/`.

6. `06_TRACKA_V12_SCIENCE_K1.ipynb`
7. `07_TRACKA_V12_SCIENCE_K2.ipynb`
8. `08_TRACKA_V12_SCIENCE_K3.ipynb`
   - attach frozen V1, full G1A bundle and execution-control bundle;
   - invoke canonical `run_tracka_v12_account.py`;
   - one independent child per T4; DDP/DataParallel/FSDP remain forbidden;
   - safe session rollover returns for continuation; actual worker failures are not treated as resumable success.

## Cross-account handoff

G1A model material and the execution-control bundle are private. Do not make them public merely to move between Kaggle accounts. Transfer exact bytes as private inputs (manual download/upload or explicitly shared private Kaggle datasets).

The G2A summary JSONs contain no scientific metrics, but the final sealer still treats them as exact qualification evidence and binds their hashes into the G2A durability contract.

## Path-mismatch policy

Automatic discovery is identity-based. If the same source is mounted twice, discovery becomes ambiguous and the notebook stops. Use the optional explicit override cell only to disambiguate; the override is still verified against frozen hashes/structure.

## QA policy

`tests/test_tracka_v12_kaggle_ops.py` and `.github/workflows/validate-tracka-v12-kaggle-ops.yml` statically validate the operator contract, eight notebook JSONs, exact source binding, profile/account mapping, canonical helper version, non-scientific G2A boundary, canonical GO schema, protected-surface exclusion and the absence of embedded notebook outputs.

Any scientific-code change requires a new exact-head CI/attestation cycle and is outside this operator bundle.
