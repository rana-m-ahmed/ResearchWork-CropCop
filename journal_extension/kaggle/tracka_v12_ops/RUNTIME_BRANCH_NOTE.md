# Track-A v1.2/v1.2.1 Kaggle Runtime v8.2

This branch is the **runtime-only** line for the repaired Track-A continuation infrastructure.

- Science authority: `4ced2fd7c764c07fa47fb57fbea38376d2ce61a4`
- Runtime branch: `ops-tracka-kaggle-master-runtime-v8r2-4ced2fd-20260915`
- Active launcher: `master_launch_guard_v8.py`
- Active driver: `master_account_driver_v8.py`
- R13 parity contract: `TRACKA-A1-R13-PRETRAINED-IDENTITY-V1.2.1`
- R13 parity maximum: `5e-5` (historical v1.2 provenance remains `1e-5`)
- Remaining scientific states: exactly 11

The runtime branch intentionally contains **no canonical K1/K2/K3 notebooks and no distribution freeze file**. A separate distribution branch is created only after runtime QA passes; that branch pins the exact runtime commit SHA and contains exactly three execution notebooks. This avoids stale notebooks and circular self-references.

Active G1A execution uses `seal_tracka_v12_g1a_v121.py`. Scientific account execution uses `run_tracka_v12_account_v121.py`. SCIENCE_GO uses `seal_tracka_v12_science_go_v124.py`. Exact-head GitHub Actions attestations are downloaded, byte-hash checked, semantically checked, and bound before dependency installation or G1A work.

Historical runtime/operator files remain for audit history only. They are not canonical execution entry points.

This branch and its CI qualify operator implementation/release integrity only. They do not claim a real Kaggle G1A, G2A, SCIENCE_GO, or scientific-result PASS.
