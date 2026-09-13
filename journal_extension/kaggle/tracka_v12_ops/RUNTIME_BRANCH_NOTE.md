Immutable runtime target: `ops-tracka-kaggle-master-runtime-v3-fix2-9fdbba6` at `9fdbba6f81bfa2f1ce235b3d31ee52a75719a779`.

Distribution notebooks must clone that branch and verify the exact SHA before invoking `master_account_driver.py`.

This runtime supersedes `5654f35...` for new execution. It retains fail-fast exact V1/principal-G1 validation, replaces the expensive broad hash sweep with bounded shallow metadata discovery, accepts byte-identical frozen metadata duplicates, selects the manifest copy that is structurally nearest to the actual image tree, and still fails closed if duplicate manifests qualify different image roots. Frozen science remains unchanged at `9a72e9466a9a3e7429e0e36a028edac662f83146`.
