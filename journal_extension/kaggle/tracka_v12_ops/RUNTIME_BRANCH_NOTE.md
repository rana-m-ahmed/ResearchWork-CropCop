Canonical immutable runtime target: `ops-tracka-kaggle-master-runtime-v6-280743c` at `280743cf619d47b093944de6ea63be62dcb8f7ae`.

Distribution notebooks must clone that branch, verify the exact SHA and clean worktree, then invoke `master_launch_guard_v6.py`. They must not invoke `master_account_driver_v6.py` directly.

v6 preserves the verified V1/root/provenance/publication hardening from prior revisions and adds two fail-closed remediations established by the real K1 log: fingerprint-bound execution compatibility for the unchanged frozen G1A sealer's missing `TORCHVISION_VERSION` global, sourced only from frozen `cropcop_je.secondary`, and whole-account launch serialization so duplicate Kaggle execution cannot race dependency repair, G1A construction, downloads or evidence publication. Frozen science remains unchanged at `9a72e9466a9a3e7429e0e36a028edac662f83146`.
