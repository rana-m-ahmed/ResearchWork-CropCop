# Track-A v1.2 — Three-Account Master Operator v6

Scientific source remains immutable at:

`9a72e9466a9a3e7429e0e36a028edac662f83146`

Canonical operator runtime:

`280743cf619d47b093944de6ea63be62dcb8f7ae` on `ops-tracka-kaggle-master-runtime-v6-280743c`.

Canonical launcher/driver:

- `master_launch_guard_v6.py`
- `master_account_driver_v6.py`

Use exactly three notebooks: K1, K2 and K3. All require T4 x2, Internet ON, Batch execution, `KAGGLE_USERNAME`, `KAGGLE_KEY`, `CROPCOP_GITHUB_TOKEN`, and the frozen V1 dataset. K1 additionally attaches the complete historical principal G1 bundle.

The notebooks prefer the verified real V1 mount (`CropCop_Final_v1/audit/final_manifest.csv`, `audit/class_to_idx.json`, `CropCop_Final_v1/dataset`) and K1's `cropcop-g1-sealed/G1_PACKAGE`; all paths are re-verified before use.

## v6 hardening

The real v5 K1 execution established two operator-level defects now explicitly mitigated:

1. the unchanged frozen `seal_tracka_v12_g1a.py` references `TORCHVISION_VERSION` without binding it; v6 fingerprints the exact frozen sealer and injects only the value from frozen `cropcop_je.secondary.TORCHVISION_VERSION` (`0.27.1`), refusing any unexpected source/AST drift;
2. the real log showed duplicate overlapping account orchestration; v6 serializes the whole account run with a filesystem lock. One owner executes; duplicate launchers wait and mirror the owner terminal code rather than starting a second pip/G1A/evidence path.

The scientific source itself is not changed by either mitigation.

## K1

1. guarded single-owner runtime/hardware/account preflight;
2. frozen V1 + principal-G1 resolution;
3. detached frozen science checkout and GitHub write preflight;
4. exact dependency stack verification/repair;
5. official EFFB0/CNXT preparation + exact tensor-provenance capture/validation;
6. fingerprinted frozen-sealer compatibility execution;
7. canonical G1A private round-trip + recovery-safe public handoff;
8. parallel `CAL-EFFB0` + `CAL-CNXTT`;
9. collect all five G2A summaries and seal barrier/scheduler/durability-bound `SCIENCE_GO`;
10. preflight K1 scientific durability;
11. run K1 frozen two-GPU queue; parent publishes audited public evidence.

K2 runs `CAL-MNV4-LOGITS` + `CAL-MNV4-FEATURE`; K3 runs `CAL-R13` on GPU0 during G2A. Once science begins, the three accounts use six independent single-GPU workers. No DDP/DataParallel/FSDP/cross-run gradient synchronization is permitted.

GitHub `run-evidence/*` branches carry public-safe text only. Private artifacts/checkpoints remain in private Kaggle datasets. Publication remains serialized/idempotent/ancestry-checked with partial-bundle repair and final full-bundle round-trip.

After a real K1 G1A PASS, share the canonical private G1A dataset with K2/K3 as **Can view**. For controlled `rc=2`, rerun the same account notebook in a fresh Batch session. Real validation/scientific failures remain hard stops.

Protected V1-test, Track B and Track C remain closed. Post-training direct/auxiliary/XAI/selection/21-state closure starts only after all 11 continuation states have terminal scientific evidence.
