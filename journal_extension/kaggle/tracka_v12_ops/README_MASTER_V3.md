# Track-A v1.2 — Three-Account Master Operator v5

Scientific source remains immutable at:

`9a72e9466a9a3e7429e0e36a028edac662f83146`

Canonical operator runtime:

`902e7774dda32106a03bfb3f5917946c11ff8e2c` on `ops-tracka-kaggle-master-runtime-v5-902e777`.

The operator separates scientific identity, operator identity, public-safe GitHub evidence, and private Kaggle durability material.

## Topology

Use exactly three notebooks: K1, K2 and K3. All require T4 x2, Internet ON, Batch execution, `KAGGLE_USERNAME`, `KAGGLE_KEY`, `CROPCOP_GITHUB_TOKEN`, and the frozen V1 dataset. K1 additionally attaches the complete historical principal G1 bundle.

The notebooks prefer the real verified V1 mount observed in Kaggle (`CropCop_Final_v1/audit/final_manifest.csv`, `audit/class_to_idx.json`, and `CropCop_Final_v1/dataset`) and K1's `cropcop-g1-sealed/G1_PACKAGE`. Every preferred path is still cryptographically/structurally verified; bounded discovery is the fallback.

## K1

1. Runtime/hardware/account preflight.
2. Frozen V1 + historical principal-G1 resolution.
3. Detached frozen science checkout and GitHub write preflight.
4. Exact dependency stack verification/repair.
5. Official EFFB0/CNXT bytes preparation **followed by exact tensor-provenance capture and frozen validation**.
6. Canonical G1A seal + private Kaggle round-trip + recovery-safe public handoff.
7. Parallel `CAL-EFFB0` and `CAL-CNXTT` on GPU0/GPU1.
8. Collect all five validated G2A summaries.
9. Seal barrier, scheduler and durability-bound `SCIENCE_GO`.
10. Create/preflight K1 scientific durability targets.
11. Run K1 frozen two-GPU queue; parent publishes audited public evidence after frozen execution.

## K2

Acquire/validate exact K1 G1A, run `CAL-MNV4-LOGITS` + `CAL-MNV4-FEATURE` in parallel, validate the published control plane, preflight K2 durability targets, then run K2's frozen two-GPU science queue.

## K3

Acquire/validate exact K1 G1A, run `CAL-R13` on GPU0 for G2A (GPU1 intentionally idle because no sixth prospective calibration exists), validate control, preflight durability, then use both GPUs for K3 science.

## TorchVision provenance rule

A `prepare_torchvision_pretrained.py` download receipt is never accepted as scientific provenance. `capture_torchvision_pretrained_provenance.py` must independently compare the artifact against the frozen TorchVision enum and produce matching candidate/official tensor identities with `official_tensor_match=true`. The frozen provenance validator must pass before G1A can seal.

## Coordination/recovery

GitHub `run-evidence/*` branches carry public-safe text only. Private artifacts/checkpoints remain in private Kaggle datasets. Parent publication is serialized and idempotent; partial bundles repair only missing/changed files and require a final full-bundle round-trip. GPU children do not receive Git credentials.

After a real K1 G1A PASS, share its private G1A dataset with K2/K3 as **Can view** once. A controlled technical/session/publication continuation uses the same notebook again. Real validation/scientific failures remain hard stops.

## Stop boundary

The master system covers pre-science qualification and the 11 remaining Track-A training states only. Protected V1-test, Track B and Track C remain closed. Post-training direct/auxiliary/XAI/selection/21-state closure begins only after actual terminal scientific results exist.
