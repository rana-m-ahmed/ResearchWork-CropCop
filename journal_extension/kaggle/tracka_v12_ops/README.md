# Track-A v1.2 Kaggle Master Operator

**Scientific source (immutable):** `9a72e9466a9a3e7429e0e36a028edac662f83146`  
**Master distribution branch:** `ops-tracka-kaggle-master-v3-20260913`  
**Pinned master runtime:** `5654f35fa52c9b6ae6c28f062969c9ebd65af3fa`  
**Immutable runtime branch:** `ops-tracka-kaggle-master-runtime-v3-fix1-5654f35`  
**Authority:** `EAAI-JE-SDL-v2.1-QA`

This branch is operator/distribution infrastructure only. It does not change the frozen scientific source.

## Canonical interface

Normal operation uses exactly three notebooks:

- `TRACKA_V12_MASTER_K1.ipynb`
- `TRACKA_V12_MASTER_K2.ipynb`
- `TRACKA_V12_MASTER_K3.ipynb`

Each notebook clones the immutable master runtime branch, verifies exact HEAD `5654f35...` and a clean worktree, then the runtime checks out the scientific source separately at exact HEAD `9a72e946...`.

The previous runtime `f606311...` remains a historical rollback point but is superseded for new execution by this input-preflight fix. The previous eight stage notebooks are retained only on the v2 rollback branch `ops-tracka-kaggle-launch-20260913`; they are not canonical for new execution.

## Fail-fast input preflight

Before cloning/installing the heavy frozen training stack, every master now verifies the exact frozen V1 manifest, class map and image root. K1 also verifies the complete historical principal G1 bundle at this stage. If the exact manifest/class map is absent or wrong, the operator reports mounted `/kaggle/input` entries plus likely metadata paths, byte counts and SHA-256 values and stops immediately.

Required V1 identities:

- manifest SHA256: `bdb82211ccc2059153724eea178a1680893a6b38ecc243fae484baa91dbf68e2`
- class map SHA256: `46f7811726c19c42bd7213b2d8178b19a5a182a1b763f60a94ee2c0e5f6688d2`

## Runtime behavior

- K1 creates/adopts one canonical G1A, runs two G2A profiles, seals the shared control plane and runs its scientific queue.
- K2 consumes that G1A, runs two G2A profiles and then its scientific queue.
- K3 consumes that G1A, runs the R13 G2A profile and then its scientific queue.
- G2A uses five T4s because exactly five prospective qualification profiles are frozen.
- Scientific execution uses six independent single-GPU workers across the three T4x2 accounts.
- No DDP, DataParallel or FSDP is used.
- Parent processes hold Git/Kaggle credentials; scientific/G2A GPU children do not receive Git publication credentials.
- Public-safe evidence is automatically audited and published to source-bound `run-evidence/*` branches.
- Private model/checkpoint material stays in private Kaggle datasets.
- A controlled session/dependency/publication rollover is resumed by rerunning the same account notebook.

## Kaggle requirements

All three accounts:

- accelerator: **T4 x2**;
- Internet: **ON**;
- execution: **Save Version -> Save & Run All / Batch**;
- secrets: `KAGGLE_USERNAME`, `KAGGLE_KEY`, `CROPCOP_GITHUB_TOKEN`;
- attach the frozen CropCop V1 dataset containing the exact manifest, class map and complete image tree.

K1 additionally attaches the complete historical principal G1 bundle.

After K1 creates the canonical private G1A dataset, give the K2 and K3 Kaggle accounts **Can view** access to that dataset once. Do not make it public.

## Safety boundaries

The master notebooks automate G1A, all five G2A profiles, G2A barrier/scheduler/final GO, private durability setup and the 11 remaining scientific continuation states.

They do **not** open the protected V1 test, Track B or Track C. Post-training direct/auxiliary evidence, XAI, four-family selection and comprehensive 21-state closure start only after all 11 continuation states are terminal PASS.

See `README_MASTER_V3.md` and `MASTER_OPERATOR_CONTRACT_V3.json` for the detailed execution/recovery contract.
