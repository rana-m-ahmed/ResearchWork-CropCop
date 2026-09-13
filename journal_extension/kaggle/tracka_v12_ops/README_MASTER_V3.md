# Track-A v1.2 — Three-Account Master Operator v3

This is the high-efficiency Kaggle operator for the frozen Track-A source.

Scientific source (never modified by this operator):

`9a72e9466a9a3e7429e0e36a028edac662f83146`

The master operator deliberately separates:

- **scientific identity** — the frozen source SHA above;
- **operator identity** — the immutable master-runtime SHA pinned by the distributed notebooks;
- **public-safe evidence** — GitHub `run-evidence/*` branches;
- **private material** — private Kaggle datasets only.

## Normal topology

Use exactly three notebooks, one per Kaggle account:

- K1 master: coordinator + two-GPU worker;
- K2 master: two-GPU worker;
- K3 master: two-GPU worker.

Every master notebook requires:

- Kaggle T4 x2;
- Internet enabled;
- `Save Version -> Save & Run All` / Batch execution;
- Kaggle secrets `KAGGLE_USERNAME`, `KAGGLE_KEY`, `CROPCOP_GITHUB_TOKEN`;
- the frozen CropCop V1 dataset attached.

K1 additionally requires the complete historical principal G1 bundle attached.

## What the notebooks do

### K1

1. Clone and verify the frozen science source.
2. Verify the exact locked environment; install it only if the kernel differs.
3. Create or adopt the one canonical Track-A G1A bundle.
4. Round-trip verify that G1A through a private Kaggle dataset.
5. Publish only the G1A seal/public report/handoff to GitHub.
6. Run `CAL-EFFB0` and `CAL-CNXTT` concurrently on GPU0/GPU1.
7. Collect all five canonical G2A summaries from GitHub.
8. Seal the G2A barrier, deterministic six-slot scheduler and durability-bound `SCIENCE_GO` using the packaged exact-head CI attestations.
9. Publish the clean control plane to GitHub.
10. Create/preflight K1 scientific durability datasets.
11. Execute K1's frozen two-GPU queue with automatic private recovery and public-safe evidence publication.

### K2

1. Wait for K1's canonical G1A handoff.
2. Download and validate the exact shared private G1A dataset.
3. Run `CAL-MNV4-LOGITS` and `CAL-MNV4-FEATURE` concurrently.
4. Publish both canonical summaries.
5. Wait for and validate K1's frozen control plane.
6. Create/preflight K2 scientific durability datasets.
7. Execute K2's frozen two-GPU queue with automatic recovery/evidence publication.

### K3

1. Wait for K1's canonical G1A handoff.
2. Download and validate the exact shared private G1A dataset.
3. Run `CAL-R13` on one T4. The second T4 is intentionally unused during G2A because no sixth prospective calibration profile exists.
4. Publish the canonical summary.
5. Wait for and validate K1's frozen control plane.
6. Create/preflight K3 scientific durability datasets.
7. Execute K3's frozen two-GPU queue with automatic recovery/evidence publication.

## One unavoidable one-time action

After K1 creates the canonical private G1A dataset, open that Kaggle dataset and add the K2 and K3 account usernames under **Settings -> Sharing** with **Can view** access.

Do not make the G1A dataset public and do not grant edit access unless independently required.

The K1 log prints the exact private dataset locator. The public handoff contains the same locator and G1A seal hash so worker notebooks can fail closed on identity.

## Recovery

A recovery run uses the **same account master notebook**.

The runtime reuses already-canonical G1A/G2A/control evidence, validates it again, restores scientific checkpoints from private Kaggle durability datasets and continues the frozen queue.

A bounded dependency timeout or planned session-budget rollover is not a scientific failure. A malformed run record, worker exception, corrupted checkpoint, source drift, protected-surface access, wrong G1A/G2A/control identity, or unauthorized experiment is a hard stop requiring investigation.

## Git evidence policy

Only allowlisted text evidence is committed. The inherited publisher rejects:

- model/checkpoint binaries;
- archives;
- secret-like content;
- `/kaggle/input` or `/kaggle/working` private paths;
- files above the public evidence size limit.

GPU children never receive Git credentials. Parent publication is serialized and retried. A publication failure is reported separately from scientific validity.

## Stop boundary

These three master notebooks automate pre-science qualification plus the 11 remaining Track-A scientific continuation states.

They do **not** open V1-test, Track B or Track C. Post-training direct evidence, auxiliary evidence, XAI, four-family selection and comprehensive 21-state closure remain under the already-frozen scientific source and begin only after all 11 continuation states are terminal PASS.
