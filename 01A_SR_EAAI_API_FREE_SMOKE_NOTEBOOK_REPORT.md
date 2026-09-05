# CropCop → EAAI Journal Extension

## Stage 01A-SR — API-Free Cross-Session Kaggle Smoke Machinery Report

- **Continuation:** Stage 01A-SR
- **Execution date/time:** 2026-09-05T17:17:00+05:00 (Asia/Karachi)
- **Repository:** `rana-m-ahmed/ResearchWork-CropCop`
- **Branch / PR:** `je-stage01a-core-20260905` / PR #8
- **Prior repository authority:** `731fcf18bd6aaf72322457d6e766296dc95d5c94`
- **Prior exact-head CI:** run #30 / ID `33964260845` / 81 of 81 tests PASS
- **Scientific authority:** unchanged
- **R04/R05 science:** unchanged
- **V1 / protected external access:** none
- **Kaggle runtime execution:** none
- **Current real-environment state:** `BLOCKED — REAL KAGGLE INFRASTRUCTURE SMOKE NOT EXECUTED`

## 1. Purpose

This patch corrects only the infrastructure-smoke semantics and human-operated Kaggle notebook.
The prior single-process checkpoint → durable write → delete → restore → resume diagnostic could not
establish true Saved-Version-to-Saved-Version recovery. The new machinery separates the qualification
into two clean Saved Versions and uses Kaggle's native Notebook Output handoff rather than the Kaggle API.

No G1, G2, calibration, R04/R05, V1 test, restricted image, teacher, external cohort or scientific
metric is accessed or produced.

## 2. API-free persistence boundary

Stage 01A-SR no longer uses a private Kaggle dataset as its smoke persistence boundary.

Smoke A writes a self-contained recovery export under:

`/kaggle/working/cropcop-smoke-a-export`

The human operator preserves it through **Save & Run All**.

Smoke B runs in a fresh Saved Version and attaches the exact successful Smoke-A Notebook Output using:

**Add Input → Notebook Output Files**

The attached output is consumed read-only from the explicit human-configured `SMOKE_A_INPUT_ROOT`
beneath `/kaggle/input/...`.

The existing Kaggle private-dataset persistence class is intentionally retained for later non-smoke
execution; it is not required by either smoke phase.

## 3. Phase-specific secrets

For `smoke-write` and `smoke-restore`, the only required Kaggle Secret is:

`CROPCOP_GITHUB_TOKEN`

It is used through temporary `GIT_ASKPASS` for private source clone and audited small Git evidence
publication. Its value is never printed or embedded in a URL.

`KAGGLE_USERNAME` and `KAGGLE_KEY` are not requested by either smoke phase. Future G1/calibration/
principal paths may retain their own requirements independently.

## 4. Smoke-A contract

Smoke A:

1. inherits the notebook-global monotonic clock;
2. verifies clean exact source and exact dependency environment;
3. requires CUDA and captures runtime/GPU/driver identity;
4. creates a synthetic/non-scientific marker bundle automatically;
5. generates a qualification ID independent of source SHA alone;
6. advances a deterministic synthetic model for three optimizer steps;
7. creates one atomic content-addressed checkpoint;
8. immediately verifies it through `recover_latest()`;
9. writes a self-contained recovery export;
10. creates and self-hashes `SMOKE_A_MANIFEST.json`;
11. records `SMOKE_A_EVIDENCE.json`;
12. publishes only the audited text evidence branch;
13. prints the copyable Smoke-A completion block;
14. stops.

The checkpoint is never eligible for Git publication.

## 5. Smoke-B contract

Smoke B requires explicit `SMOKE_A_INPUT_ROOT`.

Before any resumed training or B checkpoint write it:

1. reads exactly one Smoke-A manifest within that explicit input root;
2. validates the manifest self-hash and non-scientific/synthetic flags;
3. validates source SHA and dependency-lock SHA;
4. validates Smoke-A evidence hash/content;
5. validates every recovery-file path, size and SHA;
6. validates expected checkpoint size/SHA and checkpoint-index binding;
7. validates checkpoint identity digest and optimizer step;
8. copies only the already-verified recovery files to a new local restore root;
9. re-verifies copied bytes;
10. calls `recover_latest()`;
11. loads model, optimizer, scheduler and GradScaler state;
12. verifies the restored optimizer step;
13. advances the synthetic run;
14. only then permits `CHECKPOINT_B`;
15. verifies the new B checkpoint;
16. proves the attached A input fingerprint is unchanged;
17. writes separate B manifest/evidence;
18. publishes only B text evidence;
19. stops.

The enforced state-machine sequence is:

`READ_A → VERIFY_A → RESTORE_A → RECOVER_A → LOAD_A → RESUME → CHECKPOINT_B`

## 6. Canonical notebook

`journal_extension/kaggle/canonical_lane.ipynb` remains valid nbformat 4 with one Markdown and one
orchestration code cell. It exposes exactly:

- `smoke-write`
- `smoke-restore`
- `g1`
- `calibration`
- `principal`

The legacy ambiguous production `smoke` route is removed.

For Stage 01A-SR the operator normally changes the phase; for Smoke B they additionally set
`SMOKE_A_INPUT_ROOT`. Because a commit cannot embed its own final SHA, the operator must also set the
non-secret `AUTHORIZED_SOURCE_SHA` once to the exact final Stage-01A-SR PR head given by the PR
attestation after this patch passes CI.

## 7. Files generated for operator use

- `journal_extension/kaggle/README_SMOKE.md`
- `journal_extension/kaggle/smoke_inputs.example.json`

No CropCop data/model path and no Kaggle API credential appears in the smoke input template.

## 8. Test additions

The Stage-01A-SR test suite adds smoke-specific checks for:

- absence of Kaggle API credentials/CLI from the smoke path;
- phase-specific Git-only secret requirement;
- explicit smoke-write/smoke-restore routes and no legacy fallthrough;
- complete Smoke-A recovery export;
- manifest self-hash/checkpoint/source/dependency binding;
- missing/ambiguous input rejection;
- source/dependency/qualification/evidence/checkpoint drift rejection;
- no training before recovery verification;
- no B checkpoint before verified recovery + resume;
- attached A input immutability;
- optimizer-step preservation/advancement;
- B-to-A evidence binding;
- notebook global-clock ordering and nbformat;
- mutable-output separation from Git checkout.

The prior repository test surface is preserved. Final discovered count is determined only by fresh
exact-head Actions on the final PR head.

## 9. Publication rule

Run #30 verifies only `731fcf18bd6aaf72322457d6e766296dc95d5c94` and cannot verify this changed source.

The first Stage-01A-SR candidate `e796a5992ab2f272798d67efb5aeeb1ff7beb274` was rejected by
exact-head Actions run #31 because the JE static validator still expected the old notebook variable
token `source_sha` in the detached-checkout expression. The notebook had intentionally renamed that
operator-facing value to `AUTHORIZED_SOURCE_SHA`. Compile and the strict repository validator passed
on that candidate. The stale validator token was corrected; run #31 remains historical failed evidence.

After these changes are committed, PR #8 must receive a fresh exact-head workflow proving:

- expected SHA equals checked-out SHA;
- Python compilation PASS;
- strict repository validator PASS;
- JE static validator PASS;
- all old + new tests PASS;
- notebook structure PASS;
- validation artifact upload PASS;
- repository leak scan PASS.

The exact final source SHA and Actions identity are recorded in PR metadata after CI so the verified
commit is not modified to describe its own verification.

## 10. Gate

Until repository CI closes:

`BLOCKED — STAGE 01A-SR REPOSITORY VERIFICATION PENDING`

After repository verification only, the allowed verdict is:

`PASS — API-FREE CROSS-SESSION SMOKE MACHINERY + KAGGLE NOTEBOOK VERIFIED; REAL SMOKE-WRITE MAY BEGIN`

This report never authorizes:

`PASS — REAL KAGGLE INFRASTRUCTURE SMOKE QUALIFIED`

That later gate requires the actual human-operated Smoke A and fresh-session Smoke B Saved Versions.
