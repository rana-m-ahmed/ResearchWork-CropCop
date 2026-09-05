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
- **Verified smoke execution implementation:** `045fcf5c80366438b69a54288d9081e9b57ed973`
- **Implementation exact-head CI:** run #32 / ID `33966118966`
- **Implementation CI result:** compile PASS; strict validator PASS; JE static validator PASS; 113/113 tests PASS
- **Implementation validation artifact SHA-256:** `509d155ec16f553bcbbc61ef0430e2d0c3c8d1f199dd7dd38124f4d1115b94e9`
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

For Stage 01A-SR the operator normally changes only the phase; for Smoke B they additionally set
`SMOKE_A_INPUT_ROOT`. The smoke implementation was frozen and exact-head verified first at
`045fcf5c80366438b69a54288d9081e9b57ed973`, so this later canonical-notebook wrapper can safely hard-bind that already-verified
implementation SHA without a self-reference loop.

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

The smoke execution implementation at `045fcf5c80366438b69a54288d9081e9b57ed973` has already passed exact-head Actions run #32.
This wrapper commit still requires its own fresh exact-head CI because it changes the canonical notebook
configuration and operator documentation.

After wrapper repository verification only, the allowed verdict is:

`PASS — API-FREE CROSS-SESSION SMOKE MACHINERY + KAGGLE NOTEBOOK VERIFIED; REAL SMOKE-WRITE MAY BEGIN`

This report never authorizes:

`PASS — REAL KAGGLE INFRASTRUCTURE SMOKE QUALIFIED`

That later gate requires the actual human-operated Smoke A and fresh-session Smoke B Saved Versions.


## Stage 01A-SR-N — Canonical Notebook Serialization Hotfix

A manual pre-execution inspection identified that the PR-head notebook at
`112c1083f4abfb80888a44aa4224d432e7bf146c` decoded its orchestration code cell to one physical
Python line containing textual `\\n` separators. Structure/token tests passed because they joined the
cell source without compiling the result. No Kaggle execution had occurred.

The hotfix is intentionally narrow:

- the frozen Smoke A/B execution implementation remains
  `045fcf5c80366438b69a54288d9081e9b57ed973`;
- `canonical_lane.ipynb` is regenerated from a normal multiline Python source;
- `generate_canonical_notebook.py` serializes source with `splitlines(keepends=True)`;
- legitimate escaped newlines inside the temporary `GIT_ASKPASS` script remain escaped;
- the decoded orchestration cell now contains 207 physical Python lines;
- the unit suite now calls real Python `compile(..., "canonical_lane.ipynb", "exec")`;
- the suite and JE static validator both require sane real-line structure and the exact frozen source binding.

Runs #31, #32 and #33 remain historical evidence. Run #33 verified the prior wrapper but does not verify
this serialization repair. A fresh exact-head workflow is mandatory before the repaired notebook is
authorized for manual Smoke A.


Exact-head Actions run #34 / ID `33967049644` on
`f61f7df8a71bc5d688234b866c975fc54c0a9951` preserved the serialization repair but rejected two
legacy JE-static textual checks. Those checks searched for compact source fragments
`git','clone` and `checkout','--detach'`; normal multiline Python formatting no longer contains those
serialization-specific substrings. The new notebook compilation check itself did not fail. The
validator was corrected to use formatting-independent command tokens and source-order checks rather
than requiring a particular Python list-literal serialization. Run #34 remains historical failed CI.


Exact-head Actions run #35 / ID `33967104420` on
`0bfe49340a052ee776a6f94211ba7fb5fc067dfa` passed exact checkout, Python compilation, the strict
repository validator, and the strengthened JE static validator. The three new serialization tests also
passed: decoded-notebook `compile(...)`, physical multiline structure, and generator serialization.
The full suite discovered 116 tests and rejected three older formatting-sensitive test assertions that
still searched for compact/single-quoted notebook source fragments. Those legacy tests were updated to
assert the same semantic ordering/routes against the normalized multiline source. No notebook or smoke
execution code changed in this correction. Run #35 remains historical failed CI.


## Stage 01A-SR-AUTH — Kaggle GitHub Authentication Preflight Hotfix

The first real Kaggle Smoke-A attempt reached GitHub clone and failed with
`Invalid username or token`. Because Kaggle Secrets retrieval had already completed, this isolated the
failure to GitHub credential authorization rather than notebook serialization or smoke execution.

The human-facing notebook authentication boundary was hardened without changing scientific settings or
the Smoke A/B training/recovery state machine:

- trim and validate the Kaggle secret without printing its value;
- reject empty, whitespace/newline-contaminated, or quote-wrapped token values;
- validate the exact private repository through the GitHub REST API;
- use the repository owner as an explicit non-empty HTTPS Git username and the PAT as the password,
  matching GitHub's documented PAT-over-HTTPS flow;
- force `GIT_ASKPASS`, disable interactive prompting and ignore any cached credential helper;
- run authenticated `git ls-remote HEAD` before clone;
- clone only after read authorization succeeds;
- check out the frozen execution source while the authenticated Git environment is active;
- perform `git push --dry-run` to a dedicated evidence probe ref before smoke execution so a read-only
  PAT cannot fail late during evidence publication;
- redact the token from any captured Git stderr used in diagnostic exceptions.

No GitHub API token value, token prefix or secret body is printed. The dry-run creates no remote ref.
For the preferred fine-grained PAT, the operator guide now explicitly requires resource owner
`rana-m-ahmed`, repository `ResearchWork-CropCop`, and Contents: Read and write.


Exact-head Actions run #37 / ID `33967626505` on
`f12c0168b91f556d2f9e2e74712b83bd629b1372` passed exact checkout, Python compilation and the strict
repository validator, then was rejected by one legacy JE-static notebook-prefix assertion because the
auth hardening intentionally added `import json` and `urllib` imports ahead of the previous first line.
The validator and matching structural test were updated to require the new exact import prefix. The
authentication logic and regenerated notebook were not otherwise changed. Run #37 remains historical
failed CI.


## Stage 01A-SR-EOL — Clean Checkout Canonicalization Hotfix

The next real Kaggle retry passed GitHub API authorization, Git-over-HTTPS read, exact detached checkout
and the evidence-branch write dry-run, then correctly failed the clean-tree assertion because
`journal_extension/evidence/claim_registry.csv` appeared modified immediately after checkout.

Root cause: the Git blob stored CRLF bytes while the same source lineage's `.gitattributes` declares
`*.csv text eol=lf`. On Linux/Kaggle Git's attribute normalization therefore makes the worktree/index
comparison dirty even though no notebook step edits the file. This is a repository canonicalization
defect, not a Kaggle mutation.

The hotfix normalizes the CSV bytes to LF without changing any CSV field/content and adds a whole-repo
guard based on `git ls-files --eol`. Any tracked file governed by `eol=lf` that is stored as
`i/crlf` or `i/mixed` now fails JE static validation and the unit suite with the exact path. The
clean-tree gate is not bypassed or special-cased.


After the EOL canonicalization candidate passed run #39, an additional regression test was added that
creates a brand-new local clone with system/global Git configuration disabled, checks out the exact
candidate HEAD detached, and requires `git status --porcelain=v1 --untracked-files=all` to be empty.
This directly reproduces the clean-checkout invariant that Kaggle exposed, rather than relying only on
index-EOL inspection.


### Clean execution source re-freeze after Kaggle checkout defect

The EOL canonicalization plus fresh-clone regression candidate is frozen at
`939455c2cc8787bb295e073d706c32768820bfab`.

Exact-head Actions run #40 / ID `33968396009` verified:

- expected SHA = actual SHA = `939455c2cc8787bb295e073d706c32768820bfab`;
- Python compile PASS;
- strict repository validator PASS;
- JE static validator PASS;
- `test_42_tracked_lf_text_blobs_are_canonical_in_git_index` PASS;
- `test_43_fresh_linux_checkout_of_exact_head_is_clean` PASS;
- 124/124 tests PASS;
- validation artifact ID `9970156417`;
- validation artifact digest
  `sha256:45f078ccd802d8759be70aac0cda3661627e5ededc5aae139c96438faca50588`.

No additional LF-governed CRLF/mixed tracked blob was found after normalizing
`journal_extension/evidence/claim_registry.csv`. The Smoke A/B scientific/non-scientific execution
logic itself was not changed by this canonicalization.


## Stage 01A-SR-PUB — Evidence Publication Clean-Host Hotfix

The real Smoke-A execution on clean source `939455c2cc8787bb295e073d706c32768820bfab`
passed bootstrap/source/dependency/session qualification and reached the first public-safe evidence
publication. The temporary evidence worktree then failed at `git commit` with exit status 128.

Root cause: `publication.py` assumed a preconfigured Git `user.name` / `user.email`. A clean Kaggle
host is not guaranteed to provide either, so the evidence commit could not be created even though the
PAT had already passed both read and write authorization checks.

The publication transaction is hardened without changing Smoke A/B science or checkpoint semantics:

- deterministic isolated author/committer identity is supplied through process environment;
- no global/repository Git identity mutation is required;
- askpass now uses the same non-empty Git username/PAT-password convention as notebook auth;
- credential helpers are disabled for the publication transaction;
- branch existence uses `git ls-remote --exit-code --heads`, distinguishing a missing branch from a
  real auth/network failure;
- authenticated branch fetch/push remains fail-closed;
- Git errors surface bounded stderr/stdout with token redaction instead of opaque
  `CalledProcessError`;
- repeat publication to the same run-evidence branch is explicitly tested.

A new real Git regression test creates a bare remote and source repository with system/global Git
configuration disabled, then calls `publish_to_github_branch()` twice for the same run ID. It requires
two evidence commits, correct final content, and the deterministic evidence author identity.
