# Stage 01A-MGPU — Smoke-A Batch-Gate Incident and Fix

- **Date:** 2026-09-05
- **Incident:** first post-QA1 Smoke-A attempt stopped with `KAGGLE_KERNEL_RUN_TYPE=Interactive`
- **Observed frozen source in failed attempt:** `be9b6965d760ff6e8674623b658f66572cd57093`
- **Failure classification:** OPERATOR SESSION MODE / EARLY-GATE UX, not scientific execution failure
- **Scientific state changed:** NO
- **Qualifying Smoke-A evidence created by failed attempt:** NO

## Observed facts

The attempted notebook successfully:

- authenticated and checked out the expected source;
- installed the frozen dependency lock;
- passed clean-source/bootstrap checks;
- reported dependency lock SHA `767859a3cb1b37c0e50f753aa8c7a5c400424417cba3eaa59f0ab617ff18f701`.

The infrastructure smoke then correctly refused qualification because Kaggle exposed:

`KAGGLE_KERNEL_RUN_TYPE=Interactive`

Terminal Smoke A/B qualification requires `Batch`.

## Root cause

The canonical notebook allowed expensive setup work to occur before the repository-level smoke gate checked Kaggle session type.

The safety policy was correct; the gate was too late in the operator path.

## Remediation

### Source hardening

New frozen execution source:

`fe88e426b4698977d65efe9702f1d48cf5ff96a3`

Changes:

1. `require_qualifying_kaggle_batch()` now gives explicit operator instructions:
   - use **Save Version -> Save & Run All**;
   - select the required GPU accelerator;
   - do not qualify by manually running the editor cell.
2. `bootstrap_clean_session.py` independently records and enforces Batch run type.
3. tests cover Interactive rejection and Batch acceptance.

Exact-head source CI:

- Actions run #127 / ID `33979860009`;
- exact SHA match PASS;
- science-diff PASS;
- QA-A 18/18;
- QA-B 13/13;
- QA-C 8/8;
- MGPU 66/66;
- complete suite 234/234 PASS.

### Wrapper hardening

Final wrapper head:

`3cac2d3e37e3b6717039991d7031c9ad12312d1a`

The canonical notebook is bound to execution source:

`fe88e426b4698977d65efe9702f1d48cf5ff96a3`

The notebook now checks `KAGGLE_KERNEL_RUN_TYPE` immediately after local operator configuration and phase validation.

If it is not `Batch`, it fails **before**:

- Kaggle Secrets retrieval;
- GitHub API/network authentication;
- repository clone;
- package installation;
- bootstrap;
- checkpoint/output creation;
- evidence publication.

Exact-head wrapper CI:

- Actions run #133 / ID `33979947915`;
- exact wrapper SHA match PASS;
- generator == notebook PASS;
- science-diff PASS;
- QA-A 18/18;
- QA-B 13/13;
- QA-C 8/8;
- MGPU 66/66;
- complete suite 236/236 PASS.

## Additional audit

The Smoke-A path was re-reviewed after the incident for the next likely blockers:

- dependency lock: already proved installable by the failed Interactive attempt;
- exact source checkout/clean tree: already PASS in the failed attempt;
- CUDA support: remains required and is checked before Smoke-A work;
- public evidence payload: uses the allowlisted sanitized environment capture rather than arbitrary `os.environ`;
- public evidence publication: token-redacted, explicit file allowlist, isolated Git identity;
- recovery export: binds checkpoint index/object SHA/bytes/identity;
- Smoke-A final evidence SHA is rebound into the final manifest after export-integrity changes;
- Smoke-B remains unable to write a new checkpoint before READ→VERIFY→RESTORE→RECOVER→LOAD→RESUME.

No additional repository-level blocker was identified by static/CI audit.

## Retry authority

The failed Interactive attempt is diagnostic only and must not be used as Smoke-A qualification evidence.

The next valid attempt must use:

- canonical wrapper at/after `3cac2d3e37e3b6717039991d7031c9ad12312d1a`;
- `EXECUTION_PHASE=smoke-write`;
- Kaggle **Save Version -> Save & Run All**;
- GPU accelerator enabled;
- only required smoke secret: `CROPCOP_GITHUB_TOKEN`.

Expected first runtime preflight line:

`Kaggle Saved-Version/Batch preflight: PASS (run_type=Batch)`

Only after that should setup/install/bootstrap and Smoke A proceed.
