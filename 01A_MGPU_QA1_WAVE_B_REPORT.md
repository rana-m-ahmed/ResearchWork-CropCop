# Stage 01A-MGPU-QA1 — Wave B Report

- **Wave:** B — Finalization + Continuation + Terminal Recovery
- **Exact-head commit:** `66a5934b0c35dbd84048af84b9f927ac41b1b96d`
- **GitHub Actions:** run #112 / ID `33978769002`
- **Conclusion:** SUCCESS
- **Science-diff:** PASS
- **Wave-A tests:** 18 / 18 PASS
- **Wave-B tests:** 13 / 13 PASS
- **MGPU tests:** 66 / 66 PASS
- **Complete CPU-safe suite:** 224 / 224 PASS
- **Scientific re-lock:** NOT REQUIRED
- **`train.py` changed:** NO

## Closed findings

### S1-2 — graceful finalization

Closed.

Planned/common-deadline finalization is now separate from emergency child termination.

Planned finalization:

- sends SIGTERM to all running children together;
- derives a bounded grace from notebook finalization margin plus conservative checkpoint/sync estimates;
- waits substantially longer than the former 30-second emergency grace;
- polls children during the grace;
- SIGKILLs only child groups still hung after that finalization window;
- records termination mode, duration and return code.

Child runtime timeout remains a separate emergency path.

After graceful SIGTERM, the parent executes normal child-preflight/result collection. A valid PASS or continuation record is accepted rather than blindly converting a formerly RUNNING child into continuation-required state.

### S1-3 — publication-failure continuation laundering

Closed.

Envelope child state now separates:

- execution status;
- publication status;
- evidence-chain completeness.

A scientific/calibration execution PASS with publication FAIL:

- is not considered a terminal skip;
- enters publication-repair classification;
- is explicitly excluded from the training queue;
- validates/carries forward the prior terminal artifact;
- retries publication only;
- preserves the same run ID and result;
- records `training_relaunched=false`;
- cannot make the envelope PASS until publication/evidence completeness is restored.

### S2-8 — coherent prior envelope validation

Closed.

Continuation now requires exactly one co-located:

- `ENVELOPE_MANIFEST.json`;
- `ENVELOPE_EVIDENCE.json`;
- `ENVELOPE_STATE.json`.

The prior bundle is validated for:

- manifest self-hash;
- source/amendment/G1/G2 identity;
- evidence→manifest SHA binding;
- child/run-ID mapping;
- execution/publication/evidence-completeness coherence;
- terminal result artifact presence.

Terminal principal result carry-forward additionally validates the run record and requires metrics + segment ledger before publication repair.

Attached prior control files are SHA-fingerprinted before continuation and checked unchanged before final envelope publication.

### S1-4 — completed epoch-30 recovery

Closed outside the scientific trainer.

A new execution-state helper recognizes an already-completed scientific checkpoint only when:

- the recovered candidate is the verified latest checkpoint;
- recovered epoch is at/after the locked final epoch;
- `batch_in_epoch == 0`;
- data-order state is at/after final epoch with next batch 0;
- optimizer step is positive;
- frozen selection history covers the locked epochs;
- selected checkpoint SHA is present and verifies under the exact identity.

When those conditions hold, `scripts/run_training.py` constructs terminal evidence from persisted metadata with:

- zero optimizer steps advanced;
- zero scheduler/training loop advancement;
- no validation recomputation;
- no invented metrics.

If any terminal condition is missing, normal resume remains authoritative.

## Failure-injection coverage

Wave-B tests include:

- child exits after >30 seconds but within finalization grace → no SIGKILL;
- hung child → SIGKILL after bounded finalization grace;
- publication FAIL classified as repair, not terminal skip;
- publication-complete PASS skipped safely;
- coherent prior bundle accepted;
- corrupted prior manifest rejected;
- coherent publication failure remains incomplete/repairable;
- publication repair explicitly does not relaunch training;
- epoch-30 completed checkpoint recovers with zero optimizer steps;
- partial final epoch is not promoted to terminal;
- terminal recovery is checked before trainer entry;
- global stop uses graceful finalization followed by normal result collection.

## Exact-head verification

Run #112 proved at exact head `66a5934b...`:

- expected SHA == actual SHA;
- compile PASS;
- repository validator PASS;
- JE static validator PASS;
- science-diff PASS;
- Stage-04A hash PASS;
- canonical notebook compile PASS;
- generator consistency PASS;
- forbidden-artifact scan PASS;
- secret scan PASS;
- active operator-doc validation PASS;
- Wave-A 18/18 PASS;
- Wave-B 13/13 PASS;
- MGPU 66/66 PASS;
- total 224/224 PASS.

## Residual durable-sync fact

Unchanged and intentionally not expanded in QA1:

- local periodic checkpoint cadence remains every 250 optimizer steps;
- Layer-B durable synchronization remains segment-boundary, not every local checkpoint.

# **WAVE B PASS — WAVE C MAY BEGIN**
