# Stage 01A-G1P-v2 — Wave 0 Live State Audit

- **Artifact:** `01A_G1P_V2_00_LIVE_STATE_AUDIT.md`
- **Execution date:** 2026-09-06
- **Repository:** `rana-m-ahmed/ResearchWork-CropCop`
- **PR:** #8
- **Scope:** live authority / chronology / current-evidence verification only
- **Scientific execution performed:** NONE
- **Protected V1 test accessed:** NO

## 1. Live repository state

Independently reverified from the connected repository:

- `main`: `32190dd86293caa82170df3feea505e3c7443b4b`
- branch: `je-stage01a-core-20260905`
- PR #8: OPEN / DRAFT
- live PR head: `82fa51497af7ffd6d3a750f3c177363784f4f310`
- PR base: `32190dd86293caa82170df3feea505e3c7443b4b`

The master-prompt audit-state hypothesis for the live PR head is therefore confirmed.

## 2. Current canonical wrapper binding

`journal_extension/kaggle/generate_canonical_notebook.py` currently records:

```text
MGPU_EXECUTION_SOURCE_SHA = fe88e426b4698977d65efe9702f1d48cf5ff96a3
AUTHORIZED_SOURCE_SHA     = fe88e426b4698977d65efe9702f1d48cf5ff96a3
```

Therefore the currently qualified historical execution source is:

`fe88e426b4698977d65efe9702f1d48cf5ff96a3`

This source is not assumed to remain valid after G1P-v2 remediation.

## 3. Dependency lock

Current lock object:

`journal_extension/locks/execution_dependency_lock.json`

Current self-hash:

`767859a3cb1b37c0e50f753aa8c7a5c400424417cba3eaa59f0ab617ff18f701`

This is the dependency-lock identity bound by the already-qualified real Smoke A/B and dual-smoke evidence.

## 4. Authority stack

Live `journal_extension/locks/scientific_authority.json` records:

### Stage-03R scientific authority

- ID: `EAAI-JE-SDL-v2.1-QA`
- SHA-256: `aab17b65b0873dcb1ecedb061eb02ff60ccb09f8b830184f5e2231a600278f74`

### Stage-04 base execution architecture

- ID: `EAAI-JE-REA-v2.2-LEAN`
- SHA-256: `a1bc1f8bb2d7a133a46c3f59f83dd2ba8ca3bf0fa8d77ae323553276931bc079`

### Stage-04A additive MGPU amendment

- ID: `EAAI-JE-MGPU-A1`
- SHA-256: `3f08f2dbe7e83143c7e6f1fcf6732c7d88086295a600a74b323b89b46caed3f6`

The live `04A_EAAI_DUAL_GPU_EXECUTION_AMENDMENT_v1.md` carries the same authority stack.

## 5. Current exact-head CI

Latest exact-head workflow for PR head:

- Actions run: #136
- run ID: `33982202408`
- head SHA: `82fa51497af7ffd6d3a750f3c177363784f4f310`
- conclusion: SUCCESS

Run #136 proves:

- exact checkout SHA PASS;
- Python compile PASS;
- repository validator PASS;
- JE static validator PASS;
- MGPU science-diff sentinel PASS;
- Stage-04A amendment hash PASS;
- canonical notebook code compile PASS;
- generator == notebook PASS;
- forbidden tracked-model-artifact scan PASS;
- live-looking GitHub secret scan PASS;
- active QA1 operator-doc validator PASS;
- QA1 Wave A PASS;
- QA1 Wave B PASS;
- QA1 Wave C PASS;
- MGPU contract suite PASS;
- complete CPU-safe suite: **236 / 236 PASS**.

## 6. Science-diff state

The repository-held science sentinel and post-refactor report still bind the protected scientific surfaces.

Current verdict preserved:

`PASS — NO SCIENTIFIC DRIFT`

Protected objects include the exact Git blobs for:

- `journal_extension/src/cropcop_je/train.py`;
- `journal_extension/src/cropcop_je/data.py`;
- `journal_extension/src/cropcop_je/models.py`;
- CTC-v2;
- all six R04/R05 configs;
- experiment registry.

The Stage-03R model identities, seeds, objectives, batch semantics, 30-epoch duration and validation-only selection remain frozen.

## 7. Real Smoke A/B evidence chronology

Repository-held audit:

`01A_MGPU_SMOKE_AB_INDEPENDENT_AUDIT.md`

establishes:

### Smoke A
- execution source: `fe88e426b4698977d65efe9702f1d48cf5ff96a3`
- dependency lock: `767859a3cb1b37c0e50f753aa8c7a5c400424417cba3eaa59f0ab617ff18f701`
- terminal Batch result: PASS

### Smoke B
- execution source: `fe88e426b4698977d65efe9702f1d48cf5ff96a3`
- dependency lock: `767859a3cb1b37c0e50f753aa8c7a5c400424417cba3eaa59f0ab617ff18f701`
- terminal Batch result: PASS

Qualification ID:

`INFRA-SMOKE-fe88e426b469-f402d5b86f74`

These remain valid historical evidence for exactly the source/dependency pair above.

## 8. Real dual-GPU smoke evidence chronology

Repository-held audit:

`01A_MGPU_DUAL_GPU_SMOKE_INDEPENDENT_AUDIT.md`

establishes:

- execution source: `fe88e426b4698977d65efe9702f1d48cf5ff96a3`
- dependency lock: `767859a3cb1b37c0e50f753aa8c7a5c400424417cba3eaa59f0ab617ff18f701`
- qualification: `MGPU-DUAL-SMOKE-T4X2-V1`
- terminal Batch result: PASS
- science-diff status: PASS

This remains valid historical evidence for exactly that source/dependency pair.

It is **not** pre-authorized evidence for a future changed source or changed dependency lock.

## 9. Protected-result nonexistence check

The live tracked repository tree was searched for committed result artifacts indicating completed:

- G1 seal;
- G1 barrier;
- G2 barrier;
- pair-init model binaries;
- R04/R05 result/checkpoint artifacts;
- V1 continuity result;
- external prediction result;
- device result.

No protected result artifact was found. Only validator scripts named `validate_g1_barrier.py` and `validate_g2_barrier.py` matched the result-oriented search vocabulary.

Therefore no new JE protected result is represented by current repository state.

## 10. Wave-0 conclusion

The independent audit hypotheses for:

- live PR head;
- current wrapper source;
- dependency lock;
- authority stack;
- exact-head CI;
- historical Smoke A/B source binding;
- historical dual-smoke source binding;

are confirmed.

Historical real qualification remains preserved but source-scoped.

# **PASS — WAVE 0 LIVE AUTHORITY / CHRONOLOGY AUDIT COMPLETE**
