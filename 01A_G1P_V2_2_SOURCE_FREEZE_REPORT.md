# Stage-01A-G1P-v2.2 — Execution Source Freeze Record

**Status:** FROZEN SOURCE / PRE-WRAPPER-BINDING  
**Authority:** EAAI-JE-SDL-v2.1-QA  
**Main baseline:** `32190dd86293caa82170df3feea505e3c7443b4b`  
**Dependency lock SHA-256:** `6ea5fb51a0cc39c7940e4aaeb136d0f05214f5c558edd3c47b4cb61c6b516f37`

## Authoritative execution source

The immutable **Stage-01A-G1P-v2.2 execution source** is:

`5d70f85c2f1cd5b447040ede1eaeb4ad5331bca3`

This supersedes the former Stage-01A-G1P-v2.1 execution source:

`3c71331494b3e031bbbbc3f08d27cd2605c31097`

The superseded source and all source-bound qualification evidence remain historical and immutable. They are not valid qualification evidence for v2.2.

## Why this SHA is the execution source

Commit `5d70f85c2f1cd5b447040ede1eaeb4ad5331bca3` contains the complete v2.2 runtime implementation, including:

- canonical versioned tensor-identity hashing shared by MNV4 preparation and provenance;
- canonical `v1_test_accessed=false` seal semantics plus strict legacy compatibility;
- source-owned private-target settling;
- durable pre-mutation `G1_PUBLICATION_ATTEMPT.json`;
- version-transition publication barrier;
- exact-version Kaggle file listing/download and round-trip verification;
- idempotent explicit `g1-publication-repair`;
- stage diagnostics;
- terminal `run-evidence/G1` branch binding before Git publication.

The later verification head `c194d4d82d47d6dc3857cba4918a90fbc0c1e448` differs from the execution-source commit only in:

- `tests/test_je_qa1_wave_a.py`;
- `tests/test_je_smoke_sr.py`;
- `tests/test_je_pre_g1_integrity.py`.

No runtime source, dependency lock, scientific configuration, model identity, dataset identity, seed, Stage-03R authority, Stage-04 authority, or wrapper binding changed after the selected source commit and before verification closure.

## Acceptance evidence

GitHub Actions Run #225 (`34032833805`) at verification head `c194d4d82d47d6dc3857cba4918a90fbc0c1e448`:

- exact checked-out SHA: PASS;
- Python compilation: PASS;
- repository contract: PASS;
- JE static contract: PASS;
- MGPU science-diff sentinel: PASS;
- canonical notebook compilation: PASS;
- generator/notebook parity: PASS;
- forbidden model-artifact scan: PASS;
- GitHub-secret scan: PASS;
- operator-document validator: PASS;
- QA1 Wave-A: PASS (18/18);
- QA1 Wave-B: PASS;
- QA1 Wave-C: PASS;
- MGPU dual-envelope tests: PASS;
- complete CPU-safe suite: PASS (316 tests).

GitHub Actions Run #222 (`34032164980`), job `g1p-transformers-candidate`, on execution source `5d70f85c2f1cd5b447040ede1eaeb4ad5331bca3`:

- exact G1P dependency installation: PASS;
- `kaggle==2.2.4` publication API contract: PASS;
- offline DINOv3 / Transformers compatibility: PASS.

## Qualification reset

Wrapper binding is deliberately separate from this freeze record.

After the canonical wrapper is rebound to the v2.2 execution source, source-bound qualification must restart in this order:

```text
v2.2 readiness
→ fresh v2.2 Smoke A
→ fresh v2.2 Smoke B
→ independent audit
→ fresh v2.2 dual-GPU smoke
→ independent audit
→ fresh v2.2 CPU G1
→ independent terminal G1 audit
→ calibration-dual
```

Calibration-dual is not authorized by this freeze record.
