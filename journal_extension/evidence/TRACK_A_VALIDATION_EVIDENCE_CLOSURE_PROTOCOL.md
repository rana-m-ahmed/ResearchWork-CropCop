# Track-A validation-evidence closure protocol

Status: **PRE-EVIDENCE LOCK**

This file records the pre-result evidence-completion method for the ten already-frozen Track-A model states. It is intentionally result-free.

- Scientific authority: `EAAI-JE-SDL-v2.1-QA` / `aab17b65b0873dcb1ecedb061eb02ff60ccb09f8b830184f5e2231a600278f74`
- Principal source: `f171309fc7e9dc22241ecc137ebbb8e4bcdc5433`
- Secondary closure source: `8904b100d223e4319776199c87ab397db23600ce`
- Final-V1 manifest: `bdb82211ccc2059153724eea178a1680893a6b38ecc243fae484baa91dbf68e2`
- Class map: `46f7811726c19c42bd7213b2d8178b19a5a182a1b763f60a94ee2c0e5f6688d2`
- Authorized replay surface: `DS-V1-VAL`, exactly 16,368 rows
- Consumed V1 test: **not authorized**
- Metric replay tolerance: maximum absolute difference `1e-6`
- No training, optimizer advance, retuning, new seeds, new models, or protected external inference.

Each run closes only when its public evidence branch contains exactly:

1. `run_record.json`
2. `metrics.json`
3. `segments.jsonl`
4. `validation_evidence.json`
5. `validation_predictions.jsonl`

The two newly generated files must byte-match the locally verified outputs. After all ten runs pass this protocol, repository closure must create `02_EAAI_PRINCIPAL_MODEL_SCIENCE_RESULTS.md`, `03A_EAAI_BASELINES_ABLATIONS_RESULTS.md`, and `TRACK_A_CLOSURE_MANIFEST.json` without modifying the frozen scientific configurations or training implementation.

The machine-readable authority is `journal_extension/evidence/track_a_validation_evidence_closure_protocol.json`.
