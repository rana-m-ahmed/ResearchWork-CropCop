# Part 2 release status

## Production status

**QA-enhanced Supplement + reproducibility package: PASS within the explicit redistribution boundary.**

The final production pass preserves all frozen Part-1 science while improving reviewer readability and correcting stale early-stage QA metadata.

## Completed

- Supplement S1–S13 compiled as an Elsevier-style single-column supplement.
- Supplementary sections/tables use true S-numbering.
- Microtype evidence tables were removed; exhaustive inventories remain machine-readable.
- All Part-1 numerical anchors and claim boundaries cross-checked.
- Track-A selector independently reproduced.
- Track-B screening, mapping, state-level results and exact-content aggregate sensitivity preserved.
- Track-C S1 identity, validation replay, calibration rule, FP32/INT8 identities and physical-device/runtime evidence preserved.
- Closest-prior-work matrix retains conservative `Yes` / `Unknown` / `Not applicable` semantics.
- Claim/evidence status updated from early discovery language to final closure states.
- Public/private redistribution boundaries documented.

## Restricted or author-controlled

- consolidated source-image corpus;
- R07-S1 checkpoint;
- produced FP32 and INT8 PTE binaries;
- raw logits;
- complete accepted Track-B row-level evidence ZIP;
- final EAAI author metadata/declarations;
- final immutable archival release/tag and optional DOI.

Accepted Track-B evidence ZIP SHA-256:

`22a6c865ead6f28319a9dc8a4168f3ff7fb60108339710dcbd99e1aa047981a3`

Its accepted member inventory and a fail-closed no-inference verifier are preserved. Missing private bytes are not reconstructed from rounded manuscript prose.

## Reviewer navigation

Use [`../../docs/PAPER_EVIDENCE_AUTHORITY.md`](../../docs/PAPER_EVIDENCE_AUTHORITY.md) as the authority map for current journal evidence.
