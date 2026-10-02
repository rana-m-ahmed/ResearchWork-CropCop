# Part 2 release status

## Production status

**QA-enhanced Supplement + reproducibility package: PASS within the explicit redistribution boundary.**

The final production pass preserves the frozen scientific results while improving reviewer readability, correcting stale metadata, and making availability boundaries explicit.

## Completed

- Supplement S1-S13 compiled as an Elsevier-style single-column supplement.
- Supplementary sections/tables use true S-numbering.
- Microtype evidence tables were removed; exhaustive inventories remain machine-readable.
- All main-manuscript numerical anchors and claim boundaries cross-checked.
- Track-A selector independently reproduced.
- Track-B screening, mapping, state-level results and exact-content aggregate sensitivity preserved.
- Track-C S1 identity, validation replay, calibration rule, FP32/INT8 identities and physical-device/runtime evidence preserved.
- S13 is framed as a **representative adjacent-work comparison**, with conservative `Yes` / `Unknown` / `Not applicable` semantics and 2026 EAAI context.
- Claim/evidence status updated from early discovery language to final closure states.
- Public/private redistribution boundaries documented.
- Final author metadata and declarations are closed in the submission package.

## Statistical boundary

The accepted external bootstrap is the prospectively frozen 5,000-replicate row-level cohort bootstrap. It is not identity-cluster robust. Exact-content multiplicity is addressed separately through the deterministic one-representative-per-SHA sensitivity analysis. No post-hoc identity-cluster interval is manufactured from aggregate results after the multiplicity outcome became known.

## Restricted or release-controlled

- consolidated source-image corpus;
- R07-S1 checkpoint;
- produced FP32 and INT8 PTE binaries;
- raw logits;
- complete accepted Track-B row-level evidence ZIP;
- immutable archival release/tag and optional DOI;
- any redistribution approval required by upstream licences.

Accepted Track-B evidence ZIP SHA-256:

`22a6c865ead6f28319a9dc8a4168f3ff7fb60108339710dcbd99e1aa047981a3`

Its accepted member inventory and a fail-closed no-inference verifier are preserved. Missing private bytes are not reconstructed from rounded manuscript prose.

## Reviewer navigation

Use [`../../docs/PAPER_EVIDENCE_AUTHORITY.md`](../../docs/PAPER_EVIDENCE_AUTHORITY.md) as the authority map for current journal evidence.
