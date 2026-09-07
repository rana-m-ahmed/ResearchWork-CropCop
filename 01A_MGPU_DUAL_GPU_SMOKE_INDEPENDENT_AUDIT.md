# Stage 01A-MGPU — Real Dual-GPU Smoke Independent Audit

- **Audit date:** 2026-09-05
- **Execution source:** `fe88e426b4698977d65efe9702f1d48cf5ff96a3`
- **Dual-smoke qualification:** `MGPU-DUAL-SMOKE-T4X2-V1`
- **User archive:** `results (4).zip`
- **Verdict:** **PASS — REAL DUAL-GPU SMOKE QUALIFICATION COMPLETE**
- **Next authorized gate:** **G1**
- **G2/calibration authorized now:** NO
- **R04/R05/principal authorized now:** NO

## 1. Terminal Batch/source gate

The supplied terminal log and archived bootstrap evidence agree on:

- phase: `dual-gpu-smoke`;
- Kaggle run type: `Batch`;
- source SHA: `fe88e426b4698977d65efe9702f1d48cf5ff96a3`;
- source tree: `625792b4d4a155eac415da88a41a586d04daa6c3`;
- dependency lock:
  `767859a3cb1b37c0e50f753aa8c7a5c400424417cba3eaa59f0ab617ff18f701`;
- clean source/bootstrap status: PASS.

## 2. Smoke-B chronology binding

Dual-smoke evidence records:

`smoke_b_evidence_sha256 = e44f400c934e07b9c736330d81887a85965df947af61666c5d850c15fdf1eac4`

This is deliberately the repository's `sha256_json(smoke_b_object)`, not the raw evidence-file SHA used by the Smoke-B manifest.

Recomputed against the previously audited successful Smoke-B archive:

- raw `SMOKE_B_EVIDENCE.json` file SHA-256:
  `8437fd310f37ee5f0489b4df6418684eba76b4b324899add985646b96ddb8c48`;
- canonical JSON/object SHA-256:
  `e44f400c934e07b9c736330d81887a85965df947af61666c5d850c15fdf1eac4`.

The canonical digest exactly equals the dual-smoke chronology binding.

Therefore the real dual-GPU smoke is cryptographically tied to the independently audited terminal Smoke-B object from the same source.

## 3. Parent physical T4 inventory

Parent evidence reports exactly two physical GPUs:

### Slot 0
- name: Tesla T4
- UUID: `GPU-26d6b07b-7e29-4e2b-c6fb-ddd84a33c265`
- memory: 15360 MiB

### Slot 1
- name: Tesla T4
- UUID: `GPU-b0de92d3-a9e3-9c34-0bd1-dc1fa75f9a3c`
- memory: 15360 MiB

The UUIDs are nonempty and distinct.

## 4. Child isolation

### DUAL-SMOKE-A
- requested physical slot: 0
- bound physical UUID:
  `GPU-26d6b07b-7e29-4e2b-c6fb-ddd84a33c265`
- visible CUDA device count: 1
- visible GPU: Tesla T4
- Git credentials in child: false
- optimizer step: 4
- checkpoint bytes: 3365
- checkpoint SHA-256:
  `9bc5c6d136733cc1947de7508fd54283d6ffde3d48278013f6ff4342da73afe0`

### DUAL-SMOKE-B
- requested physical slot: 1
- bound physical UUID:
  `GPU-b0de92d3-a9e3-9c34-0bd1-dc1fa75f9a3c`
- visible CUDA device count: 1
- visible GPU: Tesla T4
- Git credentials in child: false
- optimizer step: 4
- checkpoint bytes: 3365
- checkpoint SHA-256:
  `57265a9056a3893e772b77aaa0682e28b2632bf969794e3f30942a6c4cd944d7`

Both requested slot→physical UUID mappings exactly match the parent inventory.

Both children inherit the exact notebook-global monotonic start value:

`275.936269583`

## 5. Checkpoint byte verification

The archived child checkpoint files were independently hashed.

### A
Actual file SHA-256:

`9bc5c6d136733cc1947de7508fd54283d6ffde3d48278013f6ff4342da73afe0`

Actual file size:

`3365` bytes

Both exactly match child and parent evidence.

### B
Actual file SHA-256:

`57265a9056a3893e772b77aaa0682e28b2632bf969794e3f30942a6c4cd944d7`

Actual file size:

`3365` bytes

Both exactly match child and parent evidence.

No checkpoint payload was modified during the audit.

## 6. True concurrent overlap

Archived child timings:

- A:
  - start `438.762896103`
  - end `442.740002539`
- B:
  - start `438.67440345`
  - end `442.748510937`

The independently recomputed interval intersection is:

`min(end_A, end_B) - max(start_A, start_B)`

= `442.740002539 - 438.762896103`

= `3.9771064359999855 seconds`

This exactly equals the parent evidence field:

`overlap_duration_seconds = 3.9771064359999855`

Therefore both isolated children were active concurrently.

## 7. Safety/non-scientific boundary

Terminal dual-smoke evidence explicitly records:

- `scientific=false`;
- `synthetic_unprotected_data_only=true`;
- `restricted_cropcop_data_accessed=false`;
- `g1_executed=false`;
- `g2_executed=false`;
- `r04_r05_executed=false`;
- `no_git_child_publication=true`;
- `no_output_collision=true`;
- `common_session_clock=true`;
- `parent_finalized_both=true`;
- `science_diff_status=PASS`;
- `errors=[]`.

No CropCop scientific result is inferred from this test.

## 8. Frozen-source machine validator

The exact frozen-source implementation from
`fe88e426b4698977d65efe9702f1d48cf5ff96a3`
was used to invoke:

`validate_terminal_dual_gpu_smoke_evidence(..., require_batch=True)`

against the user's archived evidence, with:

- expected source SHA;
- expected dependency-lock SHA;
- exact Stage-04A ID/hash;
- canonical successful Smoke-B evidence digest.

Returned errors:

`[]`

Therefore the evidence satisfies the repository's own terminal dual-smoke contract.

## 9. Public evidence branch

Claimed branch:

`run-evidence/DUAL-GPU-SMOKE`

was independently fetched from the connected GitHub repository.

Published evidence Git blob SHA-1:

`793cb6966cefb1fa8cc2356b70fd757b70a701b9`

The archived `DUAL_GPU_SMOKE_EVIDENCE.json` independently recomputes to the exact same Git blob SHA-1:

`793cb6966cefb1fa8cc2356b70fd757b70a701b9`

Raw archived evidence SHA-256:

`e0d5b5eacb8a519a4f9383cb48c61901ea1fb8fb61e12262e1f2978579c9123b`

Thus the public Git evidence bytes and the user-supplied archived evidence bytes are identical.

## 10. Secret/output audit

No live-looking GitHub credential was found in the generated dual-smoke output.

The only token-like match in the full supplied ZIP is the repository's intentional synthetic credential fixture in:

`tests/test_je_failure_injection_matrix.py`

This is not a live credential.

## 11. Final chronology gate

Completed and independently audited:

1. real new-source Smoke A — PASS;
2. fresh real new-source Smoke B — PASS;
3. A/B independent audit — PASS;
4. real dual-GPU smoke — PASS;
5. dual-GPU-smoke independent audit — PASS.

# **PASS — REAL DUAL-GPU SMOKE QUALIFICATION COMPLETE; G1 MAY BEGIN**

The chronology advances exactly one gate.

G1 may now be prepared/executed using the same frozen source and the exact audited Smoke-B and dual-smoke evidence.

G2/calibration and principal execution remain blocked until G1 is completed and independently validated.
