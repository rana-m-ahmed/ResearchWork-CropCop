# Stage 01A-MGPU — New-Source Smoke A/B Independent Audit

- **Audit date:** 2026-09-05
- **Execution source:** `fe88e426b4698977d65efe9702f1d48cf5ff96a3`
- **Qualification ID:** `INFRA-SMOKE-fe88e426b469-f402d5b86f74`
- **Smoke A archive:** user-supplied `results (2).zip`
- **Smoke B archive:** user-supplied `results (1).zip`
- **Verdict:** **PASS — TERMINAL SMOKE A/B QUALIFICATION COMPLETE**
- **Next authorized real gate:** `dual-gpu-smoke`
- **G1 authorized now:** **NO — only after independent dual-gpu-smoke audit**

## 1. Terminal execution state

Both terminal logs report:

- `KAGGLE_KERNEL_RUN_TYPE=Batch`;
- source `fe88e426b4698977d65efe9702f1d48cf5ff96a3`;
- dependency lock `767859a3cb1b37c0e50f753aa8c7a5c400424417cba3eaa59f0ab617ff18f701`;
- clean source checkout;
- phase-specific bootstrap PASS.

Smoke A terminates with:

- `STATUS=PASS`;
- checkpoint SHA `8c0a75e0f02f341254412606d51c927e1421ab19015547d93a499bfa34585d5f`;
- evidence SHA `48ecbf5e37a0399f12c7df706517a3ed051d8f3716d7b617e30a4bae5004650f`;
- `NEXT_PHASE=smoke-restore`.

Smoke B terminates with:

- `STATUS=PASS`;
- expected A checkpoint SHA = restored A checkpoint SHA =
  `8c0a75e0f02f341254412606d51c927e1421ab19015547d93a499bfa34585d5f`;
- `RESTORE_SUCCESS=true`;
- `RECOVER_SUCCESS=true`;
- `RESUME_SUCCESS=true`;
- optimizer step `3 -> 4`;
- B checkpoint SHA `5b4d3ecf22440009658a8ce10082355e17ac3c703368c1dfa01677b8e74c602b`;
- `SCIENTIFIC=false`;
- `SYNTHETIC_ONLY=true`.

## 2. Smoke A archive verification

`results (2).zip` contains the terminal Smoke-A export.

Verified:

- `SMOKE_A_EVIDENCE.json` status PASS;
- mode WRITE;
- Batch execution;
- CUDA available;
- exactly two Tesla T4 GPUs;
- source and tree identities correct;
- dependency lock correct;
- synthetic/non-scientific flags correct;
- no restricted CropCop data;
- G1/G2/R04/R05 all not executed;
- public evidence publication PASS;
- export integrity PASS;
- optimizer step = 3.

Cryptographic checks:

- checkpoint object bytes = 3749;
- checkpoint object SHA-256 =
  `8c0a75e0f02f341254412606d51c927e1421ab19015547d93a499bfa34585d5f`;
- checkpoint index points to the same object/SHA/step;
- checkpoint-index SHA matches manifest recovery record;
- checkpoint-object SHA matches manifest recovery record;
- raw `SMOKE_A_EVIDENCE.json` SHA-256 =
  `48ecbf5e37a0399f12c7df706517a3ed051d8f3716d7b617e30a4bae5004650f`;
- manifest stores exactly that evidence SHA;
- manifest self-hash recomputes exactly to
  `cb40dfd5728b8969dee9b8caa3769a0574d558e9540dcb4c46cc4b03b57f2961`.

The frozen-source `verify_smoke_a_export(...)` verifier accepted this archive without error.

## 3. Smoke B archive and A→B handoff verification

`results (1).zip` contains the terminal Smoke-B export and restored-A recovery copy.

Verified A binding:

- qualification ID equals Smoke A;
- source SHA equals Smoke A;
- source tree equals Smoke A;
- dependency lock equals Smoke A;
- Smoke-A manifest SHA equals
  `cb40dfd5728b8969dee9b8caa3769a0574d558e9540dcb4c46cc4b03b57f2961`;
- Smoke-A evidence SHA equals
  `48ecbf5e37a0399f12c7df706517a3ed051d8f3716d7b617e30a4bae5004650f`.

The restored A checkpoint in the B archive is byte-for-byte identical to the checkpoint object in the A archive.

Exact restored SHA:

`8c0a75e0f02f341254412606d51c927e1421ab19015547d93a499bfa34585d5f`

Required sequence is exact:

`READ_A -> VERIFY_A -> RESTORE_A -> RECOVER_A -> LOAD_A -> RESUME -> CHECKPOINT_B`

Optimizer progression is exact:

`3 -> 4`

B checkpoint verification:

- object bytes = 3749;
- SHA-256 =
  `5b4d3ecf22440009658a8ce10082355e17ac3c703368c1dfa01677b8e74c602b`;
- checkpoint index points to step 4 and the same SHA;
- index/object recovery hashes match the B manifest;
- raw `SMOKE_B_EVIDENCE.json` SHA-256 =
  `8437fd310f37ee5f0489b4df6418684eba76b4b324899add985646b96ddb8c48`;
- B manifest stores exactly that evidence SHA;
- B manifest self-hash recomputes exactly to
  `e54b8b67d7889d299d654d82dccf620116e437ed2982ae4bba6a6a30c67e9370`.

The frozen-source `validate_terminal_smoke_b_evidence(..., require_batch=True)` validator returned an empty error list.

## 4. Runtime/environment consistency

Smoke A and B are separate Batch sessions, as required.

Both expose:

- two Tesla T4 GPUs;
- CUDA runtime 13.0;
- cuDNN 92000;
- Python 3.12.13;
- torch 2.12.1;
- torchvision 0.27.1;
- timm 1.0.26;
- numpy 2.5.2;
- Pillow 12.3.0;
- safetensors 0.8.0;
- Kaggle 2.2.4.

Different physical GPU UUIDs between A and B are expected because they are separate fresh Kaggle Saved-Version sessions.

## 5. Public evidence publication

The connected GitHub repository contains both claimed evidence branches:

- `run-evidence/SMOKE-A-INFRA-SMOKE-fe88e426b469-f402d5b86f74`;
- `run-evidence/SMOKE-B-INFRA-SMOKE-fe88e426b469-f402d5b86f74`.

The Git blob identity of each published evidence JSON exactly matches the corresponding evidence file in the user-supplied archive, proving the published evidence bytes and archived evidence bytes are identical.

## 6. Safety/science boundary

Both A and B evidence explicitly report:

- `scientific=false`;
- synthetic/unprotected data only;
- restricted CropCop data not accessed;
- G1 not executed;
- G2 not executed;
- R04/R05 not executed.

No scientific result is inferred from Smoke A/B.

## 7. Verdict

# **PASS — NEW-SOURCE REAL SMOKE A/B QUALIFICATION COMPLETE**

The chronology gate now advances exactly one step:

# **AUTHORIZED NEXT: `dual-gpu-smoke`**

G1 remains blocked until the resulting dual-GPU-smoke evidence is independently audited and passes the canonical terminal dual-smoke validator.
