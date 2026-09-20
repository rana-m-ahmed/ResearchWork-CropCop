# Track B R07 — Remediation Register v1

**Baseline:** `ef5e2f9d4b8b542e13f33ba5446672f0e9cd60c0`  
**Audited implementation code head:** `64f6202dfab9d4972c14557f6506311ef5b4b614`  
**Branch:** `trackb-r07-remediation-v3-20260920`  
**Status:** **PASS — implementation remediation complete; real Kaggle qualification still required.**

No protected external R07 inference has been executed as part of this remediation. The frozen science is unchanged: GVLiD v5 + Irish Potato Version 01, R07 S1/S2/S3, native 120-way scoring, the same mappings/audit thresholds/bootstrap/support floor, V1 train+validation historical comparison only, EXT-S ceiling, no training, and the consumed V1 test closed.

## Closed findings

1. **Lock/runtime divergence — FIXED.** A validated `TrackBAuditPolicy` now drives pHash, dHash, DINO, ORB, support, bootstrap and family-order settings.
2. **Unused family-order seed — FIXED.** The frozen seed `1936263114` now drives deterministic PCG64 ordering before candidate sealing.
3. **Candidate ORB memory growth — FIXED.** Candidate ORB features are packed/file-backed in scratch and read by mmap.
4. **CPU oversubscription — FIXED.** OpenCV internal threading is fixed to one; worker concurrency remains bounded.
5. **Persistent raw/model storage — FIXED.** Heavy/raw/model material lives under `/kaggle/tmp`; final persistent hygiene rejects raw images, model files and partial downloads.
6. **Source integrity — FIXED.** Zenodo archive checksums and the GVLiD source SHA-256 ledger are mandatory; downloads are retried, atomic and checksum-verified.
7. **Lineage evidence — FIXED.** `TRACKB_EXTERNAL_LINEAGE_REVIEW_v1` is pre-results frozen and bound to candidate metadata; residual uncertainty is retained.
8. **Transport-dependent identity — FIXED.** Canonical member identity binds raw SHA-256 plus source-member identity; duplicate transport downloads are content-deduplicated.
9. **Protected-attempt chronology — FIXED.** A durable private Kaggle attempt ledger is written before the first protected R07 forward pass; changed-science restart is rejected.
10. **Premature closure — FIXED.** Closure is written only after independent QA, science manifest, package creation, CRC/member-hash verification and package manifest.
11. **Stable scientific identity — FIXED.** `trackb_science_sha256` is separate from timestamp/duration-dependent execution closure.
12. **Private publication durability — FIXED.** Publication is content-addressed/idempotent, Ready-state checked, and terminal restricted evidence is round-trip downloaded and byte-verified.
13. **GitHub failure recovery — FIXED.** Private evidence becomes durable first; GitHub failure does not authorize a science rerun.
14. **Stale v2 readiness — SUPERSEDED.** v2 reports remain historical; v3 is the current operational readiness record.
15. **Repository-wide CI incompatibilities — FIXED.** The generic Python-3.11 suite uses a compatible NumPy, while the dedicated Track-B suite continues to validate the frozen Python-3.12/NumPy-2.5.2 scientific stack.

## Exact-head verification

At `64f6202dfab9d4972c14557f6506311ef5b4b614`:

- Track B Infrastructure QA (PR): run **35492138674** — PASS
- Track B Infrastructure QA (push): run **35492136557** — PASS
- Validate secondary Track-A wave: run **35492138642** — PASS
- Validate public evidence / complete CPU-safe suite: run **35492138673** — PASS

## Remaining gates

Implementation QA is **not** the claim run. Before any protected external prediction:

- **Q1:** real Kaggle environment/transport smoke;
- **Q2:** full real prediction-blind audit on GVLiD + Irish Potato, stopping before R07 external inference;
- **Q3:** independent verification of Q2, including source/seal/family hashes, no V1-test access, and absence of protected external predictions.

Only after Q1–Q3 pass may the final protected Track-B claim run be authorized.
