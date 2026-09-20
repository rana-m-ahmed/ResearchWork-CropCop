# Track B R07 — Remediation Register v2

**Purpose:** record the operational reset from v3 all-in-one automation to the v4 two-notebook workflow.

No protected external R07 prediction was produced by the failed v3 qualification attempts.

## Observed v3 operational failures

1. Private R07 source datasets were initially inaccessible to the active Kaggle identity.
2. After permissions were corrected, the live GVLiD Mendeley-page resolver failed because the public webpage no longer exposed deterministic download URLs in the HTML shape assumed by the controller.
3. The all-in-one controller coupled source transport, core construction, historical-cache construction, external acquisition, qualification, claim execution, private publication and GitHub publication. Therefore non-scientific transport failures blocked the scientific workflow.

## v4 remediations

1. **Internal source acquisition removed from runtime.** Five internal sources are attached through Kaggle and read directly from `/kaggle/input`.
2. **One-time external acquisition isolated.** Mendeley/Zenodo acquisition exists only in Notebook 00.
3. **Historical comparison materialized once.** The 92,744-image safe comparison representation becomes part of the immutable infrastructure bundle.
4. **Two final bundles only.** Core+historical and GVLiD+Irish Potato are published as two paired private Kaggle datasets.
5. **Pairing identity added.** A shared `materialization_id` binds all four role manifests plus source/lock/attestation identities.
6. **Final execution has no live scientific downloads.** Notebook 01 consumes only attached immutable data.
7. **Qualification requires no credentials.**
8. **Claim uses only Kaggle credentials.** GitHub credentials are removed from the scientific path.
9. **Embedded repository execution.** Notebook 01 runs the repository snapshot included in the core package and performs no Git clone.
10. **GitHub publication moved post-closure.** Repository dissemination can fail/retry without authorizing any science rerun.
11. **Ephemeral readiness storage.** Heavy materialization lives under `/kaggle/tmp`; only the readiness receipt is persisted under `/kaggle/working`.
12. **Dedicated v4 static QA.** CI validates the two-notebook boundary, no live-source logic in Notebook 01, no GitHub secret use, exact load-bearing blobs, and qualification-without-secret behavior.

## Scientific non-change statement

The remediation does not alter candidates, mappings, model states, class map, preprocessing, metrics, audit thresholds, family support floor, bootstrap, family-order seed, historical comparison scope or evidence-grade rules.

## Current gate

Implementation may proceed to a real Kaggle Notebook-00 run after repository CI passes.

Protected external inference remains **HOLD** until:
- materialization PASS;
- qualification PASS;
- independent Q3 PASS;
- reviewed science digest authorization.

## Additional real-Kaggle readiness remediation

A real Notebook-00 run exposed a packaging ambiguity in the Final-V1 Kaggle dataset: the expected hash-valid `final_manifest.csv` and `class_to_idx.json` existed both in the image-backed `CropCop_Final_v1/audit/` tree and in `CropCop_Final_v1_CERTIFICATION_REPORTS/audit/`. The legacy core builder correctly failed closed because it required global uniqueness.

The v4 readiness layer now resolves exactly one authority pair only when both hash-valid files share a directory that is structurally bound to a real `dataset/train` + `dataset/val` tree. It then exposes a temporary canonical Final-V1 view to the unchanged legacy core builder. If two image-backed authority pairs exist, readiness still fails closed.

This remediation changes packaging resolution only; it does not modify Final-V1 bytes, hashes, split assignments, R07 science, or the v3 scientific attestation.

## R07 run-record packaging asymmetry remediation

A subsequent real Notebook-00 run passed Final-V1 canonicalization but failed during core construction because the attached S1 checkpoint dataset did not physically contain the exact frozen training `run_record.json`. A broader audit showed this was not a one-off missing file: R07 S1 is a historical state with an original public run record, whereas R07 S2/S3 are v1.2 continuation states for which Track-A explicitly required cryptographic terminal-metadata recovery.

The v4 readiness layer now follows the same frozen Track-A provenance paths used for final Track-A closure:

- S1 uses the exact original public run record from its frozen run-evidence branch and requires its SHA-256 to equal the already-frozen Track-B S1 run-record hash.
- S2/S3 use the frozen K3 public terminal-account report plus the exact attached selected checkpoints and Track-A terminal recovery code. The recovered records must byte-hash to the already-frozen Track-B S2/S3 run-record SHA-256 values before core construction.
- If an attached continuation dataset lacks a usable `checkpoint_index.json`, readiness may restore the same frozen durable checkpoint locator into scratch and performs the same recovery there.
- A new early source-qualification gate also validates Final-V1, all three R07 states, DINO, external-source reachability, and Kaggle publication identity before expensive materialization begins.

No R07 training, adaptation, model selection, thresholding, external prediction, or V1-test access is introduced by this remediation.
