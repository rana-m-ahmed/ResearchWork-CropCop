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
