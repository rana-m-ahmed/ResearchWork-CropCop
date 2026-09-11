# Secondary G1 transport hotfix v6

## Root cause

The qualified Secondary-G1 publisher stages a nested bundle and invokes `kaggle datasets version ... -r skip`. Kaggle defines `skip` as ignoring directories. Therefore the top-level seal can be published while nested model/evidence payload files are omitted. The local G1 seal remains valid, but the downloaded dataset cannot pass round-trip bundle validation.

## Scope

This hotfix is operator/wrapper-only. It does not change scientific source `8904b100d223e4319776199c87ab397db23600ce`, model configurations, seeds, dataset hashes, Smoke-A/B, Dual-T4 evidence, or `seal_secondary_g1.py`.

The replacement controller:

- regenerates and validates the exact local Secondary-G1 bundle using the frozen source;
- packages the full nested bundle as one deterministic uncompressed top-level tar;
- emits a per-member SHA/byte manifest and top-level Secondary-G1 seal;
- validates a local safe-extraction round trip before any remote mutation;
- detects whether the current exact Kaggle version already matches;
- otherwise publishes one new version, waits for the version number to advance, verifies the exact version inventory, downloads exact top-level transport files, safe-extracts the tar, and reruns `validate_secondary_g1_bundle`;
- emits terminal PASS only after exact remote round-trip verification.

Controller marker: `CROPCOP-K3-G1-TRANSPORT-FIX-V6`.
