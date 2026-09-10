# Secondary Kaggle Controller Wrapper Release Note

Wrapper branch: `je-tracka-secondary-wrappers-20260910`.

Immutable scientific execution source: `8904b100d223e4319776199c87ab397db23600ce`.

The wrapper branch adds only operator notebooks, their byte-hash manifest, exact input/runbook documentation, wrapper QA tests, and wrapper CI. It does not authorize a new science source.

Controller roles:

- K3 `ranamuhammadahmed6`: new-source Smoke A/B, dual-T4 smoke, Secondary G1, Secondary G2, optional later S3 evidence backfill.
- K1 `ranaabdulrehmannn`: R12 logits + R12 feature after G2 PASS.
- K2 `sabahatabbas`: R06 EfficientNet-B0 + R07 ConvNeXt-Tiny after G2 PASS.

The downloadable notebook pack must match the notebook hashes recorded in `secondary_wrapper_manifest.json` before use.
