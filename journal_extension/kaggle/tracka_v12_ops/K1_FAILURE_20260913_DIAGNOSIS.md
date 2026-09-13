# K1 master failure diagnosis — 2026-09-13

Observed failing runtime: `f606311ec5faa2d4d1a2ea44b279357bfd15fb84`.
Frozen science source remained `9a72e9466a9a3e7429e0e36a028edac662f83146`.

## Root cause

The run did **not** enter G1A and did not train any scientific state. The exact dependency repair completed, but the subsequent frozen-input resolver found zero files under `/kaggle/input` with the required manifest SHA-256:

`bdb82211ccc2059153724eea178a1680893a6b38ecc243fae484baa91dbf68e2`

Therefore the attached Kaggle inputs either omitted the frozen CropCop V1 dataset or contained a different manifest revision. This is an input/preflight failure, not a scientific failure.

## Remediation

The master runtime now:

1. resolves the exact manifest/class map/image root **before** science checkout or dependency installation;
2. on failure, prints mounted `/kaggle/input` entries plus likely metadata paths, byte counts and SHA-256 values;
3. preflights the historical principal G1 bundle on K1 before dependency installation;
4. retains strict SHA/structure validation and does not accept a similar filename as a substitute.

No scientific code, protocol, seeds, model definitions or protected-surface policy changed.
