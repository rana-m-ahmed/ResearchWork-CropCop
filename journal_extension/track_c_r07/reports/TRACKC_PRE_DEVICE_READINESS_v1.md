# Track-C R07 device readiness and execution status

Status: `PASS_TRACKC_PHYSICAL_EXECUTION_EVIDENCE_COLLECTED`

The prior materialization blocker is superseded. The user-supplied restricted
Track-C input bundle was independently hash-validated, selected from the
locked DS-V1-VAL authority, and staged only in the benchmark application's
private storage. Restricted raw images, tensors, model binaries, and archive
bytes are not committed to this repository.

The exact selected INT8 XNNPACK artifact was loaded and evaluated on a
physical ARM64 Google Pixel 7. Canonical-tensor host/device fidelity was
256/256, satisfying the prospectively locked >=99.5% engineering criterion.
The matched FP32 artifact was also loaded and achieved 256/256 on the same
canonical tensor set. The public-safe raw evidence, checksums, independent
analysis, and fail-closed evidence validator are under `results/public/` and
`analysis/`.

The raw Android image pipeline achieved 254/256 top-1 agreement after
semantic alignment of EXIF handling, RGB conversion, mean-color square pad,
antialiased bicubic resize, uint8 quantization, and normalization. Its two
remaining disagreements are retained as a bounded implementation difference;
they do not alter the canonical-tensor runtime-fidelity acceptance result.

See `TRACKC_R07_DEVICE_EXECUTION_REPORT.md` for device-specific metrics and
claim boundaries.
