# Track-C R07 secondary-device replication: POCO M3

This is a secondary hardware-tier replication attempt only. It does not alter,
replace, or reinterpret the closed primary Pixel 7 experiment.

## Device and frozen inputs

The borrowed device was a Xiaomi POCO M3 (`M2010J19CG`, `citrus` / Qualcomm
SM6115 `bengal`), Android 12 / SDK 31. It advertised `arm64-v8a`, had
3,802,472 KB RAM and 103,511,332 KB available `/data` storage at preflight.
It was charging with thermal status 0. The exact primary Track-C APK, frozen
INT8 artifact, input archive, RAW/TENSOR manifests and input lock were copied
to app-private storage and each required identity matched.

## Result

The harness successfully completed its exact-artifact load preflight. The
first canonical TENSOR-256 forward then terminated with native `SIGILL` in
`libexecutorch_jni.so`; Android recorded `APP CRASH(NATIVE)`, status 4, and no
tensor-fidelity rows were emitted. The CPU feature listing was ARM64 but did
not include `asimddp`/dot-product support. This is evidence of a runtime
compatibility blocker for this frozen artifact/runtime on this POCO device.

Accordingly, no 1,000-forward latency, RAW-256 E2E, memory, thermal-under-load
or host/device fidelity result is claimed for the POCO M3. The artifact was
not re-exported, re-quantized, retrained, or modified, and no alternate runtime
was substituted. The package under `results/public/device2_poco_m3/` contains
the public-safe preflight and failure certificate.

This result does not support an Android-wide compatibility or performance
claim. It documents one explicitly identified, lower-tier ARM64 configuration.
