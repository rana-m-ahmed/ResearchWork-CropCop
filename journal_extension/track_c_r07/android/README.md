# R07 Track-C Android benchmark harness

This is a deliberately small, research-only Android application for the frozen
R07-S1 runtime. It is not a CropCop product application and contains no UI,
database, cloud, camera, LLM, or account code.

It accepts one `trackc_mode` intent extra:

- `tensor_fidelity`
- `raw_fidelity`
- `load`
- `warmup_transition`
- `model_latency`
- `end_to_end`
- `system_snapshot`

The harness refuses all claim-producing modes until a separately provisioned,
hash-verified artifact and sealed input manifests are available. The required
runtime file is not tracked in Git:

```text
<app files>/trackc/r07_s1_xnnpack_int8.pte
sha256: 2e0c54a1b5bb7c0018d0159a642e49e4f4bd79fcc1f940e215a40734a2695bb8
bytes: 28555872
```

The expected runtime is `org.pytorch:executorch-android:1.3.1`, XNNPACK CPU,
with `Module.LOAD_MODE_FILE` and exactly four requested threads. Artifact
provisioning and input-manifest transfer are intentionally outside the APK
build and outside the timed region.

From a machine with the existing Android Gradle wrapper, build without adding
the restricted artifact:

```powershell
& D:\projects\CropCop-Mobile\apps\mobile\android\gradlew.bat `
  -p D:\projects\ResearchWork-CropCop\journal_extension\track_c_r07\android\benchmark_harness `
  :app:testDebugUnitTest :app:assembleDebug
```

Physical execution is not authorized until `DS-DEVICE-RAW-256-R07`,
`DS-DEVICE-TENSOR-256-R07`, and the Track-C analysis lock are frozen.
