# Stage-01A-G1P-v2.2 — G1 Infrastructure Forensic Remediation Record

Status: SOURCE REMEDIATION INPUT  
Historical execution source: `3c71331494b3e031bbbbc3f08d27cd2605c31097`  
Dependency lock: `6ea5fb51a0cc39c7940e4aaeb136d0f05214f5c558edd3c47b4cb61c6b516f37`

The latest old-source CPU G1 attempt is retained as forensic evidence only. It is **not** terminal G1 evidence.

It successfully reached Batch/source/dependency validation, exact Final-V1 and RFDV resolution, private-target creation/settling, MobileNetV4 official-timm provenance, teacher reconstruction/EMA/adapter/class-order checks, pair S1/S2/S3 creation, G1 seal creation, and G1 barrier PASS.

Observed old-source seal:
`f40a84a0e96eafe755b4bde98627a49fd144455ba75ead8faf1d0befaa0b8262`

Observed pair artifacts:
- S1 / seed 21270083 / `7040e48fe697c539dbbdfa8f57ffd20e9327def82f6485267f224dcf4c34cce1`
- S2 / seed 606135704 / `24a93cf97a4564d9484b80f15e28359c8d346a9ef62b7192931ff337bd1a5182`
- S3 / seed 1153870846 / `e8283753463dbd3c9743a9a09f0a7933bbfcee158fc82c42edf413d7ec81d121`

The fatal old-source failure occurred after the Kaggle version command returned, during publication round-trip. The implementation waited only for generic dataset `ready`, so the already-ready placeholder version could satisfy the gate before the newly uploaded version became current. Bare-slug download then retrieved the stale version and `locate_g1_package` correctly rejected it because the required package + manifest were absent.

Independent source audit also confirmed:
1. frozen V1 identity writer/validator field mismatch;
2. two incompatible MobileNetV4 tensor-identity delimiter implementations;
3. target-settling semantics had leaked into the notebook wrapper;
4. terminal evidence was published before its deterministic `run-evidence/G1` branch field was inserted.

No old evidence is modified by this remediation. No old seal is promoted as a v2.2 terminal seal. The existing private Kaggle target is retained for safe versioned transport reuse.
