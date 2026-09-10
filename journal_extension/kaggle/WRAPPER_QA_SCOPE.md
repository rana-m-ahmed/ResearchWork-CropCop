# Secondary Kaggle Wrapper QA Scope

Execution source under qualification: `8904b100d223e4319776199c87ab397db23600ce`.

This wrapper layer is non-scientific orchestration. It must not replace the execution source SHA or modify protected principal/secondary scientific files.

Independent acceptance checks:

- three notebooks are valid nbformat 4.5 and every code cell compiles;
- exact notebook bytes are bound by `secondary_wrapper_manifest.json`;
- `KAGGLE_KERNEL_RUN_TYPE=Batch` is enforced before secret/network work;
- notebook-global monotonic clock starts before clone/install;
- clone/check-out is hard-bound to the qualified execution source;
- attached evidence is selected by exact source/status/seal identity, not filename alone;
- K1/K2/K3 account and envelope mappings are fixed;
- Kaggle API credentials are consumed only by phases that need them;
- T4x2 phases reject any other GPU topology;
- durable locator templates preserve literal `{run_id_lower}` substitution for the qualified runner;
- no V1-test/protected external data is attached or referenced by the controllers;
- no model/checkpoint/archive/secret artifact is committed;
- principal and secondary science-diff validators remain PASS.

A green wrapper CI is necessary but not sufficient for scientific launch. K3 real source-qualified Smoke A/B, dual-T4 smoke, Secondary G1, and Secondary G2 remain runtime gates before K1/K2 scientific execution.
