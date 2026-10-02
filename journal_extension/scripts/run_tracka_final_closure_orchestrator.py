from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from pathlib import Path

ANALYSIS_SHA = "e08e471cc033902694622c12ceef03a19707b6dc"
TRAINING_SHA = "56023042e57758591df9babb3438f191dbe10312"
HISTORICAL_SHA = "f171309fc7e9dc22241ecc137ebbb8e4bcdc5433"
INTEGRATION_BRANCH = "tracka-final-closure-20260919"
FINAL_REL = Path("journal_extension/evidence/public/track_a/final")


def run(args: list[str], cwd: Path, *, capture: bool = False) -> str:
    cp = subprocess.run(args, cwd=cwd, check=False, text=True, capture_output=capture)
    if cp.returncode != 0:
        detail = (cp.stderr or cp.stdout or "").strip()
        raise RuntimeError(f"command failed rc={cp.returncode}: {' '.join(args)}\n{detail[-4000:]}")
    return cp.stdout if capture else ""


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_hash(obj: dict) -> str:
    raw = json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def load_json(path: Path) -> dict:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise RuntimeError(f"JSON object required: {path}")
    return obj


def write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def fetch_branch(repo: Path, branch: str) -> str:
    remote = f"refs/remotes/origin/{branch}"
    run(["git", "fetch", "origin", f"refs/heads/{branch}:{remote}"], repo)
    return remote


def show_bytes(repo: Path, ref: str, path: str) -> bytes:
    cp = subprocess.run(["git", "show", f"{ref}:{path}"], cwd=repo, check=False, capture_output=True)
    if cp.returncode != 0:
        raise RuntimeError(f"git show failed: {ref}:{path}\n{cp.stderr.decode(errors='replace')[-2000:]}")
    return cp.stdout


def materialize_readiness(repo: Path, inputs: Path) -> None:
    inputs.mkdir(parents=True, exist_ok=True)
    global_branch = "run-evidence/TRACKA-PREGPU-READINESS-GLOBAL-e08e471cc033"
    remote = fetch_branch(repo, global_branch)
    path = (
        "journal_extension/evidence/public/track_a/pregpu/"
        "TRACKA-PREGPU-READINESS-GLOBAL-e08e471cc033/POSTTRAINING_READINESS_GATE.json"
    )
    (inputs / "global_readiness.json").write_bytes(show_bytes(repo, remote, path))

    for account in ("K1", "K2", "K3"):
        branch = f"run-evidence/TRACKA-PREGPU-READINESS-{account}-e08e471cc033"
        remote = fetch_branch(repo, branch)
        path = (
            "journal_extension/evidence/public/track_a/pregpu/"
            f"TRACKA-PREGPU-READINESS-{account}-e08e471cc033/"
            f"{account}_POSTTRAINING_ACCOUNT_READINESS.json"
        )
        (inputs / f"{account}_readiness.json").write_bytes(show_bytes(repo, remote, path))


def reconstruct_account_completions(repo: Path, inputs: Path) -> None:
    global_gate = load_json(inputs / "global_readiness.json")
    if (
        global_gate.get("status") != "PASS"
        or global_gate.get("analysis_source_git_commit") != ANALYSIS_SHA
        or global_gate.get("state_count") != 21
        or global_gate.get("direct_state_count") != 12
        or global_gate.get("auxiliary_state_count") != 9
        or global_gate.get("unique_selected_checkpoint_count") != 21
        or global_gate.get("v1_test_closed") is not True
        or global_gate.get("external_predictions_closed") is not True
    ):
        raise RuntimeError("global readiness gate is not the expected 21-state PASS")

    union: set[str] = set()
    for account in ("K1", "K2", "K3"):
        readiness_path = inputs / f"{account}_readiness.json"
        readiness = load_json(readiness_path)
        if readiness.get("status") != "PASS" or readiness.get("analysis_source_git_commit") != ANALYSIS_SHA:
            raise RuntimeError(f"{account} readiness invalid")
        states = {}
        for experiment_id in sorted(readiness.get("states", {})):
            if experiment_id in union:
                raise RuntimeError(f"duplicate account placement: {experiment_id}")
            union.add(experiment_id)
            run_id = f"TRACKA-POST-{experiment_id.lower()}-e08e471cc033"
            publication_branch = f"run-evidence/{run_id}"
            remote = fetch_branch(repo, publication_branch)
            if subprocess.run(
                ["git", "merge-base", "--is-ancestor", ANALYSIS_SHA, remote], cwd=repo, check=False
            ).returncode != 0:
                raise RuntimeError(f"evidence branch not descended from frozen analysis source: {publication_branch}")
            path = (
                "journal_extension/evidence/public/track_a/posttraining/"
                f"{run_id}/POSTTRAINING_STATE_COMPLETION.json"
            )
            completion = json.loads(show_bytes(repo, remote, path))
            checks = {
                "status": completion.get("status") == "PASS",
                "experiment_id": completion.get("experiment_id") == experiment_id,
                "analysis_source": completion.get("analysis_source_git_commit") == ANALYSIS_SHA,
                "private_roundtrip": completion.get("private_generation_roundtrip_verified") is True,
                "publication_branch": completion.get("publication_branch") == publication_branch,
                "v1_closed": completion.get("v1_test_accessed") is False,
                "external_closed": completion.get("external_surface_accessed") is False,
                "training_closed": completion.get("training_or_adaptation_performed") is False,
            }
            if not all(checks.values()):
                raise RuntimeError(f"completion contract failed for {experiment_id}: {checks}")
            states[experiment_id] = {"status": "PASS", "completion": completion}

        payload = {
            "schema_version": "1.0",
            "status": "PASS",
            "manifest_kind": "track_a_posttraining_account_execution_reconstructed_for_closure",
            "account_id": account,
            "analysis_source_git_commit": ANALYSIS_SHA,
            "assigned_state_count": len(states),
            "completed_state_count": len(states),
            "states": states,
            "v1_test_accessed": False,
            "external_surface_accessed": False,
            "reconstructed_for_closure": True,
            "reconstruction_basis": (
                "immutable account-readiness partition plus each state's published "
                "POSTTRAINING_STATE_COMPLETION.json"
            ),
            "source_account_readiness_sha256": sha256_file(readiness_path),
        }
        payload["manifest_sha256"] = canonical_hash(payload)
        write_json(inputs / f"{account}_completion_reconstructed.json", payload)

    if union != set(global_gate["states"]):
        raise RuntimeError("reconstructed K1/K2/K3 union differs from sealed 21-state global readiness inventory")


def prepare_science_worktree(repo: Path, science: Path) -> None:
    if science.exists():
        shutil.rmtree(science)
    run(["git", "worktree", "prune"], repo)
    run(["git", "worktree", "add", "--detach", str(science), ANALYSIS_SHA], repo)
    observed = run(["git", "rev-parse", "HEAD"], science, capture=True).strip()
    if observed != ANALYSIS_SHA:
        raise RuntimeError(f"science worktree drift: {observed}")
    for name in ("audit_tracka_v12_posttraining_evidence.py", "close_tracka_v12.py"):
        shutil.copy2(repo / "journal_extension/scripts" / name, science / "journal_extension/scripts" / name)

    frozen = [
        "journal_extension/scripts/seal_tracka_v12_selection.py",
        "journal_extension/scripts/seal_tracka_v12_auxiliary_analysis.py",
        "journal_extension/scripts/seal_tracka_v12_comprehensive_closure.py",
        "journal_extension/src/cropcop_je/tracka_v12_analysis.py",
        "journal_extension/src/cropcop_je/tracka_v12_evidence.py",
        "journal_extension/src/cropcop_je/tracka_v12_xai.py",
    ]
    for path in frozen:
        base_blob = run(["git", "rev-parse", f"{ANALYSIS_SHA}:{path}"], repo, capture=True).strip()
        work_blob = run(["git", "rev-parse", f"HEAD:{path}"], science, capture=True).strip()
        if base_blob != work_blob:
            raise RuntimeError(f"frozen closure science blob mismatch: {path}")


def run_closure(science: Path, inputs: Path, global_audit: Path, final_out: Path) -> None:
    for path in (global_audit, final_out):
        if path.exists():
            shutil.rmtree(path)

    cmd = [
        "python",
        str(science / "journal_extension/scripts/audit_tracka_v12_posttraining_evidence.py"),
        "--repo-root", str(science),
        "--analysis-source-git-commit", ANALYSIS_SHA,
        "--global-readiness", str(inputs / "global_readiness.json"),
        "--account-completion", str(inputs / "K1_completion_reconstructed.json"),
        "--account-completion", str(inputs / "K2_completion_reconstructed.json"),
        "--account-completion", str(inputs / "K3_completion_reconstructed.json"),
        "--output-dir", str(global_audit),
    ]
    run(cmd, science)

    cmd = [
        "python",
        str(science / "journal_extension/scripts/close_tracka_v12.py"),
        "--repo-root", str(science),
        "--analysis-source-git-commit", ANALYSIS_SHA,
        "--global-evidence-audit", str(global_audit / "TRACKA_POSTTRAINING_GLOBAL_EVIDENCE_AUDIT.json"),
        "--direct-evidence-index", str(global_audit / "TRACKA_DIRECT_EVIDENCE_INDEX.json"),
        "--auxiliary-evidence-index", str(global_audit / "TRACKA_AUXILIARY_EVIDENCE_INDEX.json"),
        "--output-dir", str(final_out),
    ]
    run(cmd, science)


def build_integration_artifacts(
    repo: Path, inputs: Path, global_audit: Path, final_out: Path, integration: Path
) -> Path:
    remote = fetch_branch(repo, INTEGRATION_BRANCH)
    if integration.exists():
        shutil.rmtree(integration)
    run(["git", "worktree", "add", "--detach", str(integration), remote], repo)

    dest = integration / FINAL_REL
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    input_dest = dest / "closure_inputs"
    input_dest.mkdir()

    for p in global_audit.glob("*.json"):
        shutil.copy2(p, dest / p.name)
    for p in final_out.glob("*.json"):
        shutil.copy2(p, dest / p.name)
    for account in ("K1", "K2", "K3"):
        shutil.copy2(inputs / f"{account}_completion_reconstructed.json", input_dest)

    audit = load_json(dest / "TRACKA_POSTTRAINING_GLOBAL_EVIDENCE_AUDIT.json")
    selection = load_json(dest / "TRACKA_DIRECT_SELECTION.json")
    auxiliary = load_json(dest / "TRACKA_AUXILIARY_ANALYSIS.json")
    comprehensive = load_json(dest / "TRACKA_COMPREHENSIVE_CLOSURE.json")
    final = load_json(dest / "TRACKA_FINAL_CLOSURE_AUDIT.json")

    if final.get("status") != "PASS" or final.get("track_a_closed") is not True:
        raise RuntimeError("final closure audit is not PASS")
    if final.get("scientific_state_count") != 21 or final.get("private_evidence_roundtrip_verified_state_count") != 21:
        raise RuntimeError("final closure state counts invalid")
    if final.get("v1_test_accessed") is not False:
        raise RuntimeError("V1 test was accessed during Track-A closure")
    if comprehensive.get("status") != "PASS" or comprehensive.get("track_a_comprehensive_closed") is not True:
        raise RuntimeError("comprehensive Track-A closure is not PASS")
    if comprehensive.get("selection", {}).get("status") != "SELECTED":
        raise RuntimeError("Track-A did not resolve to a single selected scientific primary")
    if comprehensive.get("track_b_handoff_authorized") is not True:
        raise RuntimeError("Track-B handoff not authorized")
    if comprehensive.get("track_c_handoff_authorized") is not True:
        raise RuntimeError("Track-C handoff not authorized")

    authority = {
        "schema_version": "1.0",
        "status": "LOCKED",
        "artifact_kind": "track_a_final_authority_chain",
        "scientific_authority_id": "EAAI-JE-SDL-v2.1-QA",
        "posttraining_closure_authority_id": "EAAI-JE-TRACKA-POSTTRAINING-CLOSURE-V1",
        "campaign_id": "CROPCOP-TRACKA-POSTTRAINING-CLOSURE-V1",
        "analysis_source_git_commit": ANALYSIS_SHA,
        "training_science_source_git_commit": TRAINING_SHA,
        "historical_training_source_git_commit": HISTORICAL_SHA,
        "scientific_state_count": 21,
        "direct_state_count": 12,
        "auxiliary_state_count": 9,
        "global_evidence_audit_sha256": sha256_file(dest / "TRACKA_POSTTRAINING_GLOBAL_EVIDENCE_AUDIT.json"),
        "direct_selection_sha256": sha256_file(dest / "TRACKA_DIRECT_SELECTION.json"),
        "auxiliary_analysis_sha256": sha256_file(dest / "TRACKA_AUXILIARY_ANALYSIS.json"),
        "comprehensive_closure_sha256": sha256_file(dest / "TRACKA_COMPREHENSIVE_CLOSURE.json"),
        "final_closure_audit_sha256": sha256_file(dest / "TRACKA_FINAL_CLOSURE_AUDIT.json"),
        "account_completion_reconstruction": {
            account: sha256_file(input_dest / f"{account}_completion_reconstructed.json")
            for account in ("K1", "K2", "K3")
        },
        "account_completion_reconstruction_note": (
            "The original account-level aggregation files were not persisted to Git. "
            "For closure only, they are deterministically reconstructed from immutable "
            "account-readiness partitions and each state's already-published completion certificate. "
            "No scientific metric, checkpoint, role, or outcome is invented or changed."
        ),
        "historical_locks_modified": False,
        "training_reperformed": False,
        "selector_modified": False,
        "xai_weighted_selection": False,
        "protected_surfaces": {
            "v1_test": "CLOSED",
            "track_b_predictions": "CLOSED_DURING_SELECTION",
            "track_c_candidate_results": "CLOSED_DURING_SELECTION",
        },
        "downstream_handoff_authorized": True,
    }
    authority["authority_chain_sha256"] = canonical_hash(authority)
    write_json(dest / "TRACKA_FINAL_AUTHORITY_CHAIN.json", authority)

    pair = auxiliary["paired_analyses"]["R05_teacher_minus_R04_direct"]["metric_deltas"]
    r12 = auxiliary["paired_analyses"]["R12_logits_minus_feature"]["metric_deltas"]
    primary = comprehensive["selection"]["journal_primary_family"]
    claims = {
        "schema_version": "1.0",
        "status": "LOCKED",
        "artifact_kind": "track_a_claim_matrix",
        "authorized": [
            {
                "claim_id": "TA-C01",
                "claim": "Track A contains 21 terminal scientific states: 12 direct and 9 auxiliary.",
                "basis": "TRACKA_FINAL_CLOSURE_AUDIT.json",
            },
            {
                "claim_id": "TA-C02",
                "claim": (
                    f"The frozen selector selected {primary} as the Track-A scientific-primary family "
                    "under the common CropCop downstream protocol."
                ),
                "basis": "TRACKA_DIRECT_SELECTION.json",
            },
            {
                "claim_id": "TA-C03",
                "claim": (
                    "The matched three-seed teacher-guided MobileNetV4 condition did not show "
                    "a consistent aggregate macro-F1 advantage over direct MobileNetV4 training."
                ),
                "basis": "TRACKA_AUXILIARY_ANALYSIS.json",
                "mean_macro_f1_teacher_minus_direct": pair["validation_macro_f1"]["mean"],
            },
            {
                "claim_id": "TA-C04",
                "claim": (
                    "R12 logits and feature-transfer conditions are descriptive mechanism ablations; "
                    "differences are reported without post-hoc hypothesis testing."
                ),
                "basis": "TRACKA_AUXILIARY_ANALYSIS.json",
                "mean_macro_f1_logits_minus_feature": r12["validation_macro_f1"]["mean"],
            },
            {
                "claim_id": "TA-C05",
                "claim": (
                    "Direct candidate selection used clean validation, tail-class, corruption-robustness "
                    "and model-state efficiency evidence; XAI was audited but not weighted in selection."
                ),
                "basis": "TRACKA_DIRECT_SELECTION.json",
            },
            {
                "claim_id": "TA-C06",
                "claim": (
                    "All 21 published post-training states preserve private-evidence round-trip "
                    "verification and protected-surface closure."
                ),
                "basis": "TRACKA_POSTTRAINING_GLOBAL_EVIDENCE_AUDIT.json",
            },
        ],
        "forbidden": [
            {"claim_id": "TA-F01", "claim": "causal architecture-only superiority"},
            {"claim_id": "TA-F02", "claim": "teacher guidance materially or consistently improves CropCop"},
            {"claim_id": "TA-F03", "claim": "field, geographic, source-independent or smartphone generalization from Track A"},
            {"claim_id": "TA-F04", "claim": "physical-device latency, memory, thermal or energy performance from Track A"},
            {"claim_id": "TA-F05", "claim": "new V1-test-based selection or tuning"},
            {"claim_id": "TA-F06", "claim": "post-hoc p-values or multiple-comparison significance for R05/R12 auxiliary analysis"},
            {"claim_id": "TA-F07", "claim": "XAI maps constitute causal or biological localization validation"},
        ],
        "track_b_handoff_authorized": True,
        "track_c_handoff_authorized": True,
    }
    claims["claim_matrix_sha256"] = canonical_hash(claims)
    write_json(dest / "TRACKA_CLAIM_MATRIX.json", claims)

    rows = selection["selector_rows"]
    cond = auxiliary["conditions"]
    table = "\n".join(
        (
            f"| {row['family']} | {row['mean_validation_macro_f1']:.6f} | "
            f"{row['worst_seed_validation_macro_f1']:.6f} | "
            f"{row['bottom_12_class_mean_f1']:.6f} | "
            f"{row['mean_corruption_degradation_pp']:.3f} | "
            f"{row['model_state_tensor_bytes_fp32']:,} | {row['total_parameter_count']:,} |"
        )
        for row in rows
    )
    report = f"""# CropCop Track-A Final Closure Report

**Status:** LOCKED / PASS  
**Frozen analysis source:** `{ANALYSIS_SHA}`  
**Scientific states:** 21/21  
**Direct states:** 12/12  
**Auxiliary states:** 9/9  
**Private evidence round-trips:** 21/21  
**V1 test access during closure:** 0  
**External prediction access during selection:** 0  
**Track-C candidate-result access during selection:** 0  
**New training or optimizer advance during post-training closure:** 0  

## 1. Terminal decision

**GO - TRACK A CLOSED.**  
**GO - sealed Track-B and Track-C handoff authorized.**  
No further Track-A model-science compute is authorized under the current authority.

The frozen deterministic selector returned **{comprehensive['selection']['status']}** and selected **{primary}** as the scientific-primary family. This is the best frozen pretrained candidate system under the common CropCop downstream protocol; it is not a causal architecture-superiority claim.

## 2. Evidence integrity

The repository's own post-training global audit consumed all 21 immutable published evidence branches and verified exact state inventory, publication manifests, completion chains, direct/auxiliary evidence contracts, private durability round-trips, source lineage, and protected-surface markers before scientific selection.

The original account-level aggregation files were not persisted to Git. For closure only, K1/K2/K3 aggregation manifests were deterministically reconstructed from the sealed account-readiness partitions plus each state's immutable published `POSTTRAINING_STATE_COMPLETION.json`. The reconstructed inputs contain no new scientific measurement and are retained under `closure_inputs/`.

## 3. Frozen direct-candidate selection

| Family | Mean macro-F1 | Worst-seed macro-F1 | Bottom-12 class mean F1 | Mean corruption degradation (pp) | FP32 state bytes | Parameters |
|---|---:|---:|---:|---:|---:|---:|
{table}

The frozen Pareto/lexicographic selector selected **{primary}**. XAI remained an audit surface and was not used as a weighted selector.

## 4. Matched teacher/direct result

Three-seed validation macro-F1:
- R04 direct: **{cond['R04_direct']['three_seed']['validation_macro_f1']['mean']:.6f} +/- {cond['R04_direct']['three_seed']['validation_macro_f1']['sample_sd']:.6f}**
- R05 teacher: **{cond['R05_teacher']['three_seed']['validation_macro_f1']['mean']:.6f} +/- {cond['R05_teacher']['three_seed']['validation_macro_f1']['sample_sd']:.6f}**
- Teacher minus direct mean delta: **{pair['validation_macro_f1']['mean']:.6f}**

The journal extension must not claim that teacher guidance consistently improves aggregate performance.

## 5. R12 mechanism ablation

Three-seed validation macro-F1:
- logits: **{cond['R12_logits']['three_seed']['validation_macro_f1']['mean']:.6f} +/- {cond['R12_logits']['three_seed']['validation_macro_f1']['sample_sd']:.6f}**
- feature: **{cond['R12_feature']['three_seed']['validation_macro_f1']['mean']:.6f} +/- {cond['R12_feature']['three_seed']['validation_macro_f1']['sample_sd']:.6f}**
- logits minus feature mean delta: **{r12['validation_macro_f1']['mean']:.6f}**

These are descriptive mechanism results. The frozen authority does not authorize post-hoc hypothesis tests or multiple-comparison p-values.

## 6. Journal-extension review alignment achieved by Track A

Track A now supplies the model-science evidence requested by the review in the areas assigned to this track: matched direct MobileNetV4 control, three-seed major comparisons, compact mechanism ablations, representative same-split direct candidates, exact selected-checkpoint replay, long-tail/classwise evidence, corruption robustness, model-state efficiency, and bounded XAI sanity checks.

Track A intentionally does not close independent external validity or physical-device deployment. Those are protected downstream tracks.

## 7. Permanent claim boundary

The authoritative allowed/forbidden claim set is `TRACKA_CLAIM_MATRIX.json`. Track A does not authorize field-generalization claims, physical-device claims, causal architecture-superiority wording, unsupported teacher-benefit wording, new V1-test selection, post-hoc auxiliary p-values, or causal localization claims from XAI.

## 8. Next action

Proceed to **Track B (external validity)** and **Track C (runtime/device)** using only the sealed Track-A scientific primary authorized by `TRACKA_COMPREHENSIVE_CLOSURE.json`.

The global manuscript evidence freeze remains downstream of Track B/C terminal evidence.
"""
    (dest / "TRACKA_FINAL_CLOSURE_REPORT.md").write_text(report, encoding="utf-8")

    manifest = {
        "schema_version": "1.0",
        "status": "PASS",
        "artifact_kind": "track_a_final_closure_manifest",
        "files": {p.name: sha256_file(p) for p in sorted(dest.glob("*")) if p.is_file()},
        "closure_input_files": {
            p.name: sha256_file(p) for p in sorted(input_dest.glob("*.json"))
        },
        "track_a_closed": True,
        "track_b_handoff_authorized": True,
        "track_c_handoff_authorized": True,
    }
    manifest["manifest_sha256"] = canonical_hash(manifest)
    write_json(dest / "TRACKA_FINAL_CLOSURE_MANIFEST.json", manifest)

    return dest


def final_consistency(dest: Path) -> None:
    final = load_json(dest / "TRACKA_FINAL_CLOSURE_AUDIT.json")
    comp = load_json(dest / "TRACKA_COMPREHENSIVE_CLOSURE.json")
    claims = load_json(dest / "TRACKA_CLAIM_MATRIX.json")
    manifest = load_json(dest / "TRACKA_FINAL_CLOSURE_MANIFEST.json")
    audit = load_json(dest / "TRACKA_POSTTRAINING_GLOBAL_EVIDENCE_AUDIT.json")

    assert final["status"] == "PASS" and final["track_a_closed"] is True
    assert final["scientific_state_count"] == 21
    assert final["direct_candidate_state_count"] == 12
    assert final["auxiliary_state_count"] == 9
    assert final["private_evidence_roundtrip_verified_state_count"] == 21
    assert final["v1_test_accessed"] is False
    assert final["external_predictions_opened_before_selection"] is False
    assert final["track_c_candidate_results_opened_before_selection"] is False
    assert final["training_reperformed"] is False
    assert final["optimizer_state_advanced"] is False
    assert audit["status"] == "PASS" and audit["scientific_state_count"] == 21
    assert comp["status"] == "PASS" and comp["track_a_comprehensive_closed"] is True
    assert comp["selection"]["status"] == "SELECTED"
    assert comp["track_b_handoff_authorized"] is True
    assert comp["track_c_handoff_authorized"] is True
    assert claims["track_b_handoff_authorized"] is True
    assert claims["track_c_handoff_authorized"] is True
    assert manifest["status"] == "PASS" and manifest["track_a_closed"] is True


def commit_integration(repo: Path, integration: Path) -> None:
    run(["git", "config", "user.name", "github-actions[bot]"], integration)
    run(
        [
            "git",
            "config",
            "user.email",
            "41898282+github-actions[bot]@users.noreply.github.com",
        ],
        integration,
    )
    run(["git", "add", str(FINAL_REL)], integration)
    status = run(["git", "status", "--porcelain"], integration, capture=True)
    if not status.strip():
        print("Track-A closure artifacts already current.")
        return
    run(["git", "commit", "-m", "evidence: seal final Track-A closure [skip ci]"], integration)
    run(["git", "push", "origin", f"HEAD:refs/heads/{INTEGRATION_BRANCH}"], integration)


def main() -> int:
    repo = Path.cwd().resolve()
    observed = run(["git", "rev-parse", "--abbrev-ref", "HEAD"], repo, capture=True).strip()
    print(f"orchestrator checkout branch={observed}")

    temp = Path("/tmp/cropcop_tracka_final_closure")
    if temp.exists():
        shutil.rmtree(temp)
    temp.mkdir()
    inputs = temp / "inputs"
    science = temp / "science"
    global_audit = temp / "global_audit"
    final_out = temp / "final"
    integration = temp / "integration"

    materialize_readiness(repo, inputs)
    reconstruct_account_completions(repo, inputs)
    prepare_science_worktree(repo, science)
    run_closure(science, inputs, global_audit, final_out)
    dest = build_integration_artifacts(repo, inputs, global_audit, final_out, integration)
    final_consistency(dest)
    commit_integration(repo, integration)

    comp = load_json(dest / "TRACKA_COMPREHENSIVE_CLOSURE.json")
    final = load_json(dest / "TRACKA_FINAL_CLOSURE_AUDIT.json")
    print(
        json.dumps(
            {
                "status": "PASS",
                "track_a_closed": final["track_a_closed"],
                "scientific_state_count": final["scientific_state_count"],
                "journal_primary_family": comp["selection"]["journal_primary_family"],
                "selection_status": comp["selection"]["status"],
                "track_b_handoff_authorized": comp["track_b_handoff_authorized"],
                "track_c_handoff_authorized": comp["track_c_handoff_authorized"],
                "v1_test_accessed": final["v1_test_accessed"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
