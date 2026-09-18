from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import _bootstrap  # noqa: F401
from cropcop_je.atomic_io import atomic_write_json
from cropcop_je.hashing import sha256_file, sha256_json
from cropcop_je.tracka_v12_evidence import ALL_DIRECT_STATES, validate_direct_state_evidence_bundle

ACCOUNTS = {"K1", "K2", "K3"}
SEEDS = ("S1", "S2", "S3")
R04 = {f"R04-MNV4-DIRECT-{seed}" for seed in SEEDS}
R05 = {f"R05-MNV4-TEACHER-{seed}" for seed in SEEDS}
R12 = {f"R12-MNV4-{mode}-{seed}" for mode in ("LOGITS", "FEATURE") for seed in SEEDS}
AUX_INDEX_STATES = R04 | R05 | R12
FULL_TRACK_A = set(ALL_DIRECT_STATES) | R05 | R12
PREFIX = "journal_extension/evidence/public/track_a/posttraining"


def load_json(path: str | Path) -> dict:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"JSON object required: {path}")
    return payload


def run_git(repo: Path, args: list[str]) -> str:
    cp = subprocess.run(["git", "-C", str(repo), *args], check=False, capture_output=True, text=True, timeout=300)
    if cp.returncode != 0:
        raise RuntimeError((cp.stderr or cp.stdout or "").strip()[-1500:])
    return cp.stdout


def extract_branch(repo: Path, branch: str, run_id: str, destination: Path, analysis_sha: str) -> dict[str, Path]:
    remote_ref = f"refs/remotes/origin/{branch}"
    run_git(repo, ["fetch", "origin", f"refs/heads/{branch}:{remote_ref}"])
    ancestor = subprocess.run(
        ["git", "-C", str(repo), "merge-base", "--is-ancestor", analysis_sha, remote_ref],
        check=False, capture_output=True, text=True,
    )
    if ancestor.returncode != 0:
        raise RuntimeError(f"public evidence branch is not descended from analysis source: {branch}")
    prefix = f"{PREFIX}/{run_id}"
    names = [
        row.strip()
        for row in run_git(repo, ["ls-tree", "-r", "--name-only", remote_ref, "--", prefix]).splitlines()
        if row.strip()
    ]
    if not names:
        raise RuntimeError(f"public evidence branch contains no state evidence: {branch}")
    destination.mkdir(parents=True, exist_ok=False)
    outputs = {}
    for name in names:
        basename = Path(name).name
        if basename in outputs:
            raise RuntimeError(f"duplicate basename in public evidence branch: {branch}:{basename}")
        data = subprocess.check_output(["git", "-C", str(repo), "show", f"{remote_ref}:{name}"])
        path = destination / basename
        path.write_bytes(data)
        outputs[basename] = path
    return outputs


def verify_publication_manifest(files: dict[str, Path], experiment_id: str, run_id: str, analysis_sha: str) -> dict:
    path = files.get("POSTTRAINING_PUBLICATION_MANIFEST.json")
    if path is None:
        raise RuntimeError(f"publication manifest missing: {experiment_id}")
    manifest = load_json(path)
    expected = manifest.get("publication_manifest_sha256")
    clean = dict(manifest)
    clean.pop("publication_manifest_sha256", None)
    if expected != sha256_json(clean):
        raise RuntimeError(f"publication manifest self-hash mismatch: {experiment_id}")
    if manifest.get("status") != "PASS" or manifest.get("experiment_id") != experiment_id:
        raise RuntimeError(f"publication manifest identity/status mismatch: {experiment_id}")
    if manifest.get("run_id") != run_id or manifest.get("analysis_source_git_commit") != analysis_sha:
        raise RuntimeError(f"publication manifest provenance mismatch: {experiment_id}")
    if manifest.get("private_material_published") is not False:
        raise RuntimeError(f"publication manifest indicates private material publication: {experiment_id}")
    hashes = manifest.get("public_file_sha256") or {}
    for basename, expected_sha in hashes.items():
        path = files.get(basename)
        if path is None or sha256_file(path) != expected_sha:
            raise RuntimeError(f"published public-file hash mismatch: {experiment_id}:{basename}")
    return manifest


def validate_published_completion_chain(
    *,
    files: dict[str, Path],
    account_completion: dict,
    experiment_id: str,
    analysis_sha: str,
) -> dict:
    published_completion_path = files.get("POSTTRAINING_STATE_COMPLETION.json")
    published_sync_path = files.get("POSTTRAINING_PRIVATE_SYNC_CERTIFICATE.json")
    if published_completion_path is None or published_sync_path is None:
        raise RuntimeError(f"published completion/durability certificate missing: {experiment_id}")

    published_completion = load_json(published_completion_path)
    if published_completion != account_completion:
        raise RuntimeError(f"published state completion differs from account completion manifest: {experiment_id}")
    completion_hash = published_completion.get("completion_sha256")
    completion_clean = dict(published_completion)
    completion_clean.pop("completion_sha256", None)
    if completion_hash != sha256_json(completion_clean):
        raise RuntimeError(f"published state completion self-hash mismatch: {experiment_id}")

    sync = load_json(published_sync_path)
    sync_hash = sync.get("sync_certificate_sha256")
    sync_clean = dict(sync)
    sync_clean.pop("sync_certificate_sha256", None)
    if sync.get("status") != "PASS":
        raise RuntimeError(f"published private durability certificate is not PASS: {experiment_id}")
    if sync.get("experiment_id") != experiment_id or sync.get("analysis_source_git_commit") != analysis_sha:
        raise RuntimeError(f"published private durability certificate provenance mismatch: {experiment_id}")
    if sync.get("generation_roundtrip_verified") is not True or sync.get("private_dataset_verified") is not True:
        raise RuntimeError(f"published private durability certificate lacks round-trip/private proof: {experiment_id}")
    if sync.get("dataset_locator") != published_completion.get("private_evidence_dataset_locator"):
        raise RuntimeError(f"published private durability dataset locator mismatch: {experiment_id}")
    if sync_hash != sha256_json(sync_clean):
        raise RuntimeError(f"published private durability certificate self-hash mismatch: {experiment_id}")
    if sha256_file(published_sync_path) != published_completion.get("private_sync_certificate_sha256"):
        raise RuntimeError(f"published private durability certificate file-hash mismatch: {experiment_id}")

    return {
        "state_completion_sha256": sha256_file(published_completion_path),
        "private_sync_certificate_sha256": sha256_file(published_sync_path),
        "private_generation_roundtrip_verified": True,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", default=".")
    ap.add_argument("--analysis-source-git-commit", required=True)
    ap.add_argument("--global-readiness", required=True)
    ap.add_argument("--account-completion", action="append", required=True)
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()

    repo = Path(args.repo_root).resolve()
    analysis_sha = args.analysis_source_git_commit
    observed = run_git(repo, ["rev-parse", "HEAD"]).strip()
    if observed != analysis_sha:
        raise SystemExit(f"exact analysis checkout mismatch: expected={analysis_sha}, observed={observed}")
    readiness_path = Path(args.global_readiness).resolve()
    readiness = load_json(readiness_path)
    if (
        readiness.get("status") != "PASS"
        or readiness.get("gate_kind") != "track_a_v12_posttraining_global_readiness"
        or readiness.get("analysis_source_git_commit") != analysis_sha
        or readiness.get("ready_for_posttraining_evidence") is not True
    ):
        raise SystemExit("global post-training readiness is not a matching PASS")

    account_manifests = {}
    account_hashes = {}
    completions = {}
    for raw in args.account_completion:
        path = Path(raw).resolve()
        payload = load_json(path)
        account = str(payload.get("account_id", ""))
        if account not in ACCOUNTS or account in account_manifests:
            raise SystemExit(f"exact unique K1/K2/K3 completion manifests required; got {account!r}")
        if payload.get("status") != "PASS" or payload.get("analysis_source_git_commit") != analysis_sha:
            raise SystemExit(f"account execution is not terminal PASS: {account}")
        if payload.get("v1_test_accessed") is not False or payload.get("external_surface_accessed") is not False:
            raise SystemExit(f"account execution protected-surface marker invalid: {account}")
        account_manifests[account] = payload
        account_hashes[account] = sha256_file(path)
        for experiment_id, row in (payload.get("states") or {}).items():
            if experiment_id in completions:
                raise SystemExit(f"duplicate state across account completion manifests: {experiment_id}")
            if row.get("status") not in {"PASS", "REUSED_COMPLETE"}:
                raise SystemExit(f"state is not terminal complete: {experiment_id}")
            completion = row.get("completion") or {}
            if completion.get("status") != "PASS" or completion.get("experiment_id") != experiment_id:
                raise SystemExit(f"state completion payload invalid: {experiment_id}")
            if completion.get("analysis_source_git_commit") != analysis_sha:
                raise SystemExit(f"state completion analysis-source mismatch: {experiment_id}")
            if completion.get("private_generation_roundtrip_verified") is not True:
                raise SystemExit(f"private evidence roundtrip missing: {experiment_id}")
            if not completion.get("publication_branch"):
                raise SystemExit(f"public evidence branch missing: {experiment_id}")
            completions[experiment_id] = completion
    if set(account_manifests) != ACCOUNTS:
        raise SystemExit(f"exact K1/K2/K3 completion manifests required, got {sorted(account_manifests)}")
    if set(completions) != FULL_TRACK_A:
        missing = sorted(FULL_TRACK_A - set(completions))
        extra = sorted(set(completions) - FULL_TRACK_A)
        raise SystemExit(f"completion union must be exact 21-state Track-A inventory; missing={missing}, extra={extra}")

    output = Path(args.output_dir).resolve()
    if output.exists():
        raise SystemExit("global evidence audit output directory must not already exist")
    output.mkdir(parents=True, exist_ok=False)
    state_root = output / "states"
    state_root.mkdir()

    collected = {}
    direct_index = {"schema_version": "1.0", "states": {}}
    auxiliary_index = {"schema_version": "1.0", "states": {}, "r04_robustness_replay": {}}
    for experiment_id in sorted(FULL_TRACK_A):
        completion = completions[experiment_id]
        run_id = completion["posttraining_public_run_id"]
        files = extract_branch(
            repo,
            completion["publication_branch"],
            run_id,
            state_root / experiment_id,
            analysis_sha,
        )
        publication = verify_publication_manifest(files, experiment_id, run_id, analysis_sha)
        try:
            completion_chain = validate_published_completion_chain(
                files=files,
                account_completion=completion,
                experiment_id=experiment_id,
                analysis_sha=analysis_sha,
            )
        except RuntimeError as exc:
            raise SystemExit(str(exc)) from exc
        role = completion["role"]
        if role == "direct":
            required = {
                "DIRECT_STATE_EVIDENCE_GATE.json",
                "robustness_and_replay.json",
                "efficiency.json",
                "XAI_EVIDENCE_GATE.json",
            }
            if not required.issubset(files):
                raise SystemExit(f"direct public evidence incomplete: {experiment_id}")
            direct = load_json(files["DIRECT_STATE_EVIDENCE_GATE.json"])
            robust = load_json(files["robustness_and_replay.json"])
            efficiency = load_json(files["efficiency.json"])
            xai = load_json(files["XAI_EVIDENCE_GATE.json"])
            errors = validate_direct_state_evidence_bundle(
                experiment_id,
                direct=direct,
                robustness=robust,
                efficiency=efficiency,
                xai=xai,
            )
            if errors:
                raise SystemExit(f"independent direct evidence audit failed for {experiment_id}: " + "; ".join(errors))
            direct_index["states"][experiment_id] = {
                "direct_gate": str(files["DIRECT_STATE_EVIDENCE_GATE.json"]),
                "robustness_replay": str(files["robustness_and_replay.json"]),
                "efficiency": str(files["efficiency.json"]),
                "xai_gate": str(files["XAI_EVIDENCE_GATE.json"]),
            }
            if experiment_id in R04:
                auxiliary_index["states"][experiment_id] = str(files["DIRECT_STATE_EVIDENCE_GATE.json"])
                auxiliary_index["r04_robustness_replay"][experiment_id] = str(files["robustness_and_replay.json"])
        elif role == "auxiliary":
            required = {"AUXILIARY_STATE_EVIDENCE_GATE.json"}
            if not required.issubset(files):
                raise SystemExit(f"auxiliary public evidence incomplete: {experiment_id}")
            gate = load_json(files["AUXILIARY_STATE_EVIDENCE_GATE.json"])
            if (
                gate.get("status") != "PASS"
                or gate.get("experiment_id") != experiment_id
                or gate.get("analysis_source_git_commit") != analysis_sha
                or gate.get("replay_gate", {}).get("status") != "PASS"
                or gate.get("classwise_pass") is not True
                or gate.get("training_performed") is not False
                or gate.get("optimizer_state_advanced") is not False
                or gate.get("v1_test_accessed") is not False
                or gate.get("external_surface_accessed") is not False
            ):
                raise SystemExit(f"independent auxiliary evidence audit failed: {experiment_id}")
            auxiliary_index["states"][experiment_id] = str(files["AUXILIARY_STATE_EVIDENCE_GATE.json"])
        else:
            raise SystemExit(f"unsupported Track-A completion role: {experiment_id}:{role}")
        collected[experiment_id] = {
            "role": role,
            "publication_branch": completion["publication_branch"],
            "publication_manifest_sha256": sha256_file(files["POSTTRAINING_PUBLICATION_MANIFEST.json"]),
            **completion_chain,
            "public_file_sha256": publication["public_file_sha256"],
        }

    if set(direct_index["states"]) != set(ALL_DIRECT_STATES):
        raise SystemExit("direct evidence index is not exact 12-state inventory")
    if set(auxiliary_index["states"]) != AUX_INDEX_STATES or set(auxiliary_index["r04_robustness_replay"]) != R04:
        raise SystemExit("auxiliary analysis index inventory mismatch")

    direct_index_path = output / "TRACKA_DIRECT_EVIDENCE_INDEX.json"
    auxiliary_index_path = output / "TRACKA_AUXILIARY_EVIDENCE_INDEX.json"
    atomic_write_json(direct_index_path, direct_index)
    atomic_write_json(auxiliary_index_path, auxiliary_index)
    audit = {
        "schema_version": "1.0",
        "status": "PASS",
        "audit_kind": "track_a_posttraining_global_evidence",
        "analysis_source_git_commit": analysis_sha,
        "global_readiness_sha256": sha256_file(readiness_path),
        "account_completion_sha256": account_hashes,
        "scientific_state_count": len(collected),
        "direct_state_count": len(direct_index["states"]),
        "auxiliary_only_state_count": len(FULL_TRACK_A - set(ALL_DIRECT_STATES)),
        "private_roundtrip_verified_state_count": sum(row["private_generation_roundtrip_verified"] for row in collected.values()),
        "states": collected,
        "direct_evidence_index_sha256": sha256_file(direct_index_path),
        "auxiliary_evidence_index_sha256": sha256_file(auxiliary_index_path),
        "selector_authorized": True,
        "auxiliary_analysis_authorized": True,
        "v1_test_accessed": False,
        "external_predictions_opened": False,
        "track_c_candidate_results_opened": False,
    }
    audit["audit_sha256"] = sha256_json(audit)
    audit_path = output / "TRACKA_POSTTRAINING_GLOBAL_EVIDENCE_AUDIT.json"
    atomic_write_json(audit_path, audit)
    print(json.dumps(audit, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
