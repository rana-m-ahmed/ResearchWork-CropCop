from __future__ import annotations

import importlib.metadata
import inspect
import json
from pathlib import Path

from kaggle.api.kaggle_api_extended import KaggleApi

EXPECTED_VERSION = "2.2.4"


def _params(fn) -> list[str]:
    return list(inspect.signature(fn).parameters)


def main() -> int:
    version = importlib.metadata.version("kaggle")
    if version != EXPECTED_VERSION:
        raise SystemExit(
            f"kaggle drift: expected {EXPECTED_VERSION}, got {version}"
        )

    expected = {
        "dataset_status": ["self", "dataset", "format"],
        "dataset_list_files": [
            "self",
            "dataset",
            "page_token",
            "page_size",
        ],
        "dataset_download_file": [
            "self",
            "dataset",
            "file_name",
            "path",
            "force",
            "quiet",
            "licenses",
        ],
    }
    observed = {
        name: _params(getattr(KaggleApi, name))
        for name in expected
    }
    for name, prefix in expected.items():
        if observed[name][: len(prefix)] != prefix:
            raise SystemExit(
                f"Kaggle {name} signature drift: {observed[name]}"
            )

    create_params = _params(KaggleApi.dataset_create_version)
    if create_params[:3] != ["self", "folder", "version_notes"]:
        raise SystemExit(
            "Kaggle dataset_create_version signature drift: "
            f"{create_params}"
        )

    api = object.__new__(KaggleApi)
    owner, slug, version_number = KaggleApi.split_dataset_string(
        api,
        "owner/example-dataset/7",
    )
    if (owner, slug, version_number) != (
        "owner",
        "example-dataset",
        "7",
    ):
        raise SystemExit(
            "Kaggle exact-version dataset ref parsing changed"
        )

    sources = {
        "dataset_status": inspect.getsource(
            KaggleApi.dataset_status
        ),
        "dataset_list_files": inspect.getsource(
            KaggleApi.dataset_list_files
        ),
        "dataset_download_file": inspect.getsource(
            KaggleApi.dataset_download_file
        ),
    }
    required_tokens = {
        "dataset_status": ("current_version_number",),
        "dataset_list_files": ("dataset_version_number",),
        "dataset_download_file": (
            "dataset_version_number",
            "file_name",
        ),
    }
    for name, tokens in required_tokens.items():
        for token in tokens:
            if token not in sources[name]:
                raise SystemExit(
                    f"Kaggle {name} no longer exposes "
                    f"required token {token}"
                )

    report = {
        "schema_version": "1.0",
        "status": "PASS",
        "kaggle_version": version,
        "exact_version_ref": "owner/example-dataset/7",
        "signatures": observed,
        "dataset_create_version_signature": create_params,
        "dataset_status_exposes_current_version_number": True,
        "dataset_list_files_accepts_version_number": True,
        "dataset_download_file_accepts_version_number": True,
    }
    Path("kaggle-api-contract-report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
