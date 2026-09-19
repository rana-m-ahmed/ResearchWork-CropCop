from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401

from cropcop_je.trackb_r07 import TrackBError


def main() -> int:
    argparse.ArgumentParser(
        description=(
            "Deprecated Track-B raw historical-source preparer. "
            "Post-closure reconstruction of the complete 117,546-image raw surface is disabled "
            "because it could reopen consumed V1-test image bytes."
        )
    ).parse_args()
    raise TrackBError(
        "Raw 117,546-image historical-source reconstruction is disabled after V1-test closure. "
        "Use build_trackb_historical_compare.py on the frozen V1 train+validation development surface "
        "(92,744 images), which is prediction-blind and automatically caps evidence at EXT-S. "
        "EXT-I is available only if a complete pre-test cryptographically bound historical comparison "
        "representation is genuinely recovered without reopening V1-test image bytes."
    )


if __name__ == "__main__":
    raise SystemExit(main())
