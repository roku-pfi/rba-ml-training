"""Step 2: build a stratified per-user subset of the raw dataset.

Samples WHOLE users (not individual rows) so each sampled user keeps their full
login history, which the features depend on. Oversamples users who experienced an
account takeover so the rare positive class is usable at subset scale.

Reads:  data/raw/<the big csv>
Writes: data/subset/logins.parquet

Implemented in Step 2.
"""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a stratified per-user subset.")
    parser.add_argument("--raw", required=True, help="Path to the raw CSV.")
    parser.add_argument("--out", default="data/subset/logins.parquet")
    parser.add_argument("--users", type=int, default=50_000, help="Users to sample.")
    parser.add_argument("--seed", type=int, default=42)
    parser.parse_args()
    raise NotImplementedError("Subset builder is implemented in Phase 1, Step 2.")


if __name__ == "__main__":
    main()
