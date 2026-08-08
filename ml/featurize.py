"""Step 5 helper: build the baseline feature matrix from the login subset.

Uses the shared `rba_features` replay driver so the supervised baselines are trained
on exactly the feature vectors the online service would produce (train/serve parity).
Each output row carries the feature vector plus the metadata needed for a
chronological split, leakage analysis, and history-depth breakdowns.

Reads:  data/subset/logins.parquet
Writes: data/features/vectors.parquet  (optional; train.py builds in-memory too)
"""

from __future__ import annotations

import argparse
import os
import time

import pandas as pd

from rba_features import schema
from rba_features.features import FEATURE_NAMES
from rba_features.replay import replay_user

# Fields the features read (a subset of the event); kept explicit for clarity.
EVENT_COLS = [
    "login_timestamp",
    "ip_address",
    "asn",
    "country",
    "device_type",
    "os",
    "browser",
    "login_successful",
]

# Carried through for splitting / evaluation (NOT model inputs).
META_COLS = ["user_id", "login_timestamp", "is_attack_ip", schema.LABEL]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return a DataFrame of feature vectors + metadata, one row per login.

    Rows are ordered by (user_id, login_timestamp); metadata is aligned to that
    same order. `user_login_count` doubles as the history-depth of each event.
    """
    df = df.sort_values(["user_id", "login_timestamp"]).reset_index(drop=True)
    rows: list[dict] = []
    for _, g in df.groupby("user_id", sort=False):
        events = g[EVENT_COLS].to_dict("records")
        rows.extend(replay_user(events))
    feat = pd.DataFrame(rows, columns=list(FEATURE_NAMES))
    for c in META_COLS:
        feat[c] = df[c].to_numpy()
    return feat


def main() -> None:
    parser = argparse.ArgumentParser(description="Build baseline feature vectors.")
    parser.add_argument("--data", default="data/subset/logins.parquet")
    parser.add_argument("--out", default="data/features/vectors.parquet")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    t0 = time.time()
    df = pd.read_parquet(args.data)
    feat = build_features(df)
    feat.to_parquet(args.out, index=False)
    print(f"wrote {len(feat):,} feature rows to {args.out} in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
