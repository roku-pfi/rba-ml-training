"""Step 2: build a stratified per-user subset of the raw dataset.

The raw Wiefling CSV is ~8.4 GB / ~33M rows, which will not fit in memory. This
builder streams it in chunks and samples WHOLE users (never individual rows) so
each sampled user keeps their full login history, which every feature depends on.
All users who experienced an account takeover are kept, so the rare positive class
survives at subset scale.

Data-quality note: the dataset contains a sentinel user ID (e.g.
-4324475583306591935) with ~14M logins — an aggregation bucket for
non-attributable logins, not a real person. Such non-human accounts are dropped
via `--max-user-logins` (any user with more logins than the cap is excluded), which
restores the real per-user distribution (median 2, mean ~4).

Two passes over the file:
  1. Read only (User ID, Is Account Takeover) to count logins per user and find
     every takeover user.
  2. Choose the users to keep, then stream again and write their rows.

Reads:  data/raw/rba-dataset.csv
Writes: data/subset/logins.parquet  (columns renamed to rba_features.schema,
        timestamp parsed, tagged data_source='real')

Usage:
    python -m ml.ingest.subset --raw data/raw/rba-dataset.csv --users 50000
"""

from __future__ import annotations

import argparse
import random
import time
from collections import Counter

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from rba_features import schema

RAW_USER_COL = "User ID"
RAW_TAKEOVER_COL = "Is Account Takeover"
RAW_TS_COL = "Login Timestamp"
TS_FIELD = schema.RAW_TO_FIELD[RAW_TS_COL]  # "login_timestamp"
USER_FIELD = schema.RAW_TO_FIELD[RAW_USER_COL]  # "user_id"


def _first_pass(raw: str, chunksize: int) -> tuple[Counter, set[int]]:
    """Return (login_counts, takeover_users) by streaming two columns."""
    counts: Counter = Counter()
    takeover_users: set[int] = set()
    rows = 0
    for chunk in pd.read_csv(
        raw,
        usecols=[RAW_USER_COL, RAW_TAKEOVER_COL],
        dtype={RAW_USER_COL: "int64", RAW_TAKEOVER_COL: "bool"},
        chunksize=chunksize,
    ):
        rows += len(chunk)
        counts.update(chunk[RAW_USER_COL].value_counts().to_dict())
        tk = chunk.loc[chunk[RAW_TAKEOVER_COL], RAW_USER_COL].to_numpy().tolist()
        takeover_users.update(tk)
    print(f"  scanned {rows:,} rows")
    return counts, takeover_users


def _select_users(
    counts: Counter,
    takeover_users: set[int],
    budget: int,
    max_user_logins: int,
    seed: int,
) -> tuple[set[int], int]:
    """Keep every (eligible) takeover user + a random sample of eligible users.

    A user is "eligible" if their login count is <= max_user_logins; this drops
    non-human sentinel/bot accounts. Returns (selected, n_dropped_non_human).
    """
    dropped = {u for u, c in counts.items() if c > max_user_logins}
    eligible = {u for u in counts if u not in dropped}
    eligible_takeover = takeover_users & eligible
    non_takeover = list(eligible - eligible_takeover)
    random.Random(seed).shuffle(non_takeover)
    n_non = max(0, budget - len(eligible_takeover))
    selected = set(eligible_takeover) | set(non_takeover[:n_non])
    return selected, len(dropped)


def _transform(chunk: pd.DataFrame) -> pd.DataFrame:
    """Rename to canonical fields, parse the timestamp, tag the source."""
    out = chunk.rename(columns=schema.RAW_TO_FIELD)
    out[TS_FIELD] = pd.to_datetime(out[TS_FIELD], errors="coerce")
    out[schema.SOURCE_FIELD] = schema.SOURCE_REAL
    return out


def _second_pass(
    raw: str, selected: set[int], out_path: str, chunksize: int
) -> tuple[int, int]:
    """Stream again, keep selected users' rows, write parquet. Returns (rows, positives)."""
    writer: pq.ParquetWriter | None = None
    kept = 0
    positives = 0
    for chunk in pd.read_csv(raw, chunksize=chunksize):
        sub = chunk[chunk[RAW_USER_COL].isin(selected)]
        if sub.empty:
            continue
        sub = _transform(sub)
        kept += len(sub)
        positives += int(sub[schema.LABEL].sum())
        table = pa.Table.from_pandas(sub, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(out_path, table.schema)
        writer.write_table(table)
    if writer is not None:
        writer.close()
    return kept, positives


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a stratified per-user subset.")
    parser.add_argument("--raw", default="data/raw/rba-dataset.csv", help="Path to the raw CSV.")
    parser.add_argument("--out", default="data/subset/logins.parquet")
    parser.add_argument("--users", type=int, default=50_000, help="Target number of users to keep.")
    parser.add_argument(
        "--max-user-logins",
        type=int,
        default=10_000,
        help="Drop users with more logins than this (non-human sentinel/bot accounts).",
    )
    parser.add_argument("--chunksize", type=int, default=1_000_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    import os

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    t0 = time.time()
    print("Pass 1/2: counting logins per user + takeover labels ...")
    counts, takeover_users = _first_pass(args.raw, args.chunksize)
    print(f"  unique users: {len(counts):,} | takeover users: {len(takeover_users):,}")

    selected, n_dropped = _select_users(
        counts, takeover_users, args.users, args.max_user_logins, args.seed
    )
    print(f"  dropped {n_dropped:,} non-human users (> {args.max_user_logins:,} logins)")
    print(f"Selected {len(selected):,} users (all eligible takeover users + random sample).")

    print("Pass 2/2: writing subset ...")
    kept, positives = _second_pass(args.raw, selected, args.out, args.chunksize)

    print("-" * 60)
    print(f"Wrote {kept:,} rows to {args.out}")
    print(f"  positive rows (account takeover): {positives:,} ({positives / max(kept, 1):.4%})")
    print(f"  elapsed: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
