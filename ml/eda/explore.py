"""Step 3: exploratory data analysis on the login subset.

Answers the questions that drive every later decision:
    - class balance (how rare is account takeover? how common is attack-IP?)
    - logins-per-user distribution (history depth)
    - missingness per column ("-" and NaN both count as missing)
    - categorical cardinality (how big will one-hot / target spaces be?)
    - timestamp span (for the chronological split)

Prints a text report and writes plots under reports/eda/.

Usage:
    python -m ml.eda.explore --data data/subset/logins.parquet
"""

from __future__ import annotations

import argparse
import os

os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.getcwd(), ".mplcache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from rba_features import schema

# Values that mean "missing" in the raw data in addition to real NaN.
MISSING_TOKENS = {"-", "", "nan", "NaN", "None"}

CATEGORICAL = [
    "country", "region", "city", "asn", "ip_address",
    "device_type", "os", "browser", "user_agent",
]
BOOL_COLS = ["login_successful", "is_attack_ip", "is_account_takeover"]


def _missing_rate(s: pd.Series) -> float:
    n_missing = s.isna().sum()
    if s.dtype == object or str(s.dtype).startswith("str"):
        n_missing += s.astype("string").str.strip().isin(MISSING_TOKENS).sum()
    return float(n_missing) / max(len(s), 1)


def _section(title: str) -> None:
    print("\n" + "=" * 66 + f"\n{title}\n" + "=" * 66)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run EDA on the login subset.")
    parser.add_argument("--data", default="data/subset/logins.parquet")
    parser.add_argument("--out", default="reports/eda")
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)

    df = pd.read_parquet(args.data)
    ts = df[schema.RAW_TO_FIELD["Login Timestamp"]]

    _section("SHAPE")
    print(f"rows: {len(df):,} | columns: {df.shape[1]} | users: {df['user_id'].nunique():,}")
    print(f"memory: {df.memory_usage(deep=True).sum() / 1e6:.1f} MB")

    _section("CLASS BALANCE (booleans)")
    for c in BOOL_COLS:
        pos = int(df[c].sum())
        print(f"{c:22s} True={pos:>8,}  ({pos / len(df):.4%})   "
              f"users_with_any={df.loc[df[c], 'user_id'].nunique():,}")

    _section("LOGINS PER USER (history depth)")
    vc = df["user_id"].value_counts()
    for q in (0.5, 0.9, 0.99, 1.0):
        print(f"  p{int(q * 100):<3d} = {vc.quantile(q):.0f}")
    print(f"  mean = {vc.mean():.2f}")
    print(f"  users with only 1 login: {(vc == 1).sum():,} ({(vc == 1).mean():.1%}) "
          f"-> no history, RBA can't help these yet")

    _section("MISSINGNESS ('-' and NaN)")
    for c in df.columns:
        r = _missing_rate(df[c])
        if r > 0:
            print(f"{c:22s} {r:.2%}")

    _section("CATEGORICAL CARDINALITY")
    for c in CATEGORICAL:
        if c in df.columns:
            print(f"{c:22s} distinct={df[c].nunique():>8,}")

    _section("TIME SPAN")
    print(f"  from {ts.min()}  to  {ts.max()}")
    print(f"  span: {(ts.max() - ts.min()).days} days  |  NaT timestamps: {ts.isna().sum():,}")

    _make_plots(df, vc, ts, args.out)
    _section("DONE")
    print(f"plots written to {args.out}/")


def _make_plots(df: pd.DataFrame, vc: pd.Series, ts: pd.Series, out: str) -> None:
    # Logins-per-user (clipped so the long tail doesn't hide the mass at 1-2).
    fig, ax = plt.subplots(figsize=(7, 4))
    vc.clip(upper=30).plot.hist(bins=30, ax=ax)
    ax.set(title="Logins per user (clipped at 30)", xlabel="logins", ylabel="users")
    fig.tight_layout()
    fig.savefig(f"{out}/logins_per_user.png", dpi=110)
    plt.close(fig)

    # Logins over time.
    fig, ax = plt.subplots(figsize=(8, 4))
    ts.dropna().dt.floor("D").value_counts().sort_index().plot(ax=ax)
    ax.set(title="Logins per day", xlabel="date", ylabel="logins")
    fig.tight_layout()
    fig.savefig(f"{out}/logins_per_day.png", dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    main()
