"""Freeman in a deployment whose population differs from training (ADR-0042).

Three measurements on the Step 5 split (finding 2026-10-23):

A. Browser / OS as families (rba_features.canonical) vs the raw version strings.
B. The deployment-learned population prior, p_global = (N_dep(v) + kappa *
   p_train(v)) / (N_dep + kappa), with N_dep counted past-only in time order,
   on the same population (does it cost ranking?).
C. A population shift: the prior is fitted on Norwegian users only, thresholds
   are set on their test logins, and applied to everyone else, as fixed bands
   are in a deployment.

Usage (from rba-ml-training/, venv active):

    python -m ml.population_shift --report reports/population_shift.md
"""

from __future__ import annotations

import argparse
import math
import os
from collections import Counter

import numpy as np
import pandas as pd
from rba_features import schema

from ml.metrics import evaluate
from ml.models.freeman import FreemanScorer

LABEL = schema.LABEL


class _RawStrings(FreemanScorer):
    """Freeman before ADR-0042: browser / OS strings as the dataset spells them."""

    @staticmethod
    def _prep(df: pd.DataFrame) -> pd.DataFrame:
        d = df.copy()
        d["hour"] = pd.to_datetime(d["login_timestamp"]).dt.hour
        return d


def score_online(sc: FreemanScorer, df: pd.DataFrame, kappa: float | None) -> np.ndarray:
    """Per-row logrisk with user history and deployment population both past-only."""
    d = sc._prep(df).reset_index(drop=True)
    order = np.argsort(d["login_timestamp"].to_numpy(), kind="mergesort")
    feats = sc.features
    cols = {f: d[f].astype(str).to_numpy() for f in feats}
    users = d["user_id"].to_numpy()
    user_counts: dict[str, dict[str, Counter]] = {}
    user_totals: dict[str, int] = {}
    pop = {f: Counter() for f in feats}
    pop_total = 0
    out = np.zeros(len(d))
    for i in order:
        u = users[i]
        uc = user_counts.setdefault(u, {f: Counter() for f in feats})
        t = user_totals.get(u, 0)
        logrisk = 0.0
        for f in feats:
            v = cols[f][i]
            p_train = sc._p_global(f, v)
            pg = p_train if kappa is None else (pop[f][v] + kappa * p_train) / (pop_total + kappa)
            pu = (uc[f][v] + sc.beta * pg) / (t + sc.beta)
            logrisk += math.log(pg) - math.log(pu)
        out[i] = logrisk
        for f in feats:
            uc[f][cols[f][i]] += 1
            pop[f][cols[f][i]] += 1
        user_totals[u] = t + 1
        pop_total += 1
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/subset/logins.parquet")
    parser.add_argument("--report", default="reports/population_shift.md")
    args = parser.parse_args()

    full = pd.read_parquet(args.data).reset_index(drop=True)
    last_pos = full.loc[full[LABEL], "login_timestamp"].max()
    df = full[full["login_timestamp"] <= last_pos].reset_index(drop=True)
    cut = df["login_timestamp"].quantile(0.70)
    df["split"] = np.where(df["login_timestamp"] < cut, "train", "test")
    order = df.sort_values(["user_id", "login_timestamp"], kind="mergesort")
    df["depth"] = order.groupby("user_id", sort=False).cumcount().sort_index()
    y = df[LABEL].to_numpy().astype(int)
    test = (df["split"] == "test").to_numpy()
    train = df[df["split"] == "train"]
    lines = ["# Freeman under a population shift (ADR-0042)", ""]

    def row(name: str, scores: np.ndarray, mask: np.ndarray) -> str:
        r = evaluate(y[mask], scores[mask], df["depth"].to_numpy()[mask], 0.01)
        op = r["operating_point"]
        hits = int(round(op.recall * y[mask].sum()))
        return f"| {name} | {r['roc_auc']:.3f} | {op.recall:.3f} ({hits}/{int(y[mask].sum())}) |"

    lines += ["## A. Browser / OS as families", "", "| variant | ROC-AUC | recall@1%FPR |", "|---|---|---|"]
    raw = _RawStrings(beta=5.0).fit(train)
    fam = FreemanScorer(beta=5.0).fit(train)
    lines.append(row("raw strings (freeman-0.2.0)", raw.score_frame(df), test))
    lines.append(row("families (freeman-0.3.0)", fam.score_frame(df), test))

    lines += ["", "## B. Deployment-learned population, same population", "",
              "| kappa | ROC-AUC | recall@1%FPR |", "|---|---|---|"]
    for kappa in (None, 1000, 100):
        lines.append(row("training only" if kappa is None else str(kappa), score_online(fam, df, kappa), test))

    user_country = df.groupby("user_id")["country"].agg(lambda s: s.mode().iat[0])
    norwegian = df["user_id"].isin(set(user_country[user_country == "NO"].index)).to_numpy()
    src, tgt = test & norwegian, test & ~norwegian
    shifted = FreemanScorer(beta=5.0).fit(df[(df["split"] == "train") & norwegian])
    lines += [
        "", "## C. Population shift: thresholds set on Norwegian users, applied to the rest", "",
        f"Source: {int(src.sum()):,} test logins ({int(y[src].sum())} positives); "
        f"target: {int(tgt.sum()):,} ({int(y[tgt].sum())} positives).", "",
        "| kappa | genuine median logrisk (source → target) | target FPR / recall at the source's 1% / 5% / 20% thresholds |",
        "|---|---|---|",
    ]
    for kappa in (None, 1000, 100):
        s = score_online(shifted, df, kappa)
        genuine_src = s[src & (y == 0)]
        cells = []
        for fpr in (0.01, 0.05, 0.20):
            thr = np.quantile(genuine_src, 1 - fpr)
            cells.append(f"{(s[tgt & (y == 0)] > thr).mean():.3f} / {(s[tgt & (y == 1)] > thr).mean():.2f}")
        name = "training only" if kappa is None else str(kappa)
        lines.append(f"| {name} | {np.median(genuine_src):.1f} → {np.median(s[tgt & (y == 0)]):.1f} | {' · '.join(cells)} |")

    os.makedirs(os.path.dirname(args.report) or ".", exist_ok=True)
    with open(args.report, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
