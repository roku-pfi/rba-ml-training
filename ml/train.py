"""Step 5: train the Freeman scorer + reference baselines and evaluate them.

Primary model: Freeman likelihood-ratio scorer (unsupervised-ish; learns "normal").
Reference baselines (supervised on the rare is_account_takeover label): LogisticRegression,
RandomForest, LightGBM. Uses the shared rba_features vectors (train/serve parity) and a
strictly chronological 70/15/15 split. Metrics are RBA-appropriate (PR-AUC, recall @ low
FPR, challenge rate) and broken down by history depth. See docs ADR-0004.

Usage:
    python -m ml.train --model all
"""

from __future__ import annotations

import argparse
import os
import pickle
import time

import numpy as np
import pandas as pd

from rba_features import schema
from rba_features.features import FEATURE_NAMES

from ml.featurize import build_features
from ml.metrics import evaluate
from ml.models.freeman import FreemanScorer

LABEL = schema.LABEL


def _split_labels(ts: pd.Series, cut) -> pd.Series:
    """Chronological train/test split at a single timestamp cut."""
    return pd.Series(np.where(ts <= cut, "train", "test"), index=ts.index)


def _fmt(x: float) -> str:
    return "nan" if x != x else f"{x:.4f}"


def _train_baselines(Xtr, ytr):
    from lightgbm import LGBMClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    pos = int(ytr.sum())
    neg = int(len(ytr) - pos)
    spw = (neg / pos) if pos else 1.0

    models = {
        "logreg": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1000, class_weight="balanced"),
        ),
        "rf": RandomForestClassifier(
            n_estimators=300, class_weight="balanced_subsample",
            n_jobs=-1, random_state=42,
        ),
        "lgbm": LGBMClassifier(
            n_estimators=400, learning_rate=0.05, scale_pos_weight=spw,
            random_state=42, n_jobs=-1, verbose=-1,
        ),
    }
    fitted = {}
    for name, m in models.items():
        m.fit(Xtr, ytr)
        fitted[name] = m
    return fitted


def _report(name: str, res: dict, size_kb: float, lat_us: float, lines: list[str]) -> None:
    op = res["operating_point"]
    lines.append(f"\n### {name}")
    lines.append(f"- PR-AUC: {_fmt(res['pr_auc'])} | ROC-AUC: {_fmt(res['roc_auc'])}")
    lines.append(
        f"- @FPR≈{op.target_fpr:.0%}: recall={_fmt(op.recall)} "
        f"(achieved FPR {_fmt(op.achieved_fpr)}, challenge rate {_fmt(op.challenge_rate)})"
    )
    lines.append(f"- model size: {size_kb:.1f} KB | scoring latency: {lat_us:.1f} µs/login")
    lines.append("- recall by history depth:")
    lines.append("  | bucket | n | positives | recall |")
    lines.append("  |---|---|---|---|")
    for b in res["by_depth"]:
        lines.append(f"  | {b['bucket']} | {b['n']:,} | {b['positives']} | {_fmt(b['recall'])} |")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train RBA scorer + baselines.")
    parser.add_argument("--data", default="data/subset/logins.parquet")
    parser.add_argument("--model", choices=["freeman", "logreg", "rf", "lgbm", "all"],
                        default="all")
    parser.add_argument("--target-fpr", type=float, default=0.01)
    parser.add_argument("--train-frac", type=float, default=0.70)
    parser.add_argument("--out", default="artifacts/step5")
    parser.add_argument("--report", default="reports/step5/metrics.md")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    os.makedirs(os.path.dirname(args.report) or ".", exist_ok=True)

    full = pd.read_parquet(args.data).reset_index(drop=True)
    # Account-takeover labels only cover Feb–Nov 2020; a calendar split would leave the
    # test tail with zero positives. Restrict to the label-covered window and split
    # chronologically within it. (See docs finding for Step 5.)
    last_pos = full.loc[full[LABEL], "login_timestamp"].max()
    df = full[full["login_timestamp"] <= last_pos].reset_index(drop=True)
    n_excluded = len(full) - len(df)
    cut = df["login_timestamp"].quantile(args.train_frac)
    want = ["freeman", "logreg", "rf", "lgbm"] if args.model == "all" else [args.model]

    lines: list[str] = ["# Step 5 — Freeman scorer + baselines"]
    lines.append(
        f"\nLabel-covered window: {df['login_timestamp'].min()} → {last_pos} "
        f"({len(df):,} rows, {df['user_id'].nunique():,} users, {int(df[LABEL].sum())} "
        f"positives). Excluded {n_excluded:,} later rows with no takeover labels. "
        f"Chronological split at {cut} (train {args.train_frac:.0%} / test)."
    )

    # ---- Freeman (works on raw categorical stream) ----
    if "freeman" in want:
        w = df.copy()
        w["split"] = _split_labels(w["login_timestamp"], cut)
        order = w.sort_values(["user_id", "login_timestamp"], kind="mergesort")
        w["depth"] = order.groupby("user_id", sort=False).cumcount().sort_index()
        y = w[LABEL].to_numpy().astype(int)

        scorer = FreemanScorer().fit(w[w["split"] == "train"])
        t0 = time.time()
        scores = scorer.score_frame(w)
        lat_us = (time.time() - t0) / len(w) * 1e6

        test = (w["split"] == "test").to_numpy()
        n_pos_test = int(y[test].sum())
        res = evaluate(y[test], scores[test], w["depth"].to_numpy()[test], args.target_fpr)

        path = os.path.join(args.out, "freeman.pkl")
        with open(path, "wb") as fh:
            pickle.dump(scorer, fh)
        size_kb = os.path.getsize(path) / 1024
        lines.append(f"\n> Freeman test set: {int(test.sum()):,} logins, {n_pos_test} positives.")
        _report("freeman (PRIMARY)", res, size_kb, lat_us, lines)

    # ---- Supervised baselines (on shared feature vectors) ----
    supervised = [m for m in want if m != "freeman"]
    if supervised:
        t0 = time.time()
        feat = build_features(df)
        print(f"[featurize] {len(feat):,} rows in {time.time() - t0:.1f}s")
        feat["split"] = _split_labels(feat["login_timestamp"], cut)
        tr = feat["split"] == "train"
        te = (feat["split"] == "test").to_numpy()
        Xtr, ytr = feat.loc[tr, list(FEATURE_NAMES)], feat.loc[tr, LABEL].astype(int)
        Xte = feat.loc[te, list(FEATURE_NAMES)]
        yte = feat.loc[te, LABEL].to_numpy().astype(int)
        depth_te = feat.loc[te, "user_login_count"].to_numpy()

        fitted = _train_baselines(Xtr, ytr)
        for name in supervised:
            model = fitted[name]
            t0 = time.time()
            proba = model.predict_proba(Xte)[:, 1]
            lat_us = (time.time() - t0) / max(len(Xte), 1) * 1e6
            res = evaluate(yte, proba, depth_te, args.target_fpr)
            path = os.path.join(args.out, f"{name}.pkl")
            with open(path, "wb") as fh:
                pickle.dump(model, fh)
            size_kb = os.path.getsize(path) / 1024
            _report(name, res, size_kb, lat_us, lines)

    report = "\n".join(lines) + "\n"
    with open(args.report, "w") as fh:
        fh.write(report)
    print(report)
    print(f"[report] written to {args.report}")


if __name__ == "__main__":
    main()
