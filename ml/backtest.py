"""Backtest of the adopted model on takeovers no choice was made on (C1).

Every version choice (freeman-0.2 → 0.3 → 0.4, dropping the hour, the hybrid
over the 22/38 variant) and the supervised threshold were read off the Step 5
test split: the last 30% of the label window, 38 takeovers. The 103 takeovers
before it were never looked at: Freeman learns without labels, and no metric
was computed on them.

So the adopted configuration is frozen and replayed earlier in time: global
tables from a training window, every user scored past-only, measured on the
next window with the same metrics as Step 5 (ROC-AUC, recall at 1% FPR of that
window). Intervals resample users (a user's logins are not independent).

A second table checks whether a threshold carries over: set at 1% FPR on a
calibration slice scored like unseen data (tables from before it), then
applied to the evaluation window.

A third does the same for the supervised second opinion (logreg-0.2.0,
ADR-0027): refitted on each training window's labels, measured on the next
window, and its shipped threshold (0.869, read off the test split) applied
unchanged to see what rate it actually fires at.

Usage (from rba-ml-training/, venv active):

    python -m ml.backtest --report reports/backtest.md
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
from rba_features import schema
from sklearn.metrics import roc_auc_score

from rba_features.features import FEATURE_NAMES

from ml.featurize import build_features
from ml.models.freeman import FreemanScorer
from ml.metrics import recall_at_fpr
from ml.train import build_models
from ml.models.hierarchical import (
    HIERARCHIES,
    fit_lambdas,
    global_tables,
    level_matrix,
    logrisk,
    prep,
    rooted,
)

LABEL = schema.LABEL
SHIPPED_LOGREG_THRESHOLD = 0.869127494041951  # artifacts/serving/logreg-0.2.0.json
UNIFORM = {n: (np.full(4, 0.25), np.full(4, 0.25)) for n in HIERARCHIES}
# (train up to, evaluate up to) as quantiles of the label window, in time.
FOLDS = [(0.40, 0.70), (0.50, 0.70), (0.30, 0.50)]
REFERENCE = (0.70, 1.00)  # the Step 5 split every choice was made on


def _rooted_lambdas(d: pd.DataFrame, train: np.ndarray, y: np.ndarray) -> dict:
    """The 22/38 variant's interpolation weights, fitted as in hierarchical_eval:
    on the legitimate logins of the training window's last 40%."""
    t = d[train].reset_index(drop=True)
    early = (t["login_timestamp"] < t["login_timestamp"].quantile(0.60)).to_numpy()
    held = ~early & (y[train] == 0)
    lv_fit = level_matrix(t, global_tables(t[early]))
    out = {}
    for name in HIERARCHIES:
        g, u = lv_fit[name]
        warm = held & (u.sum(axis=1) > 0)
        lam_g = fit_lambdas(g[held])
        out[name] = (lam_g, fit_lambdas(rooted(u, g @ lam_g)[warm]))
    return out


def scores(df: pd.DataFrame, d: pd.DataFrame, train: np.ndarray, y: np.ndarray) -> dict[str, np.ndarray]:
    """Adopted model and the ones it was chosen over, fitted on ``train`` only."""
    lv = level_matrix(d, global_tables(d[train]))
    network = logrisk(lv, UNIFORM, ["network"], root_at_global=True)
    agent = FreemanScorer(beta=5.0, features=("device_type", "os", "browser")).fit(df[train])
    flat = FreemanScorer(beta=5.0).fit(df[train])
    flat_contrib = flat.contributions_frame(df)
    hour = flat_contrib[:, flat.features.index("hour")]
    rooted_all = logrisk(lv, _rooted_lambdas(d, train, y), root_at_global=True)
    return {
        "adopted: freeman-0.4.0 hybrid": network + agent.score_frame(df),
        "rejected: rooted network + agent (22/38)": rooted_all,
        "rejected: rooted + hour": rooted_all + hour,
        "earlier: flat freeman-0.3.0": flat_contrib.sum(axis=1),
    }


def by_user_interval(y, flagged, users, n_boot=2000, seed=7) -> tuple[float, float]:
    """95% interval of recall, resampling users with replacement."""
    rng = np.random.default_rng(seed)
    pos = y == 1
    ids, inv = np.unique(users[pos], return_inverse=True)
    hits = np.bincount(inv, weights=flagged[pos].astype(float), minlength=len(ids))
    counts = np.bincount(inv, minlength=len(ids))
    out = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(ids), len(ids))
        out.append(hits[pick].sum() / max(counts[pick].sum(), 1))
    return float(np.quantile(out, 0.025)), float(np.quantile(out, 0.975))


def logreg_section(df: pd.DataFrame, fpr: float) -> list[str]:
    """Supervised second opinion: refit per fold, in-window recall, shipped cut-off."""
    feat = build_features(df)  # past-only per user, so one pass serves every fold
    X = feat[list(FEATURE_NAMES)]
    ts = feat["login_timestamp"]
    y = feat[LABEL].to_numpy().astype(int)
    users = feat["user_id"].to_numpy()
    out = [
        "",
        "## C. Supervised second opinion (logreg-0.2.0)",
        "",
        "Refitted on each training window's labels (same estimator as Step 5). "
        f"Recall at {fpr:.0%} FPR of the evaluation window, then the shipped "
        f"threshold {SHIPPED_LOGREG_THRESHOLD:.3f} applied unchanged.",
        "",
        "| fold | takeovers | ROC-AUC | recall@1%FPR (95% CI) | shipped cut-off: FPR | shipped cut-off: recall |",
        "|---|---|---|---|---|---|",
    ]
    for fit_q, eval_q in [*FOLDS, REFERENCE]:
        fit_cut, eval_cut = ts.quantile(fit_q), ts.quantile(eval_q)
        train = (ts < fit_cut).to_numpy()
        held = ((ts >= fit_cut) & (ts <= eval_cut)).to_numpy()
        model = build_models(y[train])["logreg"].fit(X[train], y[train])
        sh, yh = model.predict_proba(X[held])[:, 1], y[held]
        op = recall_at_fpr(yh, sh, fpr)
        flagged = sh >= op.threshold
        lo, hi = by_user_interval(yh, flagged, users[held])
        fixed = sh >= SHIPPED_LOGREG_THRESHOLD
        tag = f"{fit_q:.0%} → {eval_q:.0%}" + (" (Step 5 test, used for choices)" if (fit_q, eval_q) == REFERENCE else "")
        out.append(
            f"| {tag} | {int(yh.sum())} | {roc_auc_score(yh, sh):.3f} | "
            f"{op.recall:.2f} ({int(flagged[yh == 1].sum())}/{int(yh.sum())}; {lo:.2f}–{hi:.2f}) | "
            f"{fixed[yh == 0].mean():.2%} | {fixed[yh == 1].mean():.2f} ({int(fixed[yh == 1].sum())}/{int(yh.sum())}) |"
        )
        print(out[-1], flush=True)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/subset/logins.parquet")
    parser.add_argument("--report", default="reports/backtest.md")
    parser.add_argument("--fpr", type=float, default=0.01)
    args = parser.parse_args()

    full = pd.read_parquet(args.data).reset_index(drop=True)
    last_pos = full.loc[full[LABEL], "login_timestamp"].max()
    df = full[full["login_timestamp"] <= last_pos].reset_index(drop=True)
    d = prep(df)
    ts = df["login_timestamp"]
    y = df[LABEL].to_numpy().astype(int)
    users = df["user_id"].to_numpy()

    lines = [
        "# Backtest: the adopted model on takeovers no choice was made on",
        "",
        "## A. Same metrics as Step 5, on each later window",
        "",
        "Frozen configuration, refitted on each training window; recall at "
        f"{args.fpr:.0%} FPR of the evaluation window (as the reported 20/38). "
        "95% intervals resample users.",
        "",
        "| fold | variant | takeovers | ROC-AUC | recall@1%FPR (95% CI) |",
        "|---|---|---|---|---|",
    ]
    transfer = [
        "",
        "## B. Does a threshold carry over?",
        "",
        "Adopted model only. Tables from the training window's first 75%; threshold "
        f"at {args.fpr:.0%} FPR on its last 25% (scored as unseen data); applied, "
        "unchanged, to the evaluation window.",
        "",
        "| fold | FPR on the evaluation window | recall |",
        "|---|---|---|",
    ]
    for fit_q, eval_q in [*FOLDS, REFERENCE]:
        fit_cut, eval_cut = ts.quantile(fit_q), ts.quantile(eval_q)
        train = (ts < fit_cut).to_numpy()
        held = ((ts >= fit_cut) & (ts <= eval_cut)).to_numpy()
        tag = f"{fit_q:.0%} → {eval_q:.0%}" + (" (Step 5 test, used for choices)" if (fit_q, eval_q) == REFERENCE else "")
        for name, s in scores(df, d, train, y).items():
            yh, sh = y[held], s[held]
            op = recall_at_fpr(yh, sh, args.fpr)
            flagged = sh >= op.threshold
            lo, hi = by_user_interval(yh, flagged, users[held])
            lines.append(
                f"| {tag} | {name} | {int(yh.sum())} | {roc_auc_score(yh, sh):.3f} | "
                f"{op.recall:.2f} ({int(flagged[yh == 1].sum())}/{int(yh.sum())}; {lo:.2f}–{hi:.2f}) |"
            )
            print(lines[-1], flush=True)

        # B: threshold from a calibration slice the tables never saw.
        cal_cut = ts[train].quantile(0.75)
        fit_part = (ts < cal_cut).to_numpy()
        cal = train & ~fit_part
        s = scores(df, d, fit_part, y)["adopted: freeman-0.4.0 hybrid"]
        thr = float(np.quantile(s[cal & (y == 0)], 1 - args.fpr))
        flagged = s[held] >= thr
        yh = y[held]
        transfer.append(
            f"| {tag} | {flagged[yh == 0].mean():.2%} | "
            f"{flagged[yh == 1].mean():.2f} ({int(flagged[yh == 1].sum())}/{int(yh.sum())}) |"
        )
        print(transfer[-1], flush=True)
    lines += transfer
    lines += logreg_section(df, args.fpr)

    os.makedirs(os.path.dirname(args.report) or ".", exist_ok=True)
    with open(args.report, "w") as fh:
        fh.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
