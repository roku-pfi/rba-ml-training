"""Step 6: the mandatory `is_attack_ip` leakage A/B experiment.

The Wiefling dataset ships an `is_attack_ip` flag that is *derived from the attack
itself*, so a supervised model handed that column can look excellent while learning
almost nothing about user behaviour. Plan section 6 therefore mandates training every
baseline twice on identical chronological splits:

    - Variant B (HONEST): behavioural/context features only (`rba_features` vectors).
    - Variant A (LEAKY):  Variant B + `is_attack_ip`.

If A >> B, the model is memorising known-bad IPs rather than learning behaviour, and
**B is the number the thesis reports**. We also score `is_attack_ip` on its own as the
pure-leakage ceiling, and report its coverage of the positive/negative classes so the
size of the effect is legible. Because there are only ~38 test positives, every metric
carries a paired bootstrap CI (and so does the A-B delta) — single-model deltas here
are noisy and must not be over-read (see the Step 5 caveats + dataset-sufficiency note).

Usage:
    python -m ml.leakage
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
from ml.metrics import recall_at_fpr
from ml.train import _split_labels, build_models

LABEL = schema.LABEL
LEAK = schema.LEAKAGE_SENSITIVE_FIELDS[0]  # "is_attack_ip"

BASE_FEATURES = list(FEATURE_NAMES)          # Variant B
LEAK_FEATURES = BASE_FEATURES + [LEAK]        # Variant A


def _fmt(x: float) -> str:
    return "nan" if x != x else f"{x:.3f}"


def _ci(lo: float, hi: float) -> str:
    return f"[{_fmt(lo)}, {_fmt(hi)}]"


def _roc_auc(y: np.ndarray, s: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score
    p = int(y.sum())
    if p == 0 or p == len(y):
        return float("nan")
    return float(roc_auc_score(y, s))


def _pr_auc(y: np.ndarray, s: np.ndarray) -> float:
    from sklearn.metrics import average_precision_score
    p = int(y.sum())
    if p == 0 or p == len(y):
        return float("nan")
    return float(average_precision_score(y, s))


def _recall_at(y: np.ndarray, s: np.ndarray, fpr: float) -> float:
    return recall_at_fpr(y, s, fpr).recall


def _point_metrics(y: np.ndarray, s: np.ndarray, fpr: float) -> dict:
    op = recall_at_fpr(y, s, fpr)
    return {
        "roc_auc": _roc_auc(y, s),
        "pr_auc": _pr_auc(y, s),
        "recall": op.recall,
        "achieved_fpr": op.achieved_fpr,
        "challenge_rate": op.challenge_rate,
    }


def paired_bootstrap(
    y: np.ndarray,
    score_a: np.ndarray,
    score_b: np.ndarray,
    fpr: float,
    n_boot: int = 2000,
    seed: int = 42,
) -> dict:
    """Percentile CIs for A, B and the paired A-B delta on ROC-AUC and recall@FPR.

    Each iteration resamples test rows with replacement ONCE and evaluates both
    variants on the same rows, so the delta CI captures the correlated noise between
    the two models (the honest way to ask "is A really better than B, or is it the
    38-positive lottery?").
    """
    rng = np.random.default_rng(seed)
    n = len(y)
    keys = ("roc_auc", "recall")
    a_samples = {k: [] for k in keys}
    b_samples = {k: [] for k in keys}
    d_samples = {k: [] for k in keys}

    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        ys = y[idx]
        if ys.sum() == 0 or ys.sum() == len(ys):
            continue  # undefined ranking metrics on a degenerate resample
        a_s, b_s = score_a[idx], score_b[idx]
        a_vals = {"roc_auc": _roc_auc(ys, a_s), "recall": _recall_at(ys, a_s, fpr)}
        b_vals = {"roc_auc": _roc_auc(ys, b_s), "recall": _recall_at(ys, b_s, fpr)}
        for k in keys:
            a_samples[k].append(a_vals[k])
            b_samples[k].append(b_vals[k])
            d_samples[k].append(a_vals[k] - b_vals[k])

    def pct(vals: list[float]) -> tuple[float, float]:
        arr = np.asarray(vals, dtype=float)
        arr = arr[~np.isnan(arr)]
        if arr.size == 0:
            return float("nan"), float("nan")
        return float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))

    return {
        "a": {k: pct(a_samples[k]) for k in keys},
        "b": {k: pct(b_samples[k]) for k in keys},
        "delta": {k: pct(d_samples[k]) for k in keys},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Step 6: is_attack_ip leakage A/B.")
    parser.add_argument("--data", default="data/subset/logins.parquet")
    parser.add_argument("--target-fpr", type=float, default=0.01)
    parser.add_argument("--train-frac", type=float, default=0.70)
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--out", default="artifacts/step6")
    parser.add_argument("--report", default="reports/step6/leakage.md")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    os.makedirs(os.path.dirname(args.report) or ".", exist_ok=True)

    full = pd.read_parquet(args.data).reset_index(drop=True)
    # Same label-covered restriction + chronological split as Step 5 (ADR-0007).
    last_pos = full.loc[full[LABEL], "login_timestamp"].max()
    df = full[full["login_timestamp"] <= last_pos].reset_index(drop=True)
    cut = df["login_timestamp"].quantile(args.train_frac)

    t0 = time.time()
    feat = build_features(df)  # carries is_attack_ip + label in its metadata columns
    print(f"[featurize] {len(feat):,} rows in {time.time() - t0:.1f}s")
    feat[LEAK] = feat[LEAK].astype(int)
    feat["split"] = _split_labels(feat["login_timestamp"], cut)

    tr = (feat["split"] == "train").to_numpy()
    te = (feat["split"] == "test").to_numpy()
    ytr = feat.loc[tr, LABEL].astype(int)
    yte = feat.loc[te, LABEL].to_numpy().astype(int)
    n_pos_test = int(yte.sum())
    fpr = args.target_fpr

    # --- Coverage of the leakage flag (why A can beat B at all) ---
    leak_te = feat.loc[te, LEAK].to_numpy().astype(int)
    cov_pos = float(leak_te[yte == 1].mean()) if n_pos_test else float("nan")
    cov_neg = float(leak_te[yte == 0].mean())

    models = ["logreg", "rf", "lgbm"]
    point: dict[str, dict[str, dict]] = {m: {} for m in models}
    boot: dict[str, dict] = {}

    for variant, cols in (("B", BASE_FEATURES), ("A", LEAK_FEATURES)):
        Xtr = feat.loc[tr, cols]
        Xte = feat.loc[te, cols]
        fitted = {name: est.fit(Xtr, ytr) for name, est in build_models(ytr).items()}
        for name in models:
            proba = fitted[name].predict_proba(Xte)[:, 1]
            point[name][variant] = _point_metrics(yte, proba, fpr)
            point[name].setdefault("_scores", {})[variant] = proba
            path = os.path.join(args.out, f"{name}_{variant}.pkl")
            with open(path, "wb") as fh:
                pickle.dump(fitted[name], fh)

    for name in models:
        boot[name] = paired_bootstrap(
            yte, point[name]["_scores"]["A"], point[name]["_scores"]["B"],
            fpr, n_boot=args.n_boot,
        )

    # is_attack_ip on its own — the pure-leakage ceiling.
    flag_pt = _point_metrics(yte, leak_te.astype(float), fpr)

    # ---- Report ----
    L: list[str] = ["# Step 6 — `is_attack_ip` leakage A/B experiment"]
    L.append(
        f"\nLabel-covered window split at {cut} (train {args.train_frac:.0%} / test). "
        f"Test set: **{int(te.sum()):,} logins, {n_pos_test} positives**. "
        f"Baselines identical to Step 5; the only change is the feature matrix. "
        f"CIs are 95% percentile paired bootstraps ({args.n_boot:,} resamples)."
    )
    L.append(
        f"\n**Leakage flag coverage (test):** `is_attack_ip`=1 on "
        f"**{cov_pos:.0%}** of takeovers vs **{cov_neg:.2%}** of legit logins."
    )

    L.append("\n## Per-model: Variant B (honest) vs Variant A (+is_attack_ip)\n")
    L.append("| model | variant | ROC-AUC | ROC-AUC 95% CI | PR-AUC | recall@1%FPR | recall 95% CI | challenge |")
    L.append("|---|---|---|---|---|---|---|---|")
    for name in models:
        for v in ("B", "A"):
            p = point[name][v]
            bkey = "b" if v == "B" else "a"
            rl, rh = boot[name][bkey]["roc_auc"]
            cl, ch = boot[name][bkey]["recall"]
            tag = "B (honest)" if v == "B" else "A (+leak)"
            L.append(
                f"| {name} | {tag} | {_fmt(p['roc_auc'])} | {_ci(rl, rh)} | "
                f"{_fmt(p['pr_auc'])} | {_fmt(p['recall'])} | {_ci(cl, ch)} | "
                f"{_fmt(p['challenge_rate'])} |"
            )

    L.append("\n## A - B deltas (paired bootstrap)\n")
    L.append("| model | ΔROC-AUC (A-B) | ΔROC-AUC 95% CI | Δrecall@1%FPR (A-B) | Δrecall 95% CI |")
    L.append("|---|---|---|---|---|")
    for name in models:
        d_roc = point[name]["A"]["roc_auc"] - point[name]["B"]["roc_auc"]
        d_rec = point[name]["A"]["recall"] - point[name]["B"]["recall"]
        drl, drh = boot[name]["delta"]["roc_auc"]
        dcl, dch = boot[name]["delta"]["recall"]
        L.append(
            f"| {name} | {_fmt(d_roc)} | {_ci(drl, drh)} | "
            f"{_fmt(d_rec)} | {_ci(dcl, dch)} |"
        )

    L.append("\n## Reference: `is_attack_ip` alone (pure-leakage ceiling)\n")
    L.append("| ROC-AUC | PR-AUC | recall@1%FPR | achieved FPR | challenge rate |")
    L.append("|---|---|---|---|---|")
    L.append(
        f"| {_fmt(flag_pt['roc_auc'])} | {_fmt(flag_pt['pr_auc'])} | "
        f"{_fmt(flag_pt['recall'])} | {_fmt(flag_pt['achieved_fpr'])} | "
        f"{_fmt(flag_pt['challenge_rate'])} |"
    )

    L.append(
        "\n## How to read this\n"
        "- If a model's **ΔROC-AUC / Δrecall CI sits well above 0**, `is_attack_ip` is "
        "materially lifting it — i.e. the model is leaning on the attack-derived flag, "
        "not behaviour. The thesis then reports **Variant B**.\n"
        "- If the delta CI **straddles 0**, the flag adds no honest signal beyond "
        "behaviour at this operating point (with only "
        f"{n_pos_test} positives the test is under-powered — say so).\n"
        "- The `is_attack_ip`-alone row bounds how much pure leakage is available: it is "
        "an oracle-ish signal in this synthesised set and would not exist at serve time "
        "as a free label, which is exactly why it is barred from the honest model."
    )

    report = "\n".join(L) + "\n"
    with open(args.report, "w") as fh:
        fh.write(report)
    print(report)
    print(f"[report] written to {args.report}")


if __name__ == "__main__":
    main()
