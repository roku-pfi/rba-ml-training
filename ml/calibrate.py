"""Freeman calibration: fix the primary scorer's strict-FPR operating point.

Step 5 diagnosis: the raw Freeman score ranks well (ROC-AUC ~0.76) but its
equally-weighted sum is dominated by the near-unique IP term, so it over-flags
first-time-IP legit logins and only recovers 3/38 takeovers at a 1% FPR. Note that
plain probability calibration (Platt/isotonic) is monotone and would NOT move
recall@FPR — the fix has to change the *ranking*. Two levers, both evaluated here on
the identical chronological split from Step 5:

    freeman            baseline (all 7 features, equal weight)         [label-free]
    freeman_noip       drop the near-unique IP feature                 [label-free]
    freeman_weighted   logistic per-feature weights on the LLRs        [uses labels]
    freeman_weighted_noip  weighted, IP dropped                        [uses labels]

Deltas vs the baseline carry a paired bootstrap CI (only 38 test positives). The
unsupervised variants keep Freeman's label-light appeal; the weighted ones are a
calibrated variant (the raw FreemanScorer stays the label-free primary).

Usage:
    python -m ml.calibrate
"""

from __future__ import annotations

import argparse
import os
import pickle
import time

import numpy as np
import pandas as pd

from rba_features import schema

from ml.leakage import paired_bootstrap
from ml.metrics import evaluate
from ml.models.freeman import FREEMAN_FEATURES, FreemanScorer, WeightedFreemanScorer

LABEL = schema.LABEL
NOIP_FEATURES = tuple(f for f in FREEMAN_FEATURES if f != "ip_address")


def _fmt(x: float) -> str:
    return "nan" if x != x else f"{x:.3f}"


def _ci(pair: tuple[float, float]) -> str:
    return f"[{_fmt(pair[0])}, {_fmt(pair[1])}]"


def _prep(df: pd.DataFrame, train_frac: float):
    """Restrict to the label-covered window and add split/depth (as in Step 5)."""
    last_pos = df.loc[df[LABEL], "login_timestamp"].max()
    w = df[df["login_timestamp"] <= last_pos].reset_index(drop=True).copy()
    cut = w["login_timestamp"].quantile(train_frac)
    w["split"] = np.where(w["login_timestamp"] <= cut, "train", "test")
    order = w.sort_values(["user_id", "login_timestamp"], kind="mergesort")
    w["depth"] = order.groupby("user_id", sort=False).cumcount().sort_index()
    return w, cut, last_pos


def _score_variant(name: str, w: pd.DataFrame):
    """Fit a variant on the train split and score the whole frame. Returns
    (scores, size_kb, latency_us, weights_or_None)."""
    train = w[w["split"] == "train"]
    y_train = train[LABEL].to_numpy().astype(int)

    # Baselines pinned to beta=10 (the Step 5 default) so this table is a stable
    # "before" picture; the smoothing sweep below explores the "after".
    if name == "freeman":
        model = FreemanScorer(beta=10).fit(train)
    elif name == "freeman_noip":
        model = FreemanScorer(features=NOIP_FEATURES, beta=10).fit(train)
    elif name == "freeman_weighted":
        model = WeightedFreemanScorer(FreemanScorer(beta=10).fit(train)).fit(train, y_train)
    elif name == "freeman_weighted_noip":
        base = FreemanScorer(features=NOIP_FEATURES, beta=10).fit(train)
        model = WeightedFreemanScorer(base).fit(train, y_train)
    else:
        raise ValueError(name)

    t0 = time.time()
    scores = model.score_frame(w)
    lat_us = (time.time() - t0) / max(len(w), 1) * 1e6
    blob = pickle.dumps(model)
    weights = getattr(model, "weights", None)
    return scores, len(blob) / 1024, lat_us, weights, model


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeman calibration experiment.")
    parser.add_argument("--data", default="data/subset/logins.parquet")
    parser.add_argument("--target-fpr", type=float, default=0.01)
    parser.add_argument("--train-frac", type=float, default=0.70)
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--betas", type=float, nargs="+", default=[2, 5, 10, 20, 50])
    parser.add_argument("--out", default="artifacts/freeman_calibration")
    parser.add_argument("--report", default="reports/freeman_calibration/metrics.md")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    os.makedirs(os.path.dirname(args.report) or ".", exist_ok=True)

    df = pd.read_parquet(args.data).reset_index(drop=True)
    w, cut, last_pos = _prep(df, args.train_frac)
    te = (w["split"] == "test").to_numpy()
    y = w[LABEL].to_numpy().astype(int)
    depth = w["depth"].to_numpy()
    yte, depth_te = y[te], depth[te]
    fpr = args.target_fpr

    variants = ["freeman", "freeman_noip", "freeman_weighted", "freeman_weighted_noip"]
    results, scores_te, weights = {}, {}, {}
    for name in variants:
        scores, size_kb, lat_us, wts, model = _score_variant(name, w)
        res = evaluate(yte, scores[te], depth_te, fpr)
        results[name] = {"res": res, "size_kb": size_kb, "lat_us": lat_us}
        scores_te[name] = scores[te]
        weights[name] = wts
        with open(os.path.join(args.out, f"{name}.pkl"), "wb") as fh:
            pickle.dump(model, fh)

    # ---- Report ----
    L: list[str] = ["# Freeman calibration — fixing the strict-FPR operating point"]
    L.append(
        f"\nLabel-covered window → chronological split at {cut} (train "
        f"{args.train_frac:.0%} / test). Test set: **{int(te.sum()):,} logins, "
        f"{int(yte.sum())} positives**. Same split as Step 5. CIs are 95% percentile "
        f"paired bootstraps ({args.n_boot:,} resamples) vs the `freeman` baseline."
    )

    L.append("\n## Ranking + operating point (test set)\n")
    L.append("| variant | labels? | ROC-AUC | PR-AUC | recall@1%FPR | achieved FPR | challenge | size | latency |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    uses_labels = {
        "freeman": "no", "freeman_noip": "no",
        "freeman_weighted": "yes", "freeman_weighted_noip": "yes",
    }
    for name in variants:
        r = results[name]["res"]
        op = r["operating_point"]
        L.append(
            f"| {name} | {uses_labels[name]} | {_fmt(r['roc_auc'])} | {_fmt(r['pr_auc'])} | "
            f"{_fmt(op.recall)} | {_fmt(op.achieved_fpr)} | {_fmt(op.challenge_rate)} | "
            f"{results[name]['size_kb']:.1f} KB | {results[name]['lat_us']:.1f} µs |"
        )

    L.append("\n## Improvement vs the `freeman` baseline (paired bootstrap)\n")
    L.append("| variant | ΔROC-AUC | ΔROC-AUC 95% CI | Δrecall@1%FPR | Δrecall 95% CI |")
    L.append("|---|---|---|---|---|")
    base_roc = results["freeman"]["res"]["roc_auc"]
    base_rec = results["freeman"]["res"]["operating_point"].recall
    for name in variants[1:]:
        b = paired_bootstrap(yte, scores_te[name], scores_te["freeman"], fpr, n_boot=args.n_boot)
        d_roc = results[name]["res"]["roc_auc"] - base_roc
        d_rec = results[name]["res"]["operating_point"].recall - base_rec
        L.append(
            f"| {name} | {_fmt(d_roc)} | {_ci(b['delta']['roc_auc'])} | "
            f"{_fmt(d_rec)} | {_ci(b['delta']['recall'])} |"
        )

    L.append("\n## Recall by history depth (at each variant's 1%-FPR threshold)\n")
    L.append("| bucket | n | positives | " + " | ".join(variants) + " |")
    L.append("|---|---|---|" + "---|" * len(variants))
    depth_rows = {name: results[name]["res"]["by_depth"] for name in variants}
    for i, b0 in enumerate(depth_rows["freeman"]):
        cells = " | ".join(_fmt(depth_rows[name][i]["recall"]) for name in variants)
        L.append(f"| {b0['bucket']} | {b0['n']:,} | {b0['positives']} | {cells} |")

    # ---- Smoothing (beta) sweep on the raw all-feature scorer ----
    L.append(
        "\n## Smoothing sweep (raw `freeman`, all features)\n"
        "`beta` = Dirichlet prior strength pulling a user toward the global "
        "distribution (higher = trust the user's own short history less).\n"
    )
    L.append("| beta | ROC-AUC | recall@1%FPR | challenge |")
    L.append("|---|---|---|---|")
    train = w[w["split"] == "train"]
    for b in args.betas:
        s = FreemanScorer(beta=b).fit(train).score_frame(w)
        r = evaluate(yte, s[te], depth_te, fpr)
        op = r["operating_point"]
        star = " (default)" if b == 10 else ""
        L.append(f"| {b:g}{star} | {_fmt(r['roc_auc'])} | {_fmt(op.recall)} | {_fmt(op.challenge_rate)} |")

    L.append("\n## Learned per-feature weights (calibrated variants)\n")
    for name in ("freeman_weighted", "freeman_weighted_noip"):
        if weights[name]:
            ordered = sorted(weights[name].items(), key=lambda kv: kv[1], reverse=True)
            pretty = ", ".join(f"`{k}` {v:+.2f}" for k, v in ordered)
            L.append(f"- **{name}**: {pretty}")

    L.append(
        "\n## Reading it\n"
        "- Positive weights push toward 'risky'; a near-zero/negative weight on "
        "`ip_address` confirms the calibration is *down-weighting the noisy IP term* "
        "that hurt the raw score.\n"
        "- The label-free `freeman_noip` shows how much of the fix is free (no labels); "
        "the weighted variants show the ceiling when the few labels are used to tune "
        "per-feature weights. The raw `freeman` remains the label-free primary; a "
        "weighted variant is the calibrated option when labels are available.\n"
        "- 38 positives → wide CIs; treat a ΔCI that clears 0 as 'real but noisy'."
    )

    report = "\n".join(L) + "\n"
    with open(args.report, "w") as fh:
        fh.write(report)
    print(report)
    print(f"[report] written to {args.report}")


if __name__ == "__main__":
    main()
