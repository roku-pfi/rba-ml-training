"""Paper-faithful hierarchical Freeman vs our flat Freeman, on the Step 5 split.

Usage (from rba-ml-training/, venv active):

    python -m ml.hierarchical_eval --report reports/hierarchical.md

Protocol as Step 5 / calibration: label window, chronological 70/30 split,
global tables from the training period, every user scored past-only. The
interpolation weights are learned as in the paper (Alg. 1): on legitimate logins
from a held-out slice of the training period (its last 40%), with global tables
built from the first 60%; test scoring then uses tables from the whole training
period with those weights.
"""

from __future__ import annotations

import argparse
import math
import os

import numpy as np
import pandas as pd
from rba_features import schema

from ml.metrics import evaluate, recall_at_fpr
from ml.models.freeman import FreemanScorer
from ml.models.hierarchical import HIERARCHIES, chain_from_stats, chain_logrisk, chain_stats, fit_betas, fit_lambdas, global_tables, level_matrix, logrisk, prep, rooted

LABEL = schema.LABEL


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/subset/logins.parquet")
    parser.add_argument("--report", default="reports/hierarchical.md")
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
    depth = df["depth"].to_numpy()
    d = prep(df)

    # --- interpolation weights on a held-out slice of the training period
    train = d[d["split"] == "train"].reset_index(drop=True)
    y_train = train[LABEL].to_numpy().astype(int)
    fit_cut = train["login_timestamp"].quantile(0.60)
    early = (train["login_timestamp"] < fit_cut).to_numpy()
    held = ~early & (y_train == 0)
    lv_fit = level_matrix(train, global_tables(train[early]))
    lambdas, lambdas_rooted = {}, {}
    for name in HIERARCHIES:
        g, u = lv_fit[name]
        warm = held & (u.sum(axis=1) > 0)
        lam_g = fit_lambdas(g[held])
        lambdas[name] = (lam_g, fit_lambdas(u[warm]))
        lambdas_rooted[name] = (lam_g, fit_lambdas(rooted(u, g @ lam_g)[warm]))

    # --- chain-rule β per level, by likelihood on the same held-out slice (no labels)
    betas = fit_betas(chain_stats(train, train[early]), ~early)

    # --- test scoring
    lv = level_matrix(d, global_tables(d[d["split"] == "train"]))
    hier = logrisk(lv, lambdas)

    flat_sc = FreemanScorer(beta=5.0).fit(df[df["split"] == "train"])
    flat_contrib = flat_sc.contributions_frame(df)
    flat = flat_contrib.sum(axis=1)
    hour = flat_contrib[:, flat_sc.features.index("hour")]

    # p(u|A) / p(u|L) of Eq. 7: users equally likely to be attacked; p(u|L) the
    # user's share of prior logins (smoothed), past-only.
    ts_order = np.argsort(df["login_timestamp"].to_numpy(), kind="mergesort")
    users = df["user_id"].to_numpy()
    n_users = df["user_id"].nunique()
    seen: dict[str, int] = {}
    user_term = np.zeros(len(df))
    for n, i in enumerate(ts_order):
        c = seen.get(users[i], 0)
        user_term[i] = math.log(1 / n_users) - math.log((c + 1) / (n + n_users))
        seen[users[i]] = c + 1

    variants = {
        "flat freeman-0.3.0 (7 features)": flat,
        "hierarchical: network only": logrisk(lv, lambdas, ["network"]),
        "hierarchical: agent only": logrisk(lv, lambdas, ["agent"]),
        "hierarchical: network + agent (paper)": hier,
        "hierarchical + hour": hier + hour,
        "hierarchical + p(u) term (Eq. 7)": hier + user_term,
        "hierarchical, user rooted at global": logrisk(lv, lambdas_rooted, root_at_global=True),
        "hierarchical, rooted, + hour": logrisk(lv, lambdas_rooted, root_at_global=True) + hour,
        "chain-rule Freeman (Dirichlet at every level)": chain_logrisk(d, d[d["split"] == "train"]),
        "hybrid: rooted network + flat agent": logrisk(lv, {n: (np.full(4, .25), np.full(4, .25)) for n in HIERARCHIES}, ["network"], root_at_global=True)
            + FreemanScorer(beta=5.0, features=("device_type", "os", "browser")).fit(df[df["split"] == "train"]).score_frame(df),
        "chain-rule Freeman, β per level fitted": chain_from_stats(chain_stats(d, d[d["split"] == "train"]), betas),
    }

    lines = ["# Hierarchical Freeman (paper, Sect. II-C) vs flat Freeman", "",
             f"Test split: {int(test.sum()):,} logins, {int(y[test].sum())} takeovers. "
             "Interpolation weights (world → leaf):", ""]
    for name, (lg, lu) in lambdas.items():
        lines.append(f"- {name}: global {np.round(lg, 3).tolist()} · user {np.round(lu, 3).tolist()}"
                     f" · user rooted {np.round(lambdas_rooted[name][1], 3).tolist()}")
    g_test, _ = lv["network"]
    unseen = (g_test[:, -1] == 0)
    lines.append(f"- takeover IPs never seen in the training period: {unseen[test & (y == 1)].mean():.0%}"
                 f" (legitimate test logins: {unseen[test & (y == 0)].mean():.0%})")
    lines.append(f"- chain β per level (fitted): " + "; ".join(f"{n} {np.round(b, 3).tolist()}" for n, b in betas.items()))
    lines += ["", "| variant | ROC-AUC | recall@1%FPR | TPR@10%FPR (paper's metric) |", "|---|---|---|---|"]
    for name, s in variants.items():
        r = evaluate(y[test], s[test], depth[test], 0.01)
        op = r["operating_point"]
        op10 = recall_at_fpr(y[test], s[test], 0.10)
        hits = int(round(op.recall * y[test].sum()))
        lines.append(f"| {name} | {r['roc_auc']:.3f} | {op.recall:.3f} ({hits}/{int(y[test].sum())}) | {op10.recall:.3f} |")

    os.makedirs(os.path.dirname(args.report) or ".", exist_ok=True)
    with open(args.report, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
