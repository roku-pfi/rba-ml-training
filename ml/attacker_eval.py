"""Flat vs rooted hierarchical Freeman against Wiefling et al.'s attacker models.

Wiefling, Jørgensen, Thunem, Lo Iacono, "Pump Up Password Security!", ACM TOPS
2022, Sect. 5 (the paper behind this dataset). All attackers have the password:

- naive (random):     a random attack IP from any country, a random user agent;
- naive (resourced):  the most frequent attack IPs and user agents;
- VPN:                an attack IP in the victim's main country, a popular legitimate UA;
- targeted:           a frequent attack IP in the victim's country, the victim's own UA;
- very targeted:      the labelled account takeovers (ml.hierarchical_eval).

Attack IPs are the dataset's `is_attack_ip` failed attempts, as in the paper.
Each probe is a test-period legitimate login's moment, replayed with the
attacker's features against the victim's history at that time; probes are
scored and never learned. Legitimate set: test-period legitimate logins of users
with history (a first login scores 0 under both models).

Usage (from rba-ml-training/, venv active):

    python -m ml.attacker_eval --report reports/attackers.md
"""

from __future__ import annotations

import argparse
import math
import os
from collections import Counter

import numpy as np
import pandas as pd
from rba_features import schema
from sklearn.metrics import roc_auc_score

from ml.metrics import recall_at_fpr
from ml.models.freeman import FreemanScorer
from ml.models.hierarchical import HIERARCHIES, chain_from_stats, chain_logrisk, chain_stats, fit_betas, fit_lambdas, global_tables, level_matrix, logrisk, prep, rooted

LABEL = schema.LABEL
AGENT = ["device_type", "os", "browser"]
NET = ["ip_address", "asn", "country"]


def flat_scores(sc: FreemanScorer, d: pd.DataFrame, learn: np.ndarray) -> np.ndarray:
    """freeman-0.3.0's logrisk per row, past-only per user; probes are not learned."""
    feats = sc.features
    cols = {f: d[f].astype(str).to_numpy() for f in feats}
    users = d["user_id"].to_numpy()
    order = np.argsort(d["login_timestamp"].to_numpy(), kind="mergesort")
    counts: dict[str, dict[str, Counter]] = {}
    totals: dict[str, int] = {}
    out = np.zeros(len(d))
    for i in order:
        u = users[i]
        uc = counts.setdefault(u, {f: Counter() for f in feats})
        t = totals.get(u, 0)
        lr = 0.0
        for f in feats:
            v = cols[f][i]
            pg = sc._p_global(f, v)
            lr += math.log(pg) - math.log((uc[f][v] + sc.beta * pg) / (t + sc.beta))
        out[i] = lr
        if learn[i]:
            for f in feats:
                uc[f][cols[f][i]] += 1
            totals[u] = t + 1
    return out


def victim_state(d: pd.DataFrame) -> pd.DataFrame:
    """Per row: the user's main country and most-used agent before this login."""
    order = np.argsort(d["login_timestamp"].to_numpy(), kind="mergesort")
    users = d["user_id"].to_numpy()
    country = d["country"].to_numpy()
    agent = list(zip(*(d[c].to_numpy() for c in AGENT)))
    cs: dict[str, Counter] = {}
    ag: dict[str, Counter] = {}
    main_c = np.empty(len(d), dtype=object)
    main_a = np.empty(len(d), dtype=object)
    n_prior = np.zeros(len(d), dtype=int)
    for i in order:
        u = users[i]
        c, a = cs.setdefault(u, Counter()), ag.setdefault(u, Counter())
        n_prior[i] = sum(c.values())
        main_c[i] = c.most_common(1)[0][0] if c else None
        main_a[i] = a.most_common(1)[0][0] if a else None
        c[country[i]] += 1
        a[agent[i]] += 1
    return pd.DataFrame({"main_country": main_c, "main_agent": main_a, "n_prior": n_prior}, index=d.index)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/subset/logins.parquet")
    parser.add_argument("--report", default="reports/attackers.md")
    parser.add_argument("--probes", type=int, default=4000, help="per attacker model")
    parser.add_argument("--seed", type=int, default=6)
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)

    full = pd.read_parquet(args.data).reset_index(drop=True)
    last_pos = full.loc[full[LABEL], "login_timestamp"].max()
    df = full[full["login_timestamp"] <= last_pos].reset_index(drop=True)
    cut = df["login_timestamp"].quantile(0.70)
    df["split"] = np.where(df["login_timestamp"] < cut, "train", "test")
    d = prep(df)
    d["hour"] = pd.to_datetime(d["login_timestamp"]).dt.hour
    state = victim_state(d)

    # Attack IPs: failed attempts from known attack IPs (the paper's source).
    attack = d[d["is_attack_ip"].astype(bool) & ~d["login_successful"].astype(bool)]
    if len(attack) < 1000:
        attack = d[d["is_attack_ip"].astype(bool)]
    ip_freq = attack.groupby(NET).size().rename("n").reset_index()
    legit_agents = d.loc[d[LABEL] == 0].groupby(AGENT).size().rename("n").reset_index()
    all_agents = d.groupby(AGENT).size().rename("n").reset_index()

    def pick(table: pd.DataFrame, weighted: bool, top: int | None = None) -> pd.Series:
        t = table.nlargest(top, "n") if top else table
        w = (t["n"] / t["n"].sum()).to_numpy() if weighted else None
        return t.iloc[rng.choice(len(t), p=w)]

    # Moments to attack: test-period legitimate logins of users with history.
    y = d[LABEL].to_numpy().astype(int)
    moments = d.index[(d["split"] == "test") & (y == 0) & (state["n_prior"] > 0)].to_numpy()
    by_country = {c: g for c, g in ip_freq.groupby("country")}
    countries = list(by_country)

    probes = []
    for kind in ("naive (random)", "naive (resourced)", "VPN", "targeted"):
        made = 0
        for i in rng.permutation(moments):
            if made == args.probes:
                break
            row = d.loc[i]
            if kind == "naive (random)":
                ip = pick(by_country[countries[rng.integers(len(countries))]], weighted=False)
                ua = pick(all_agents, weighted=False)
            elif kind == "naive (resourced)":
                ip, ua = pick(ip_freq, True, top=100), pick(all_agents, True, top=20)
            else:
                pool = by_country.get(state.at[i, "main_country"])
                if pool is None:
                    continue
                if kind == "VPN":
                    ip, ua = pick(pool, True), pick(legit_agents, True, top=10)
                else:
                    ip = pick(pool, True, top=10)
                    ua = dict(zip(AGENT, state.at[i, "main_agent"]))
            probe = row.copy()
            for c in NET:
                probe[c] = ip[c]
            for c in AGENT:
                probe[c] = ua[c]
            probe["login_timestamp"] = row["login_timestamp"] + pd.Timedelta(milliseconds=1)
            probe["kind"] = kind
            probes.append(probe)
            made += 1

    p = pd.DataFrame(probes)
    d["kind"] = np.where(y == 1, "takeover", "legit")
    allrows = pd.concat([d, p], ignore_index=True)
    learn = (allrows.index < len(d))
    train = d[d["split"] == "train"].reset_index(drop=True)

    flat_sc = FreemanScorer(beta=5.0).fit(df[df["split"] == "train"])
    flat = flat_scores(flat_sc, allrows, learn)

    # Interpolation weights as in ml.hierarchical_eval, here without labels.
    early = (train["login_timestamp"] < train["login_timestamp"].quantile(0.60)).to_numpy()
    lvf = level_matrix(train, global_tables(train[early]))
    lambdas = {}
    for name in HIERARCHIES:
        g, u = lvf[name]
        warm = ~early & (u.sum(axis=1) > 0)
        lam_g = fit_lambdas(g[~early])
        lambdas[name] = (lam_g, fit_lambdas(rooted(u, g @ lam_g)[warm]))
    lv = level_matrix(allrows, global_tables(train), learn)
    hier = logrisk(lv, lambdas, root_at_global=True)
    uniform = {n: (np.full(len(c) + 1, 1 / (len(c) + 1)),) * 2 for n, c in HIERARCHIES.items()}
    hier_u = logrisk(lv, uniform, root_at_global=True)

    is_legit = np.zeros(len(allrows), dtype=bool)
    is_legit[moments] = True  # the same legitimate logins the probes were drawn from
    # The flat model's hour term, for the "+ hour" candidate.
    hour_c = np.zeros(len(allrows))
    hours = allrows["hour"].astype(str).to_numpy()
    users = allrows["user_id"].to_numpy()
    seen_h: dict[str, Counter] = {}
    for i in np.argsort(allrows["login_timestamp"].to_numpy(), kind="mergesort"):
        c = seen_h.setdefault(users[i], Counter())
        t = sum(c.values())
        pg = flat_sc._p_global("hour", hours[i])
        hour_c[i] = math.log(pg) - math.log((c[hours[i]] + flat_sc.beta * pg) / (t + flat_sc.beta))
        if learn[i]:
            c[hours[i]] += 1
    agent_sc = FreemanScorer(beta=5.0, features=("device_type", "os", "browser")).fit(df[df["split"] == "train"])
    hybrid = logrisk(lv, uniform, ["network"], root_at_global=True) + flat_scores(agent_sc, allrows, learn)
    models = {"flat freeman-0.3.0": flat, "hybrid: rooted network + flat agent": hybrid, "hierarchical, rooted (λ fitted)": hier,
              "hierarchical, rooted (uniform λ)": hier_u,
              "hierarchical, rooted (uniform λ) + hour": hier_u + hour_c,
              "chain-rule Freeman, β fitted": chain_from_stats(
                  chain_stats(allrows, train, learn=learn), fit_betas(chain_stats(train, train[early]), ~early))}
    lines = ["# Flat vs rooted hierarchical Freeman against Wiefling et al.'s attackers", "",
             f"Legitimate: {is_legit.sum():,} test-period logins of users with history. "
             f"Probes per attacker model: up to {args.probes:,}. Attack IP pool: {len(ip_freq):,} "
             f"(IP, ASN, country) from {len(attack):,} attack-IP attempts.", "",
             "Columns: ROC-AUC · legitimate logins challenged when blocking 99% of the attacker "
             "(the paper's view) · attackers caught at 1% legitimate FPR.", "",
             "| attacker | n | " + " | ".join(models) + " |", "|---|---|" + "---|" * len(models)]
    kinds = ["naive (random)", "naive (resourced)", "VPN", "targeted", "takeover"]
    for kind in kinds:
        att = (allrows["kind"] == kind).to_numpy().copy()
        if kind == "takeover":
            att &= (allrows["split"] == "test").to_numpy()
        cells = []
        for s in models.values():
            yy = np.r_[np.zeros(is_legit.sum()), np.ones(att.sum())]
            ss = np.r_[s[is_legit], s[att]]
            auc = roc_auc_score(yy, ss)
            thr99 = np.quantile(s[att], 0.01)
            fpr99 = float((s[is_legit] >= thr99).mean())
            r1 = recall_at_fpr(yy, ss, 0.01).recall
            cells.append(f"{auc:.3f} · {fpr99:.1%} · {r1:.2f}")
        name = "very targeted (labelled takeovers)" if kind == "takeover" else kind
        lines.append(f"| {name} | {int(att.sum())} | " + " | ".join(cells) + " |")

    lines += ["", "Attackers caught at a given share of legitimate logins challenged "
              "(5% / 10% / 20% / 30%, the policies' operating range):", "",
              "| attacker | " + " | ".join(models) + " |", "|---|" + "---|" * len(models)]
    for kind in kinds:
        att = (allrows["kind"] == kind).to_numpy().copy()
        if kind == "takeover":
            att &= (allrows["split"] == "test").to_numpy()
        cells = []
        for s in models.values():
            legit_s = s[is_legit]
            cells.append(" / ".join(f"{(s[att] > np.quantile(legit_s, 1 - f)).mean():.2f}" for f in (0.05, 0.10, 0.20, 0.30)))
        name = "very targeted (labelled takeovers)" if kind == "takeover" else kind
        lines.append(f"| {name} | " + " | ".join(cells) + " |")

    os.makedirs(os.path.dirname(args.report) or ".", exist_ok=True)
    with open(args.report, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
