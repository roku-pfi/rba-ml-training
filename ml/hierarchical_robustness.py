"""Robustness of the rooted hierarchical Freeman (finding 2026-10-23-hierarchical).

Usage (from rba-ml-training/, venv active):  python -m ml.hierarchical_robustness

Labels-free weights, another held-out slice, uniform weights, recall by history
depth, and a users bootstrap of recall@1%FPR (rooted - flat).
"""
import numpy as np, pandas as pd
from rba_features import schema
from ml.metrics import evaluate, recall_at_fpr
from ml.models.freeman import FreemanScorer
from ml.models.hierarchical import HIERARCHIES, fit_lambdas, global_tables, level_matrix, logrisk, prep, rooted

LABEL = schema.LABEL
full = pd.read_parquet("data/subset/logins.parquet").reset_index(drop=True)
last_pos = full.loc[full[LABEL], "login_timestamp"].max()
df = full[full["login_timestamp"] <= last_pos].reset_index(drop=True)
cut = df["login_timestamp"].quantile(0.70)
df["split"] = np.where(df["login_timestamp"] < cut, "train", "test")
order = df.sort_values(["user_id", "login_timestamp"], kind="mergesort")
df["depth"] = order.groupby("user_id", sort=False).cumcount().sort_index()
y = df[LABEL].to_numpy().astype(int); test = (df["split"] == "test").to_numpy(); depth = df["depth"].to_numpy()
d = prep(df)
train = d[d["split"] == "train"].reset_index(drop=True)
yt = train[LABEL].to_numpy().astype(int)
lv = level_matrix(d, global_tables(d[d["split"] == "train"]))

def lambdas_for(fit_frac, use_labels):
    fc = train["login_timestamp"].quantile(fit_frac)
    early = (train["login_timestamp"] < fc).to_numpy()
    held = ~early & ((yt == 0) if use_labels else True)
    lvf = level_matrix(train, global_tables(train[early]))
    lam = {}
    for name in HIERARCHIES:
        g, u = lvf[name]; warm = held & (u.sum(axis=1) > 0)
        lg = fit_lambdas(g[held]); lam[name] = (lg, fit_lambdas(rooted(u, g @ lg)[warm]))
    return lam

def report(name, s):
    r = evaluate(y[test], s[test], depth[test], 0.01); op = r["operating_point"]
    print(f"{name:44s} roc={r['roc_auc']:.3f} recall@1%={op.recall:.3f} ({int(round(op.recall*38))}/38) "
          f"tpr@10%={recall_at_fpr(y[test], s[test], 0.10).recall:.3f}")
    return op.threshold

base = lambdas_for(0.60, True)
thr = report("rooted, λ on held-out legit (as reported)", logrisk(lv, base, root_at_global=True))
report("rooted, λ on all held-out logins (no labels)", logrisk(lv, lambdas_for(0.60, False), root_at_global=True))
report("rooted, λ held-out = last 20% of train", logrisk(lv, lambdas_for(0.80, True), root_at_global=True))
uni = {n: (np.full(len(c) + 1, 1 / (len(c) + 1)),) * 2 for n, c in HIERARCHIES.items()}
report("rooted, uniform λ (nothing learned)", logrisk(lv, uni, root_at_global=True))

s = logrisk(lv, base, root_at_global=True)
flat = FreemanScorer(beta=5.0).fit(df[df["split"] == "train"]).score_frame(df)
print("\nby history depth at each model's own 1%-FPR threshold (takeovers caught / takeovers):")
for name, sc in (("flat", flat), ("rooted", s)):
    r = evaluate(y[test], sc[test], depth[test], 0.01)
    print(f"  {name:7s}", r["by_depth"] if not isinstance(r["by_depth"], pd.DataFrame) else r["by_depth"].to_string())

# Bootstrap over users: recall@1%FPR difference rooted - flat.
rng = np.random.default_rng(6)
ut = df.loc[test, "user_id"].to_numpy(); yy = y[test]; a = s[test]; b = flat[test]
uniq = np.unique(ut); idx = {u: np.where(ut == u)[0] for u in uniq}
diffs = []
for _ in range(1000):
    pick = np.concatenate([idx[u] for u in rng.choice(uniq, len(uniq))])
    if yy[pick].sum() == 0: continue
    ra = recall_at_fpr(yy[pick], a[pick], 0.01).recall; rb = recall_at_fpr(yy[pick], b[pick], 0.01).recall
    diffs.append(ra - rb)
lo, hi = np.percentile(diffs, [2.5, 97.5])
print(f"\nrecall@1%FPR rooted − flat: {np.mean(diffs):+.3f}, 95% CI (users bootstrap) [{lo:+.3f}, {hi:+.3f}]")
