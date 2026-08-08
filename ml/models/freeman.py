"""Freeman et al. (2016) likelihood-ratio risk scorer — the PRIMARY RBA model.

Reference: Freeman, Jain, Duermuth, Biggio, Giacinto, "Who Are You? A Statistical
Approach to Measuring User Authenticity" (NDSS 2016). It scores a login by the
likelihood ratio "attacker vs this legitimate user", factorised over features under a
naive-Bayes assumption:

    logrisk(x) = sum_k  [ log p_global(x_k) - log p_user(x_k) ]

- p_global(x_k): how common value x_k is in the population (the attacker model,
  approximated by the global login distribution). Fit on the TRAIN split only.
- p_user(x_k): how common x_k is for THIS user, estimated from their history and
  smoothed toward the global distribution (Dirichlet prior) so sparse/unseen values
  fall back to the population. Built past-only per user (no leakage).

High logrisk = the user is presenting values common in the population but rare/unseen
for themselves → likely takeover. It needs almost no attack labels, which is why it is
the primary scorer given only 141 positives (see docs ADR-0004).

NOTE (productionisation): per-user counts are maintained here for the offline
feasibility study. Serving this online means carrying per-value counts in the shared
`rba_features.ProfileState` (today it stores seen-sets, not counts) — tracked as a
Phase 3 follow-up so train/serve parity extends to the model's state too.
"""

from __future__ import annotations

import math
from collections import Counter

import numpy as np
import pandas as pd

# Categorical attributes scored by Freeman. Region/city excluded (~42% missing and
# synthetic geo); RTT excluded (~94% missing). `hour` is derived from the timestamp.
FREEMAN_FEATURES: tuple[str, ...] = (
    "ip_address",
    "asn",
    "country",
    "device_type",
    "os",
    "browser",
    "hour",
)


class FreemanScorer:
    """Smoothed naive-Bayes likelihood-ratio scorer.

    Parameters
    ----------
    alpha : additive (Laplace) smoothing for the global distribution.
    beta  : Dirichlet prior strength pulling a user's distribution toward global
            (higher = trust the user's own history less until they have more of it).
    """

    def __init__(self, alpha: float = 0.5, beta: float = 10.0,
                 features: tuple[str, ...] = FREEMAN_FEATURES) -> None:
        self.alpha = alpha
        self.beta = beta
        self.features = list(features)
        self.global_counts: dict[str, Counter] = {}
        self.global_total: dict[str, int] = {}
        self.vocab: dict[str, int] = {}

    @staticmethod
    def _prep(df: pd.DataFrame) -> pd.DataFrame:
        d = df.copy()
        d["hour"] = pd.to_datetime(d["login_timestamp"]).dt.hour
        return d

    def fit(self, train_df: pd.DataFrame) -> "FreemanScorer":
        d = self._prep(train_df)
        for f in self.features:
            counts = Counter(d[f].astype(str).tolist())
            self.global_counts[f] = counts
            self.global_total[f] = int(sum(counts.values()))
            self.vocab[f] = len(counts)
        return self

    def _p_global(self, f: str, v: str) -> float:
        c = self.global_counts[f].get(v, 0)
        # +1 in the denominator vocab reserves mass for values unseen in training.
        return (c + self.alpha) / (self.global_total[f] + self.alpha * (self.vocab[f] + 1))

    def score_frame(self, df: pd.DataFrame) -> np.ndarray:
        """Return per-row logrisk aligned to `df` order (past-only per user)."""
        d = self._prep(df).reset_index(drop=True)
        d["_pos"] = np.arange(len(d))
        d = d.sort_values(["user_id", "login_timestamp"], kind="mergesort")

        out = np.zeros(len(d), dtype=float)
        feats = self.features
        pos_arr = d["_pos"].to_numpy()
        # Materialise columns once for speed.
        cols = {f: d[f].astype(str).to_numpy() for f in feats}
        user_arr = d["user_id"].to_numpy()

        cur_user = None
        counts: dict[str, Counter] = {}
        totals: dict[str, int] = {}
        for i in range(len(d)):
            u = user_arr[i]
            if u != cur_user:
                cur_user = u
                counts = {f: Counter() for f in feats}
                totals = {f: 0 for f in feats}
            logrisk = 0.0
            for f in feats:
                v = cols[f][i]
                pg = self._p_global(f, v)
                pu = (counts[f].get(v, 0) + self.beta * pg) / (totals[f] + self.beta)
                logrisk += math.log(pg) - math.log(pu)
                counts[f][v] += 1
                totals[f] += 1
            out[pos_arr[i]] = logrisk
        return out
