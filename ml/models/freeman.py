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

NOTE (productionisation): per-user counts for online scoring live in
`rba_features.ProfileState.freeman_counts` / `freeman_totals` (ADR-0009). Offline
`contributions_frame` still rebuilds Counters while scanning; online uses the
materialised profile + `score_event` / the decision-service JSON artifact.
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
            Default 5.0 was chosen by the calibration sweep (ml/calibrate.py): it
            improves both ROC-AUC and recall@1%FPR over the original 10.0 by trusting
            each user's short history sooner. See docs findings 2026-08-08.
    """

    def __init__(self, alpha: float = 0.5, beta: float = 5.0,
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

    def contributions_frame(self, df: pd.DataFrame) -> np.ndarray:
        """Per-row, per-feature log-likelihood-ratio, aligned to `df` order.

        Returns an (n_rows, n_features) array where column k is
        `log p_global(x_k) - log p_user(x_k)` — the additive contribution of feature
        `self.features[k]` to the total logrisk. This is what makes the score
        explainable (each feature's push toward/away from "risky"), and it is the
        input the weighted/calibrated variant learns per-feature weights on. Counts
        are accumulated strictly past-only within each user (no leakage).
        """
        d = self._prep(df).reset_index(drop=True)
        d["_pos"] = np.arange(len(d))
        d = d.sort_values(["user_id", "login_timestamp"], kind="mergesort")

        feats = self.features
        out = np.zeros((len(d), len(feats)), dtype=float)
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
            row = pos_arr[i]
            for k, f in enumerate(feats):
                v = cols[f][i]
                pg = self._p_global(f, v)
                pu = (counts[f].get(v, 0) + self.beta * pg) / (totals[f] + self.beta)
                out[row, k] = math.log(pg) - math.log(pu)
                counts[f][v] += 1
                totals[f] += 1
        return out

    def score_frame(self, df: pd.DataFrame) -> np.ndarray:
        """Return per-row logrisk aligned to `df` order (past-only per user)."""
        return self.contributions_frame(df).sum(axis=1)

    def contributions_event(
        self,
        values: dict[str, str],
        user_counts: dict[str, dict[str, int]] | None = None,
        user_totals: dict[str, int] | None = None,
    ) -> dict[str, float]:
        """Per-feature LLR for a single login given materialised prior counts.

        `values` maps feature name → string value (same encoding as offline
        `astype(str)`, including hour as ``\"12\"``). Missing prior counts default
        to empty (cold-start user). Does not mutate counts — caller updates the
        profile after scoring (compute-then-update).
        """
        counts = user_counts or {}
        totals = user_totals or {}
        out: dict[str, float] = {}
        for f in self.features:
            v = values[f]
            pg = self._p_global(f, v)
            c = counts.get(f, {}).get(v, 0)
            t = totals.get(f, 0)
            pu = (c + self.beta * pg) / (t + self.beta)
            out[f] = math.log(pg) - math.log(pu)
        return out

    def score_event(
        self,
        values: dict[str, str],
        user_counts: dict[str, dict[str, int]] | None = None,
        user_totals: dict[str, int] | None = None,
    ) -> float:
        """Sum of per-feature LLRs for one event (native Freeman logrisk)."""
        return float(sum(self.contributions_event(values, user_counts, user_totals).values()))

    @staticmethod
    def logrisk_to_proba(logrisk: float) -> float:
        """Map native logrisk → risk_score in [0, 1] (contracts `logistic_logrisk`)."""
        # Stable logistic; clamp extremes for overflow safety.
        if logrisk >= 0:
            z = math.exp(-logrisk)
            return 1.0 / (1.0 + z)
        z = math.exp(logrisk)
        return z / (1.0 + z)

    def to_serving_dict(self) -> dict:
        """JSON-serialisable globals for decision-service (no pickle / ml import)."""
        return {
            "alpha": self.alpha,
            "beta": self.beta,
            "features": list(self.features),
            "global_counts": {f: dict(self.global_counts[f]) for f in self.features},
            "global_total": {f: int(self.global_total[f]) for f in self.features},
            "vocab": {f: int(self.vocab[f]) for f in self.features},
        }

    @classmethod
    def from_serving_dict(cls, data: dict) -> "FreemanScorer":
        obj = cls(
            alpha=float(data["alpha"]),
            beta=float(data["beta"]),
            features=tuple(data["features"]),
        )
        obj.global_counts = {f: Counter(c) for f, c in data["global_counts"].items()}
        obj.global_total = {f: int(t) for f, t in data["global_total"].items()}
        obj.vocab = {f: int(v) for f, v in data["vocab"].items()}
        return obj


class WeightedFreemanScorer:
    """Supervised per-feature calibration of the Freeman likelihood-ratios.

    The raw `FreemanScorer` sums every feature's LLR with **equal weight**. Because
    IPs are near-unique per user, the IP term fires positive on almost every
    first-time-IP *legit* login and dominates the sum — good ranking (ROC-AUC ~0.76)
    but a poor strict-FPR operating point (see the Step 5 finding). This wraps a
    fitted `FreemanScorer` with a logistic layer that learns **one weight per feature
    contribution** from the (few) training labels:

        logit(risk) = b + Σ_k  w_k · llr_k(login)

    so a noisy feature like IP can be down-weighted while user-novel country/ASN is
    up-weighted. The per-signal explanation is preserved (feature k contributes
    `w_k · llr_k` to the log-odds). It consumes labels, so it is a *calibrated
    variant* — the unsupervised `FreemanScorer` stays the label-free primary.
    """

    def __init__(self, base: FreemanScorer, C: float = 1.0) -> None:
        from sklearn.linear_model import LogisticRegression

        self.base = base
        self.lr = LogisticRegression(max_iter=1000, class_weight="balanced", C=C)

    @property
    def features(self) -> list[str]:
        return self.base.features

    def fit(self, train_df: pd.DataFrame, y: np.ndarray) -> "WeightedFreemanScorer":
        contrib = self.base.contributions_frame(train_df)
        self.lr.fit(contrib, np.asarray(y).astype(int))
        return self

    def score_frame(self, df: pd.DataFrame) -> np.ndarray:
        """Calibrated log-odds risk (monotone with predict_proba; used for ranking)."""
        return self.lr.decision_function(self.base.contributions_frame(df))

    def predict_proba_frame(self, df: pd.DataFrame) -> np.ndarray:
        return self.lr.predict_proba(self.base.contributions_frame(df))[:, 1]

    @property
    def weights(self) -> dict[str, float]:
        """Learned weight per feature contribution (for the explainability write-up)."""
        return dict(zip(self.features, self.lr.coef_[0].tolist()))
