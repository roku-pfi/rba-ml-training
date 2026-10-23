"""Freeman et al. (2016) with the paper's hierarchical smoothing (Sect. II-C).

Our first Freeman (`freeman.py`) scores seven flat categoricals as independent
evidence. The paper scores two features, each a nested hierarchy whose levels
are combined by linear interpolation:

    network: world → country → ASN → IP                 (paper: … → org → AS → country → world)
    agent:   world → device → device+OS → device+OS+browser   (paper: UA → browser → OS → app → world)

For a value x with ancestor h at level k (k = 0 is the world, L the leaf):

    p_k(x) = p(x | h) · p(h),  p(h) = c(h) / N  (maximum likelihood),
    p(x | h) = max(c(x), 1) / (N_h + M_h)      (Eq. 8: c(x) / (N_h + M_h) if seen),
    p_L(x)  = c(x) / N,
    p(x)    = Σ_k λ_k p_k(x)                   (Eq. 11; λ learned on held-out legit logins),

with M_h unseen leaves per entity, μ_h = |h| (the paper's better choice):
M_h = Σ_children M_child + |h|, which gives M_h = (L − k)·|h|, |h| = distinct
leaves seen under h. The same estimate restricted to the user's own history
gives p(x | u, L). logrisk = Σ_features log p(x) − log p(x | u, L).

A new IP inside the user's usual ASN now keeps most of its probability, so a
new network counts once, not three times (IP, ASN and country were separate
features before), and a new country still falls to the world level.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Iterable

import numpy as np
import pandas as pd
from rba_features.canonical import browser_family, os_family

HIERARCHIES: dict[str, tuple[str, ...]] = {
    "network": ("country", "asn", "ip_address"),
    "agent": ("device_type", "os", "browser"),
}


class _Tables:
    """Counts for one hierarchy: c(prefix) and distinct leaves under each prefix."""

    __slots__ = ("depth", "count", "leaves", "total")

    def __init__(self, depth: int) -> None:
        self.depth = depth
        self.count: dict[tuple, int] = defaultdict(int)  # prefix → logins
        self.leaves: dict[tuple, set] = defaultdict(set)  # prefix → distinct full paths
        self.total = 0

    def add(self, path: tuple) -> None:
        self.total += 1
        for k in range(1, self.depth + 1):
            self.count[path[:k]] += 1
        for k in range(0, self.depth):
            self.leaves[path[:k]].add(path)

    def level_probs(self, path: tuple) -> list[float]:
        """[p_0(x), …, p_L(x)] for the leaf `path` (Eq. 9 per level)."""
        n, depth = self.total, self.depth
        if n == 0:
            return [0.0] * (depth + 1)
        c_leaf = self.count.get(path, 0)
        out = []
        for k in range(depth + 1):
            if k == depth:
                out.append(c_leaf / n)
                continue
            prefix = path[:k]
            n_h = n if k == 0 else self.count.get(prefix, 0)
            if n_h == 0:  # unseen entity: p(h) = 0
                out.append(0.0)
                continue
            m_h = (depth - k) * len(self.leaves.get(prefix, ()))
            out.append(max(c_leaf, 1) / (n_h + m_h) * (n_h / n))
        return out


def fit_lambdas(rows: np.ndarray, iters: int = 300) -> np.ndarray:
    """Interpolation weights maximising Σ log(λ · p) over legit logins (Alg. 1, via EM)."""
    rows = rows[rows.sum(axis=1) > 0]
    lam = np.full(rows.shape[1], 1.0 / rows.shape[1])
    for _ in range(iters):
        mix = rows * lam
        lam = (mix / mix.sum(axis=1, keepdims=True)).mean(axis=0)
    return lam


def prep(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["browser"] = d["browser"].map(browser_family)
    d["os"] = d["os"].map(os_family)
    d["hour"] = pd.to_datetime(d["login_timestamp"]).dt.hour
    for cols in HIERARCHIES.values():
        for c in cols:
            d[c] = d[c].astype(str)
    return d


def paths(d: pd.DataFrame, name: str) -> list[tuple]:
    return list(zip(*(d[c].to_numpy() for c in HIERARCHIES[name])))


def global_tables(d: pd.DataFrame) -> dict[str, _Tables]:
    tables = {name: _Tables(len(cols)) for name, cols in HIERARCHIES.items()}
    for name in HIERARCHIES:
        for path in paths(d, name):
            tables[name].add(path)
    return tables


def level_matrix(d: pd.DataFrame, tables: dict[str, _Tables] | None) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Per hierarchy: (global level probs, user level probs) for every row, past-only per user.

    `tables` None means global probabilities are not needed (only user rows).
    Rows are in `d` order; a user's first login has an all-zero user row.
    """
    order = np.argsort(d["login_timestamp"].to_numpy(), kind="mergesort")
    users = d["user_id"].to_numpy()
    out = {}
    for name, cols in HIERARCHIES.items():
        p = paths(d, name)
        g = np.zeros((len(d), len(cols) + 1))
        u = np.zeros((len(d), len(cols) + 1))
        per_user: dict[str, _Tables] = {}
        for i in order:
            if tables is not None:
                g[i] = tables[name].level_probs(p[i])
            t = per_user.setdefault(users[i], _Tables(len(cols)))
            u[i] = t.level_probs(p[i])
            t.add(p[i])
        out[name] = (g, u)
    return out


def rooted(u: np.ndarray, pg: np.ndarray) -> np.ndarray:
    """User levels with the world level replaced by the global estimate p(x).

    The paper smooths a user's hierarchy within the user's own counts, so a
    value new to the user gets 1/(N_u + M_u) whatever its global rarity, and a
    globally rare value new to the user looks safe. Rooting the user's
    hierarchy at the population (Freeman's Dirichlet prior, applied at the top)
    makes a value new at every level cost -log λ_0, bounded and counted once.
    """
    out = u.copy()
    out[:, 0] = pg
    return out


def logrisk(levels: dict[str, tuple[np.ndarray, np.ndarray]], lambdas: dict[str, tuple[np.ndarray, np.ndarray]],
            names: Iterable[str] = tuple(HIERARCHIES), root_at_global: bool = False) -> np.ndarray:
    """Σ_features log p(x) − log p(x|u,L); a user's first login scores 0 (no history)."""
    total = None
    for name in names:
        g, u = levels[name]
        lam_g, lam_u = lambdas[name]
        pg = g @ lam_g
        pu = (rooted(u, pg) if root_at_global else u) @ lam_u
        cold = u.sum(axis=1) == 0
        contrib = np.where(cold, 0.0, np.log(np.maximum(pg, 1e-300)) - np.log(np.maximum(pu, 1e-300)))
        total = contrib if total is None else total + contrib
    return total
