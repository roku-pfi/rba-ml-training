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


def level_matrix(d: pd.DataFrame, tables: dict[str, _Tables] | None,
                 learn: np.ndarray | None = None) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Per hierarchy: (global level probs, user level probs) for every row, past-only per user.

    `tables` None means global probabilities are not needed (only user rows).
    Rows are in `d` order; a user's first login has an all-zero user row.
    `learn` False marks probes (simulated attackers): scored, never folded in.
    """
    order = np.argsort(d["login_timestamp"].to_numpy(), kind="mergesort")
    users = d["user_id"].to_numpy()
    learn = np.ones(len(d), dtype=bool) if learn is None else learn
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
            if learn[i]:
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


# --- Chain-rule Freeman: flat Freeman applied to each level's conditional ---------
#
# p(path) = Π_k q(h_k | h_{k-1}). Global: q_g = (c_g(h_k) + α) / (c_g(h_{k-1}) +
# α (V(h_{k-1}) + 1)), V = distinct children. User: q_u = (c_u(h_k) + β q_g) /
# (c_u(h_{k-1}) + β), Freeman's Dirichlet toward the population at every level.
# logrisk = Σ_k log q_g − log q_u: a novelty adds log((c_u(parent) + β) / β) once,
# at the level where the login leaves the user's history; below that level the
# user's counts are empty and the terms vanish.


class _ChainTables(_Tables):
    """_Tables plus distinct children per prefix (for the global smoothing)."""

    __slots__ = ("children",)

    def __init__(self, depth: int) -> None:
        super().__init__(depth)
        self.children: dict[tuple, set] = defaultdict(set)

    def add(self, path: tuple) -> None:
        super().add(path)
        for k in range(self.depth):
            self.children[path[:k]].add(path[k])

    def conditionals(self, path: tuple, alpha: float) -> list[float]:
        out = []
        for k in range(1, self.depth + 1):
            parent = path[: k - 1]
            c_parent = self.total if k == 1 else self.count.get(parent, 0)
            v = len(self.children.get(parent, ()))
            out.append((self.count.get(path[:k], 0) + alpha) / (c_parent + alpha * (v + 1)))
        return out


def chain_logrisk(d: pd.DataFrame, train: pd.DataFrame, beta: float = 5.0, alpha: float = 0.5,
                  learn: np.ndarray | None = None, per_level: bool = False):
    """Per-row chain-rule logrisk (sum over hierarchies), users past-only.

    `per_level` also returns {hierarchy: (n_rows, depth) contributions}.
    """
    learn = np.ones(len(d), dtype=bool) if learn is None else learn
    order = np.argsort(d["login_timestamp"].to_numpy(), kind="mergesort")
    users = d["user_id"].to_numpy()
    total = np.zeros(len(d))
    levels = {}
    for name, cols in HIERARCHIES.items():
        g = _ChainTables(len(cols))
        for path in paths(train, name):
            g.add(path)
        p = paths(d, name)
        contrib = np.zeros((len(d), len(cols)))
        per_user: dict[str, _Tables] = {}
        for i in order:
            t = per_user.setdefault(users[i], _Tables(len(cols)))
            if t.total:
                qg = g.conditionals(p[i], alpha)
                for k in range(1, len(cols) + 1):
                    c_parent = t.total if k == 1 else t.count.get(p[i][: k - 1], 0)
                    qu = (t.count.get(p[i][:k], 0) + beta * qg[k - 1]) / (c_parent + beta)
                    contrib[i, k - 1] = math.log(qg[k - 1]) - math.log(qu)
            if learn[i]:
                t.add(p[i])
        total += contrib.sum(axis=1)
        levels[name] = contrib
    return (total, levels) if per_level else total


def chain_stats(d: pd.DataFrame, train: pd.DataFrame, alpha: float = 0.5,
                learn: np.ndarray | None = None) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Per hierarchy and level: (c_u(h_k), c_u(h_{k-1}), q_g(h_k | h_{k-1})), users past-only."""
    learn = np.ones(len(d), dtype=bool) if learn is None else learn
    order = np.argsort(d["login_timestamp"].to_numpy(), kind="mergesort")
    users = d["user_id"].to_numpy()
    out = {}
    for name, cols in HIERARCHIES.items():
        g = _ChainTables(len(cols))
        for path in paths(train, name):
            g.add(path)
        p = paths(d, name)
        depth = len(cols)
        cc, cp, qg = (np.zeros((len(d), depth)) for _ in range(3))
        per_user: dict[str, _Tables] = {}
        for i in order:
            t = per_user.setdefault(users[i], _Tables(depth))
            qg[i] = g.conditionals(p[i], alpha)
            for k in range(1, depth + 1):
                cp[i, k - 1] = t.total if k == 1 else t.count.get(p[i][: k - 1], 0)
                cc[i, k - 1] = t.count.get(p[i][:k], 0)
            if learn[i]:
                t.add(p[i])
        out[name] = (cc, cp, qg)
    return out


def chain_from_stats(stats, betas: dict[str, np.ndarray], per_level: bool = False):
    """logrisk from chain_stats; a level whose parent the user never saw adds 0."""
    total, levels = None, {}
    for name, (cc, cp, qg) in stats.items():
        b = betas[name][None, :]
        contrib = np.where(cp > 0, np.log(qg) - np.log((cc + b * qg) / (cp + b)), 0.0)
        levels[name] = contrib
        total = contrib.sum(axis=1) if total is None else total + contrib.sum(axis=1)
    return (total, levels) if per_level else total


def fit_betas(stats, rows: np.ndarray, grid=np.logspace(-3, 2, 61)) -> dict[str, np.ndarray]:
    """Per level, the β maximising the likelihood of held-out legitimate logins under q_u."""
    out = {}
    for name, (cc, cp, qg) in stats.items():
        betas = []
        for k in range(cc.shape[1]):
            m = rows & (cp[:, k] > 0)
            c, n, q = cc[m, k], cp[m, k], qg[m, k]
            ll = [np.sum(np.log((c + b * q) / (n + b))) for b in grid]
            betas.append(float(grid[int(np.argmax(ll))]))
        out[name] = np.array(betas)
    return out
