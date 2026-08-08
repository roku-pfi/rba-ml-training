"""RBA-appropriate evaluation metrics.

Account takeover is extraordinarily rare, so we NEVER headline plain accuracy. We
report ranking quality (PR-AUC, ROC-AUC), the operational trade-off (recall at a fixed
low FPR, and the resulting challenge rate), and a breakdown by history depth — because
users with little/no history cannot be protected by behavioural RBA yet.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve


@dataclass
class OperatingPoint:
    target_fpr: float
    threshold: float
    achieved_fpr: float
    recall: float
    challenge_rate: float


def recall_at_fpr(y_true: np.ndarray, scores: np.ndarray, target_fpr: float = 0.01) -> OperatingPoint:
    """Highest-recall threshold whose FPR on legit logins is <= target_fpr.

    Uses roc_curve so tied scores (common with tree probabilities) are grouped
    correctly, rather than a naive quantile that lands on a mass point.
    """
    y = np.asarray(y_true).astype(bool)
    s = np.asarray(scores, dtype=float)
    if y.sum() == 0 or (~y).sum() == 0:
        thr = float(s.max()) + 1.0
        return OperatingPoint(target_fpr, thr, 0.0, 0.0, 0.0)
    fpr, tpr, thresholds = roc_curve(y.astype(int), s)
    ok = np.where(fpr <= target_fpr)[0]
    idx = ok[-1] if len(ok) else 0  # largest FPR not exceeding target → best recall
    thr = float(thresholds[idx])
    flagged = s >= thr
    return OperatingPoint(
        target_fpr=target_fpr,
        threshold=thr,
        achieved_fpr=float(fpr[idx]),
        recall=float(tpr[idx]),
        challenge_rate=float(flagged.mean()),
    )


def ranking_metrics(y_true: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    y = np.asarray(y_true).astype(int)
    s = np.asarray(scores, dtype=float)
    out = {"pr_auc": float("nan"), "roc_auc": float("nan")}
    if y.sum() > 0 and y.sum() < len(y):
        out["pr_auc"] = float(average_precision_score(y, s))
        out["roc_auc"] = float(roc_auc_score(y, s))
    return out


# History-depth buckets keyed on user_login_count (logins BEFORE the current event).
_BUCKETS = [(0, 0, "0 (first login)"), (1, 2, "1-2"), (3, 9, "3-9"), (10, 10**9, "10+")]


def by_history_depth(y_true: np.ndarray, scores: np.ndarray, depth: np.ndarray,
                     threshold: float) -> list[dict]:
    """Per history-depth bucket: n, positives, and recall at the given threshold."""
    y = np.asarray(y_true).astype(bool)
    s = np.asarray(scores, dtype=float)
    d = np.asarray(depth)
    flagged = s >= threshold
    rows = []
    for lo, hi, label in _BUCKETS:
        m = (d >= lo) & (d <= hi)
        pos = int(y[m].sum())
        rec = float(flagged[m & y].sum() / pos) if pos else float("nan")
        rows.append({
            "bucket": label,
            "n": int(m.sum()),
            "positives": pos,
            "recall": rec,
        })
    return rows


def evaluate(y_true: np.ndarray, scores: np.ndarray, depth: np.ndarray,
             target_fpr: float = 0.01) -> dict:
    op = recall_at_fpr(y_true, scores, target_fpr)
    return {
        **ranking_metrics(y_true, scores),
        "operating_point": op,
        "by_depth": by_history_depth(y_true, scores, depth, op.threshold),
    }
