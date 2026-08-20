"""Export the fitted LogisticRegression baseline to a JSON serving artifact.

Step 5 left the project serving Freeman (recall@1%FPR 0.105) while the LogReg
baseline reached 0.50 on the same split. ADR-0027 keeps Freeman as the primary,
explainable, label-free score and adds LogReg on the request path as a
*supervised second opinion* that can only escalate the action. For that the PDP
needs the model without sklearn or pickle on the hot path: a StandardScaler +
linear model is ten coefficients and a mean/scale vector, so it ships as JSON
and scores with a dot product.

The artifact also carries its own **operating point** — the decision threshold
at the target FPR, recomputed here on the same chronological test split that
Step 5 reported. Serving must not re-derive that number.

Usage (from rba-ml-training/, venv active):

    python -m ml.export_logreg \\
        --pickle artifacts/step5/logreg.pkl \\
        --out artifacts/serving/logreg-0.1.0.json
"""

from __future__ import annotations

import argparse
import json
import pickle
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from rba_features import schema
from rba_features.features import FEATURE_NAMES

from ml.featurize import build_features
from ml.metrics import evaluate
from ml.train import _split_labels

LABEL = schema.LABEL


def unwrap(model) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Pull (mean, scale, coef, intercept) out of the fitted sklearn pipeline.

    Mirrors `build_models`: StandardScaler → LogisticRegression. Anything else
    is a modelling change that must be reflected here deliberately, so we fail
    loudly instead of exporting a silently wrong artifact.
    """
    steps = getattr(model, "named_steps", None)
    if not steps or "standardscaler" not in steps or "logisticregression" not in steps:
        raise SystemExit(
            "expected a make_pipeline(StandardScaler, LogisticRegression); "
            f"got {type(model).__name__} — update ml/export_logreg.py"
        )
    scaler = steps["standardscaler"]
    clf = steps["logisticregression"]
    if clf.coef_.shape[0] != 1:
        raise SystemExit("expected a binary LogisticRegression")
    return (
        np.asarray(scaler.mean_, dtype=float),
        np.asarray(scaler.scale_, dtype=float),
        np.asarray(clf.coef_[0], dtype=float),
        float(clf.intercept_[0]),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pickle", type=Path, default=Path("artifacts/step5/logreg.pkl"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--data", default="data/subset/logins.parquet")
    parser.add_argument("--target-fpr", type=float, default=0.01)
    parser.add_argument("--train-frac", type=float, default=0.70)
    parser.add_argument("--model-version", default="logreg-0.1.0")
    args = parser.parse_args()

    with args.pickle.open("rb") as fh:
        model = pickle.load(fh)
    mean, scale, coef, intercept = unwrap(model)

    # Same window + split as ml.train, so the threshold matches the reported metrics.
    full = pd.read_parquet(args.data).reset_index(drop=True)
    last_pos = full.loc[full[LABEL], "login_timestamp"].max()
    df = full[full["login_timestamp"] <= last_pos].reset_index(drop=True)
    cut = df["login_timestamp"].quantile(args.train_frac)

    feat = build_features(df)
    feat["split"] = _split_labels(feat["login_timestamp"], cut)
    te = (feat["split"] == "test").to_numpy()
    Xte = feat.loc[te, list(FEATURE_NAMES)]
    yte = feat.loc[te, LABEL].to_numpy().astype(int)
    depth_te = feat.loc[te, "user_login_count"].to_numpy()

    proba = model.predict_proba(Xte)[:, 1]
    res = evaluate(yte, proba, depth_te, args.target_fpr)
    op = res["operating_point"]

    payload = {
        "model_id": "logreg",
        "model_version": args.model_version,
        "model_family": "logistic_regression",
        "input_kind": "feature_vector_v1",
        "feature_schema_version": "1.0.0",
        "features": list(FEATURE_NAMES),
        "scaler": {"mean": mean.tolist(), "scale": scale.tolist()},
        "coef": coef.tolist(),
        "intercept": intercept,
        "proba_mapping": {
            "method": "logistic",
            "detail": "risk_score = 1 / (1 + exp(-(coef · z + intercept))), z = (x - mean) / scale",
        },
        "operating_point": {
            "target_fpr": args.target_fpr,
            "threshold": op.threshold,
            "achieved_fpr": op.achieved_fpr,
            "recall": op.recall,
            "challenge_rate": op.challenge_rate,
            "n_test": int(te.sum()),
            "n_positives": int(yte.sum()),
            "pr_auc": res["pr_auc"],
            "roc_auc": res["roc_auc"],
            "split_cut": str(cut),
        },
        "trained_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "framework": "rba-ml-training/0.1.0",
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, separators=(",", ":")))
    print(
        f"wrote {args.out} ({args.out.stat().st_size} bytes) "
        f"threshold={op.threshold:.6f} recall={op.recall:.4f} "
        f"fpr={op.achieved_fpr:.4f} challenge_rate={op.challenge_rate:.4f}"
    )


if __name__ == "__main__":
    main()
