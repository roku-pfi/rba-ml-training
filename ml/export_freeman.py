"""Export a fitted FreemanScorer to a JSON serving artifact (no pickle on the hot path).

Usage (from rba-ml-training/, venv active):

    python -m ml.export_freeman \\
        --pickle artifacts/step5/freeman.pkl \\
        --out artifacts/serving/freeman-0.1.0.json \\
        --beta 5.0

Globals come from the pickle; ``--beta`` overrides the Dirichlet prior to the
calibrated default (findings 2026-08-08) without re-fitting population counts.
"""

from __future__ import annotations

import argparse
import json
import pickle
from datetime import datetime, timezone
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pickle", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--beta",
        type=float,
        default=None,
        help="Override Dirichlet beta (default: keep pickle value).",
    )
    parser.add_argument("--model-version", default="freeman-0.1.0")
    args = parser.parse_args()

    with args.pickle.open("rb") as fh:
        scorer = pickle.load(fh)

    if args.beta is not None:
        scorer.beta = float(args.beta)

    serving = scorer.to_serving_dict()
    payload = {
        "model_id": "freeman",
        "model_version": args.model_version,
        "model_family": "freeman",
        "input_kind": "freeman_categoricals",
        "feature_schema_version": "1.0.0",
        "freeman_features": serving["features"],
        "hyperparameters": {"alpha": serving["alpha"], "beta": serving["beta"]},
        "proba_mapping": {
            "method": "logistic_logrisk",
            "detail": "risk_score = 1 / (1 + exp(-logrisk))",
        },
        "trained_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "framework": "rba-ml-training/0.1.0",
        "scorer": serving,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, separators=(",", ":")))
    print(f"wrote {args.out} ({args.out.stat().st_size} bytes) beta={serving['beta']}")


if __name__ == "__main__":
    main()
