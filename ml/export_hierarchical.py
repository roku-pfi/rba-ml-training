"""Export freeman-0.4.0, the hybrid Freeman, as a JSON serving artifact.

The network is the paper's hierarchy rooted at the population (world → country
→ ASN → IP); device, OS and browser stay flat Freeman features. In this dataset
value names are pseudonymized per unique value, so which browser is popular has
no real-world meaning, and the rooted agent hierarchy (which leans on that
popularity) misjudged a common browser in a small deployment; the hybrid keeps
almost all of the offline gain without it (finding 2026-10-23-hierarchical).

Usage (from rba-ml-training/, venv active):

    python -m ml.export_hierarchical --out artifacts/serving/freeman-0.4.0.json

Global tables come from the Step 5 training period (label window, first 70% in
time), built with rba_features.hierarchy so the PDP reads exactly what training
counted. Interpolation is uniform (finding 2026-10-23-hierarchical); κ is the
deployment-population prior strength of ADR-0042.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from rba_features import schema
from rba_features.hierarchy import HIERARCHIES, hier_add, hier_path

from ml.models.freeman import FreemanScorer

SERVED = ("network",)
AGENT = ("device_type", "os", "browser")

LABEL = schema.LABEL


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/subset/logins.parquet")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--train-frac", type=float, default=0.70)
    parser.add_argument("--population-prior-strength", type=float, default=100.0)
    parser.add_argument("--model-version", default="freeman-0.4.0")
    args = parser.parse_args()

    full = pd.read_parquet(args.data)
    last_pos = full.loc[full[LABEL], "login_timestamp"].max()
    df = full[full["login_timestamp"] <= last_pos]
    train = df[df["login_timestamp"] < df["login_timestamp"].quantile(args.train_frac)]

    fields = sorted({f for name in SERVED for f in HIERARCHIES[name]})
    tables = {name: {"counts": {}, "leaves": {}} for name in SERVED}
    for row in train[fields].to_dict("records"):
        for name in SERVED:
            hier_add(tables[name]["counts"], tables[name]["leaves"], hier_path(row, name))
    flat = FreemanScorer(beta=5.0, features=AGENT).fit(train).to_serving_dict()
    flat["population_prior_strength"] = args.population_prior_strength

    payload = {
        "model_id": "freeman",
        "model_version": args.model_version,
        "model_family": "freeman_hybrid",
        "input_kind": "freeman_hierarchies+categoricals",
        "feature_schema_version": "1.0.0",
        "hierarchies": {name: list(HIERARCHIES[name]) for name in SERVED},
        "hyperparameters": {
            "interpolation": "uniform",
            "unseen_per_entity": "mu_h = |h|",
            "user_root": "population",
            "population_prior_strength": args.population_prior_strength,
        },
        "proba_mapping": {"method": "logistic_logrisk", "detail": "risk_score = 1 / (1 + exp(-logrisk))"},
        "trained_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "training_rows": int(len(train)),
        "framework": "rba-ml-training/0.1.0",
        "tables": tables,
        "flat": flat,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, separators=(",", ":")))
    print(f"wrote {args.out} ({args.out.stat().st_size / 1e6:.1f} MB, {len(train):,} training logins)")


if __name__ == "__main__":
    main()
