"""Step 5: train baseline scorers on the feature vectors.

Models: Freeman likelihood-ratio baseline, LogisticRegression, RandomForest,
and LightGBM/XGBoost (primary). Uses rba_features.replay to build past-only
feature vectors, then a chronological 70/15/15 split.

Implemented in Step 5.
"""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Train RBA baseline models.")
    parser.add_argument("--data", default="data/subset/logins.parquet")
    parser.add_argument(
        "--model",
        choices=["freeman", "logreg", "rf", "lgbm", "xgb", "all"],
        default="all",
    )
    parser.add_argument(
        "--no-attack-ip",
        action="store_true",
        help="Variant B: drop is_attack_ip to test for leakage (plan section 6).",
    )
    parser.add_argument("--out", default="artifacts/")
    parser.parse_args()
    raise NotImplementedError("Training is implemented in Phase 1, Step 5.")


if __name__ == "__main__":
    main()
