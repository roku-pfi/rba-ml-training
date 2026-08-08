"""Step 5/6: evaluate models with RBA-appropriate metrics.

Reports (never plain accuracy):
    - account-takeover recall @ a fixed low FPR (e.g. <= 1%)
    - PR-AUC and ROC-AUC
    - challenge rate (% of legit logins pushed to MFA/reauth)
    - logins-to-protection / per-history-depth breakdown
    - inference latency + model size (these size the serving pod later)

Also runs the Variant A (with is_attack_ip) vs Variant B (without) comparison.
Implemented in Step 5/6.
"""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate RBA models.")
    parser.add_argument("--artifacts", default="artifacts/")
    parser.add_argument("--data", default="data/subset/logins.parquet")
    parser.add_argument("--target-fpr", type=float, default=0.01)
    parser.parse_args()
    raise NotImplementedError("Evaluation is implemented in Phase 1, Step 5/6.")


if __name__ == "__main__":
    main()
