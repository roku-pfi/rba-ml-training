"""Step 3: exploratory data analysis.

Answers the questions that drive every later decision:
    - class balance (how rare is account takeover?)
    - logins-per-user distribution (history depth)
    - missingness per column
    - timestamp span (for the chronological split)

Writes summary tables + plots under reports/eda/. Implemented in Step 3.
"""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Run EDA on the login subset.")
    parser.add_argument("--data", default="data/subset/logins.parquet")
    parser.add_argument("--out", default="reports/eda")
    parser.parse_args()
    raise NotImplementedError("EDA is implemented in Phase 1, Step 3.")


if __name__ == "__main__":
    main()
