# rba-ml-training

Offline **data + modelling pipeline** for the risk-based authentication system.
This is where Phase 1 (feasibility) happens: acquire the dataset, run EDA, build
feature vectors via the shared `rba-features` library, train baselines, and run
the leakage comparison.

> Part of the RBA polyrepo. See `../plans/development_plan.md` (Phase 1) for the
> full plan.

## Layout

```
ml/
├── ingest/subset.py   # Step 2: stratified per-user subset of the raw dataset
├── eda/explore.py     # Step 3: class balance, history depth, missingness, time span
├── train.py           # Step 5: Freeman / LogReg / RF / LightGBM / XGBoost
└── evaluate.py        # Step 5/6: RBA metrics + is_attack_ip leakage comparison
data/
└── README.md          # HOW TO DOWNLOAD THE DATASET (start here)
notebooks/             # scratch EDA only (not the source of truth)
tests/
```

Features themselves live in the sibling `rba-features` package (imported here) so
training and online scoring share one implementation.

## Setup

Clone `rba-features` next to this repo, then:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt      # includes -e ../rba-features
```

> macOS note: LightGBM/XGBoost need OpenMP — `brew install libomp` if imports fail.

## Get the data

See [`data/README.md`](data/README.md). Start with a stratified per-user subset,
not the full 9 GB.

## Phase 1 order of work

1. **Step 2** — `python -m ml.ingest.subset --raw data/raw/<file>.csv`
2. **Step 3** — `python -m ml.eda.explore`
3. **Step 4** — implement features in `../rba-features` (+ parity tests)
4. **Step 5** — `python -m ml.train --model all`
5. **Step 6** — `python -m ml.evaluate` and `python -m ml.train --no-attack-ip`

## Status

Phase 1 scaffold. CLI entry points are stubbed; each is implemented in its step.
