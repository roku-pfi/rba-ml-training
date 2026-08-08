# AGENTS.md — rba-ml-training

Offline **data + modelling pipeline** for a risk-based authentication (RBA) thesis
project. Portable orientation for any AI coding tool. (Cursor users additionally get
the always-on rules in `../.cursor/rules/`; this file is the tool-agnostic mirror of
the essentials.)

## Where we are / where things are stated

This is a **polyrepo**: `rba-features`, `rba-ml-training`, `docs` are separate git repos
cloned side-by-side (org `github.com/roku-pfi`). All roadmap/status/decisions live in
the **`docs`** repo (a sibling checkout, `../docs`):

- **Current status & step checklist → `../docs/plans/status.md`** (single source of truth).
- Phase roadmap & rationale → `../docs/plans/development_plan.md` (§8 is the phase list).
- Narrative progress → `../docs/devlog.md` (newest entry on top).
- Decisions (why) → `../docs/decisions/` (ADRs). Numbers → `../docs/findings/`.

**Read `status.md` first**, then the top `devlog.md` entry, before non-trivial work.

## What this repo does

```
ml/
├── ingest/subset.py   # Step 2: stratified per-user subset of the raw dataset
├── eda/explore.py     # Step 3: class balance, history depth, missingness, time span
├── train.py           # Step 5: Freeman / LogReg / RF / LightGBM + RBA metrics
├── metrics.py         # RBA-appropriate metrics (PR-AUC, recall@low-FPR, by-depth)
├── models/freeman.py  # the primary explainable likelihood-ratio scorer
├── leakage.py         # Step 6: is_attack_ip Variant A vs B leakage experiment
└── calibrate.py       # Freeman calibration study (smoothing sweep, per-feature weights)
```

Features themselves live in the sibling **`rba-features`** package (installed editable)
so training and future online scoring share ONE implementation — the parity contract.

## Setup & key commands

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # includes -e ../rba-features
# macOS: brew install libomp if LightGBM/XGBoost fail to import
```

Data is NOT in git — see `data/README_personal.md` to acquire the Wiefling dataset and
build the subset. Then, with the venv active:

```bash
python -m ml.train --model all   # baselines + RBA metrics  -> reports/step5/
python -m ml.leakage             # is_attack_ip A/B          -> reports/step6/
python -m ml.calibrate           # Freeman calibration       -> reports/freeman_calibration/
```

## Guardrails (do not break)

- **Chronological splits only** (never random). Label `is_account_takeover` is
  ultra-rare (141 positives total).
- **`is_attack_ip` is leakage-sensitive** — only for the A/B experiment, never a
  training target.
- **Never headline plain accuracy** — report PR-AUC, recall @ low FPR, challenge rate,
  and always break down by history depth.
- **Never commit data or secrets** (`data/`, `*.csv`, `*.parquet`, `.venv/`, artifacts,
  reports — all gitignored). Only commit when explicitly asked; Conventional Commits.
- After changing features/profiles, keep `../rba-features/tests/test_parity.py` green.
