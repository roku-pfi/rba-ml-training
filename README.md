# rba-ml-training

Offline **data + modelling pipeline** for the risk-based authentication system.
Phase 1 (feasibility) lives here: acquire the Wiefling dataset, EDA, build
feature vectors via shared `rba-features`, train Freeman + reference baselines,
run the leakage comparison, calibrate Freeman, export a JSON serving artifact.

This repo is **not** on the login path. The PDP scores with the exported JSON
artifacts — `rba-decision-service/artifacts/freeman-0.2.0.json` (primary) and
`logreg-0.1.0.json` (supervised second opinion, ADR-0027). Features are never
re-implemented here.

> Current numbers: [`../docs/findings/2026-08-20-step5-rerun.md`](../docs/findings/2026-08-20-step5-rerun.md).
> The 2026-08-08 baselines note predates ADR-0027 and is superseded.

> Status: [`../docs/plans/status.md`](../docs/plans/status.md) — Phase 1 complete.
> Plan: `../docs/plans/development_plan.md` (Phase 1). AI: [`AGENTS.md`](AGENTS.md).

## Layout

```
ml/
├── ingest/subset.py   # Step 2: stratified per-user subset (whole users)
├── eda/explore.py     # Step 3: imbalance, history depth, missingness, span
├── featurize.py       # replay_user → training matrix (parity with online)
├── train.py           # Step 5: Freeman / LogReg / RF / LightGBM + RBA metrics
├── metrics.py         # PR-AUC, recall @ low FPR, challenge rate, by-depth
├── models/freeman.py  # primary explainable likelihood-ratio scorer
├── leakage.py         # Step 6: is_attack_ip Variant A vs B
├── calibrate.py       # Freeman smoothing sweep; weighting rejected
└── export_freeman.py  # pickle → JSON serving artifact (no pickle online)
data/
├── README.md          # HOW TO DOWNLOAD THE DATASET
├── raw/               # gitignored CSV / zip
└── subset/            # gitignored parquet (logins.parquet)
notebooks/             # scratch only — not the source of truth
reports/               # gitignored metric markdown + EDA plots
artifacts/             # gitignored pickles + serving JSON copy
tests/                 # placeholder (.gitkeep)
requirements.txt       # includes -e ../rba-features
```

`reports/` and `artifacts/` are gitignored; the thesis cites the numbers in
[`../docs/findings/`](../docs/findings/).

## Modelling rules

- **Primary scorer:** Freeman likelihood-ratio (learns “normal”, needs few
  labels). LogReg / RF / LightGBM are **reference baselines** only
  ([ADR-0004](../docs/decisions/0004-modelling-and-label-strategy.md)).
- **Label:** `is_account_takeover` (141 positives in the full set).
- **Splits:** chronological only, inside the **label-covered window**
  (ADR-0007). Never a random split.
- **Metrics:** never headline accuracy. Report PR-AUC, recall @ FPR ≤ 1%,
  challenge rate, **and** breakdown by history depth (~40% of users have a
  single login).
- **`is_attack_ip`:** leakage-sensitive — A/B experiment only, never a target
  or a serving feature.
- **Sentinel users:** drop non-human buckets via `--max-user-logins`
  (ADR-0005). The ~14M-login id is not a person.

Dataset framing: Wiefling is a **privacy-preserving synthesis** of real SSO
logins — cite Wiefling et al. (2022). Rely on per-user consistency; do **not**
claim real-world geolocation or IP-reputation performance
([ADR-0003](../docs/decisions/0003-dataset-selection-and-synthetic-data.md)).

## Setup

Clone `rba-features` next to this repo, then:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt      # includes -e ../rba-features
```

macOS: LightGBM/XGBoost need OpenMP — `brew install libomp` if imports fail.

## Get the data

See [`data/README.md`](data/README.md). Start with a stratified per-user
subset, not the full ~9 GB CSV.

## Phase 1 commands (all implemented)

From this repo, venv active:

```bash
# Step 2 — subset (whole users; keeps every takeover user)
python -m ml.ingest.subset --raw data/raw/rba-dataset.csv --users 50000

# Step 3 — EDA
python -m ml.eda.explore

# Step 5 — Freeman + baselines (writes reports/step5/, artifacts/step5/)
python -m ml.train --model all

# Step 6 — leakage A/B
python -m ml.leakage

# Freeman calibration (β 10 → 5; per-feature weights rejected)
python -m ml.calibrate

# Export JSON for the PDP — primary scorer
python -m ml.export_freeman \
  --pickle artifacts/step5/freeman.pkl \
  --out artifacts/serving/freeman-0.2.0.json \
  --beta 5.0 --model-version freeman-0.2.0

# Export JSON for the PDP — supervised second opinion (ADR-0027).
# Re-derives the 1%-FPR threshold on the same split and bakes it into the
# artifact, so serving never guesses an operating point.
python -m ml.export_logreg --out artifacts/serving/logreg-0.1.0.json
```

Copy both serving JSONs into `../rba-decision-service/artifacts/` when
refreshing the online scorers. A retrain that moves the supervised operating
point must also update
[`../docs/findings/2026-08-19-supervised-escalation-and-failed-logins.md`](../docs/findings/2026-08-19-supervised-escalation-and-failed-logins.md)
and the pinned assertion in `rba-decision-service/tests/test_escalation.py`.

Cited write-ups:

| Step | Finding |
|---|---|
| EDA | [`../docs/findings/2026-08-08-phase1-eda.md`](../docs/findings/2026-08-08-phase1-eda.md) |
| Features | [`../docs/findings/2026-08-08-step4-feature-validation.md`](../docs/findings/2026-08-08-step4-feature-validation.md) |
| Baselines | [`../docs/findings/2026-08-08-step5-baselines.md`](../docs/findings/2026-08-08-step5-baselines.md) |
| Leakage | [`../docs/findings/2026-08-08-step6-leakage.md`](../docs/findings/2026-08-08-step6-leakage.md) |
| Calibration | [`../docs/findings/2026-08-08-freeman-calibration.md`](../docs/findings/2026-08-08-freeman-calibration.md) |
| Dataset size | [`../docs/findings/2026-08-08-dataset-sufficiency.md`](../docs/findings/2026-08-08-dataset-sufficiency.md) |

## Status

**Phase 1 complete.** Next modelling work is Phase 6 (ML lifecycle + scenario
generator) — not the current path (IdP-5). Roadmap: `../docs/plans/status.md`.
