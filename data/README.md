# Dataset acquisition — Wiefling RBA login dataset

This folder holds the raw and derived data used for Phase 1 (feasibility). **The
data itself is never committed** (see `.gitignore`). Follow the steps below on the
machine where you'll do the training.

## What the dataset is

**"Login Data Set for Risk-Based Authentication"** (Wiefling, Jørgensen, Thunem,
Lo Iacono). Synthesised from the real behaviour of ~3.3M users / ~33M login
attempts at a Norwegian SSO service (Feb 2020 – Feb 2021). Purpose-built for RBA
research and safe to publish results on.

- Size: **~1.1 GB compressed / ~9 GB uncompressed** (a single large CSV).
- Label column: `Is Account Takeover`.

## Option A — Zenodo (no login required, recommended)

Record: <https://zenodo.org/records/6782156>

```bash
cd data/
# Download the archive (check the exact filename on the record page):
curl -L -O "https://zenodo.org/records/6782156/files/rba-dataset.zip?download=1"
unzip rba-dataset.zip -d raw/
```

If `zenodo_get` is preferred:

```bash
pip install zenodo_get
zenodo_get 6782156 -o data/raw/
```

## Option B — Kaggle API (needs a token)

Dataset: `dasgroup/rba-dataset`

1. Create a Kaggle API token: kaggle.com → Account → "Create New API Token".
   This downloads `kaggle.json`.
2. Place it at `~/.kaggle/kaggle.json` and `chmod 600 ~/.kaggle/kaggle.json`.
3. Then:

```bash
pip install kaggle
cd data/
kaggle datasets download -d dasgroup/rba-dataset
unzip rba-dataset.zip -d raw/
```

## Option C — GitHub mirror

<https://github.com/das-group/rba-dataset> (may link back to the Zenodo archive
for the full CSV).

## Where to put it

```
data/
├── raw/            # the unzipped original CSV(s) go here  (git-ignored)
├── subset/         # stratified per-user subset for fast iteration (git-ignored)
└── README.md       # this file (committed)
```

Start with a **stratified per-user subset** rather than the full 9 GB — the subset
script (`ml/ingest/subset.py`, Step 2) samples whole users so per-user history
stays intact.

## Expected columns (raw headers)

| Raw header | Meaning |
|---|---|
| `index` | row index |
| `Login Timestamp` | event time |
| `User ID` | pseudonymous user id |
| `Round-Trip Time [ms]` | network RTT |
| `IP Address` | login IP |
| `Country`, `Region`, `City` | geo of the IP |
| `ASN` | autonomous system number |
| `User Agent String` | raw UA |
| `Browser Name and Version` | parsed browser |
| `OS Name and Version` | parsed OS |
| `Device Type` | mobile / desktop / … |
| `Login Successful` | bool |
| `Is Attack IP` | **leakage-sensitive** — see plan section 6 |
| `Is Account Takeover` | **the label** |

The canonical snake_case mapping lives in `rba_features.schema`.

## Known gotchas to design around (from the plan)

- **Extreme class imbalance** (account takeover is rare) → use PR-AUC and recall
  at a fixed low FPR; never headline "accuracy".
- **Sparse per-user history** (mean ~3.8 / median 2 logins per user) → report
  metrics conditioned on history depth.
- **`Is Attack IP` leakage** → train two variants (with / without it) and compare.
- **Temporal integrity** → chronological split (oldest 70% / 15% / newest 15%),
  never a random split.
