# Dataset acquisition — Wiefling RBA login dataset

This folder holds the raw and derived data used for Phase 1 (feasibility). **The
data itself is never committed** (see the repo `.gitignore`). Follow the steps
below on the machine where you train.

Parent pipeline: [`../README.md`](../README.md). Schema mapping:
`rba_features.schema`.

## What the dataset is

**"Login Data Set for Risk-Based Authentication"** (Wiefling, Jørgensen, Thunem,
Lo Iacono). Synthesised from the real behaviour of ~3.3M users / ~33M login
attempts at a Norwegian SSO service (Feb 2020 – Feb 2021). Purpose-built for RBA
research and safe to publish results on.

- Size: **~1.1 GB compressed / ~9 GB uncompressed** (a single large CSV).
- Label column: `Is Account Takeover`.
- Zenodo: <https://zenodo.org/records/6782156>
- Kaggle: `dasgroup/rba-dataset`

## Is this synthetic data?

The Zenodo record says the feature values are "plausible but totally artificial."
That is true, but it does **not** make this the kind of made-up synthetic data you
should avoid training on:

- **Not** invented from rules/assumptions (which would risk circularity).
- It is a **privacy-preserving synthesis of real data**: the original real dataset
  had to be deleted for privacy, so the authors regenerated the *surface values*
  while **preserving the statistical properties and per-user relationships**.

The signal RBA exploits is **per-user consistency** ("has this user been seen
with this value before?"), plus class imbalance and temporal patterns — those
were preserved. Cite Wiefling et al. (2022). Do **not** claim real-world
geolocation or IP-reputation performance from these numbers
([ADR-0003](../../docs/decisions/0003-dataset-selection-and-synthetic-data.md)).

### What is faithful vs. degraded

| Faithful (rely on these) | Degraded / artificial (do NOT over-trust) |
|---|---|
| Per-user value consistency → all `*_seen_before` features | Real IP reputation / blocklist meaning |
| Class imbalance (rare takeovers) | True real-world geolocation of an IP |
| Temporal / login-frequency patterns | Absolute geo distance → literal "impossible travel" km |
| Attack / takeover ground-truth labels | Exact timestamps (randomized), RTT positions (shuffled per geo) |

## Option A — Zenodo (no login required, recommended)

```bash
cd data/
curl -L -O "https://zenodo.org/records/6782156/files/rba-dataset.zip?download=1"
unzip rba-dataset.zip -d raw/
```

Or: `pip install zenodo_get && zenodo_get 6782156 -o data/raw/`

## Option B — Kaggle API

1. Create a Kaggle API token (Account → "Create New API Token") → `kaggle.json`.
2. Place at `~/.kaggle/kaggle.json` and `chmod 600 ~/.kaggle/kaggle.json`.
3. Then:

```bash
pip install kaggle
cd data/
kaggle datasets download -d dasgroup/rba-dataset
unzip rba-dataset.zip -d raw/
```

## Option C — GitHub mirror

<https://github.com/das-group/rba-dataset> (may link back to Zenodo for the CSV).

## Where to put it

```
data/
├── raw/            # unzipped original CSV(s)  (git-ignored)
├── subset/         # stratified per-user parquet  (git-ignored)
└── README.md       # this file (committed)
```

Start with a **stratified per-user subset**, not the full 9 GB — `ml/ingest/subset.py`
samples whole users so per-user history stays intact.

```bash
python -m ml.ingest.subset --raw data/raw/rba-dataset.csv --users 50000
```

Writes `data/subset/logins.parquet` with columns renamed to `rba_features.schema`,
timestamp parsed, `data_source='real'`.

## Expected columns (raw headers)

| Raw header | Internal name (`rba_features.schema`) |
|---|---|
| `index` | `index` |
| `Login Timestamp` | `login_timestamp` |
| `User ID` | `user_id` |
| `Round-Trip Time [ms]` | `rtt_ms` |
| `IP Address` | `ip_address` |
| `Country`, `Region`, `City` | `country`, `region`, `city` |
| `ASN` | `asn` |
| `User Agent String` | `user_agent` |
| `Browser Name and Version` | `browser` |
| `OS Name and Version` | `os` |
| `Device Type` | `device_type` |
| `Login Successful` | `login_successful` |
| `Is Attack IP` | `is_attack_ip` (**leakage-sensitive**) |
| `Is Account Takeover` | `is_account_takeover` (**the label**) |

## Known gotchas

- **Extreme class imbalance** → PR-AUC and recall at a fixed low FPR; never
  headline accuracy.
- **Sparse per-user history** (mean ~3.8 / median 2) → metrics by history depth.
- **`Is Attack IP` leakage** → train two variants (with / without) and compare.
- **Temporal integrity** → chronological split only.
- **Sentinel user** (~14M logins) → drop via `--max-user-logins` (ADR-0005).
- **Label-covered window** ends 2020-11-23 — later rows have zero positives
  (ADR-0007).
