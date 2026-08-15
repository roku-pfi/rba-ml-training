# Dataset acquisition — Wiefling RBA login dataset

> Canonical copy: [`README.md`](README.md) (same content). Keep both in sync.

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

## Is this synthetic data? (nature of the data — read before training)

The Zenodo record says the feature values are "plausible but totally artificial."
That is true, but it does **not** make this the kind of made-up synthetic data you
should avoid training on. The distinction matters:

- **Not** invented from rules/assumptions (which would risk circularity — a model
  just re-learning the generator's assumptions).
- It is a **privacy-preserving synthesis of real data**: the original real dataset
  (3.3M users / ~33M logins at a Norwegian SSO) had to be deleted for privacy, so
  the authors regenerated the *surface values* while **preserving the statistical
  properties and per-user relationships** of the original.

Why that is safe to train on: the signal RBA actually exploits is **per-user
consistency** ("has this user been seen with this value before?"), plus the class
imbalance and temporal patterns — and those are the properties that were preserved.
If a real user logged in 100× from one IP/country/device, that maps to 100× from a
single *artificial* IP/country/device here. This is why the authors show the
synthesized set **reproduces the results obtained on the original real data**. It is
also the standard peer-reviewed open benchmark for RBA (there is no comparable public
login dataset *with attack labels* — privacy makes real ones unreleasable).

### What is faithful vs. degraded

| Faithful (rely on these) | Degraded / artificial (do NOT over-trust) |
|---|---|
| Per-user value consistency → all `*_seen_before` features | Real IP reputation / blocklist meaning |
| Class imbalance (rare takeovers) | True real-world geolocation of an IP |
| Temporal / login-frequency patterns | Absolute geo distance → literal "impossible travel" km |
| Attack / takeover ground-truth labels | Exact timestamps (randomized), RTT positions (shuffled per geo) |

Regeneration specifics (from the paper/Zenodo): IPs and user-agent strings were
randomly generated from public data; country→ASN→IP→city were regenerated so the geo
chain is internally consistent but does **not** match the real world; timestamps
contain added randomness; RTTs resemble real per-geolocation distributions but were
reassigned across users.

### Implications for this project / the report

- **Train on it with confidence**, and describe it accurately as *"a
  privacy-preserving synthesized dataset that preserves the statistical properties of
  a real 33M-login SSO service."* Cite Wiefling et al. (2022).
- **Lean on the seen-before / per-user-consistency features**; treat the absolute
  geo/distance features as approximate on this data (a second reason, beyond the
  `Is Attack IP` leakage issue, to handle them carefully).
- **Do not claim real-world geolocation/IP-reputation performance** from these
  numbers.
- This is exactly why the plan also uses (a) the `is_attack_ip` with/without leakage
  experiment and (b) our own scenario generator layered on top — to *complement*, not
  replace, this empirically-grounded backbone.

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
