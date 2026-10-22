"""D-2: what each authentication policy would have done on the held-out logins.

Replays the chronological test split (the same label window and 70/30 cut as
Step 5) through the **production decision code**, not a re-implementation: the
deployed Freeman and LogReg artifacts, `rba_contracts.apply_policy` with the
starter bands, and the decision service's rule functions (failed-login burst /
lockout, impossible travel and VPN, the supervised second opinion), each only
raising the action (`escalate`). Every login is scored against its user's
history up to that moment (train period included), then folded in, exactly as
online.

Only logins with the right password reach a policy, so the population is the
test split's successful logins: the 38 account takeovers (attackers who had the
password) and the legitimate rest. Failed attempts still feed the profiles and
the failed-login rule.

Policies compared:
- password_only : today's default; never challenges.
- always_mfa    : a passkey on every login.
- adaptive      : the engine as deployed (ACTIVE, starter policy).
- adaptive_no_block : the same, with CRITICAL stepping up instead of blocking;
                    only the lockout rule still blocks.
- freeman_only  : ablation, the score and bands without rules or second opinion.

Per policy: legitimate friction (any step-up or block), legitimate blocks, and
takeovers stopped (any non-ALLOW; a step-up stops an attacker without the
user's passkey). 95% intervals: Wilson per login, and a percentile bootstrap
that resamples users (logins of one user are not independent).

Usage (from rba-ml-training/, venv active):
    python -m ml.policy_compare            # -> reports/policy_compare/{report.md,results.json}
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from rba_contracts import Action, BasePolicy, EngineMode, LevelToAction, RiskLevel, ScoreBand, TenantPolicy
from rba_contracts.policy import apply_policy
from rba_decision_service.scoring.freeman import FreemanOnlineScorer
from rba_decision_service.scoring.logreg import LogRegOnlineScorer
from rba_decision_service.services.escalate import escalate
from rba_decision_service.services.reasons import failed_login_signal
from rba_features import schema
from rba_features.features import compute_features, update_profile
from rba_features.profile import ProfileState
from rba_features.travel import compute_travel

EVENT_COLS = ["login_timestamp", "ip_address", "asn", "country", "device_type", "os", "browser", "login_successful"]
# The decision service's defaults (config.py).
BURST, LOCKOUT = 3, 10
POLICIES = ("password_only", "always_mfa", "adaptive", "adaptive_no_block", "freeman_only")
LOW_CUTS = (0.10, 0.20, 0.30, 0.40, 0.50, 0.60)


def starter_policy(low_max: float = 0.30, *, critical: Action = Action.BLOCK) -> TenantPolicy:
    """The IdP's starter policy (rba_idp.services.admin.starter_policy), ACTIVE."""
    return TenantPolicy(
        tenant_id="replay",
        policy_version="replay",
        base_policy=BasePolicy.PASSWORD_ONLY,
        engine_mode=EngineMode.ACTIVE,
        score_to_level=[
            ScoreBand(max=low_max, level=RiskLevel.LOW),
            ScoreBand(max=max(0.60, low_max + 0.01), level=RiskLevel.MEDIUM),
            ScoreBand(max=max(0.80, low_max + 0.02), level=RiskLevel.HIGH),
            ScoreBand(max=1.00, level=RiskLevel.CRITICAL),
        ],
        level_to_action=LevelToAction(LOW=Action.ALLOW, MEDIUM=Action.REQUIRE_MFA, HIGH=Action.REAUTHENTICATE, CRITICAL=critical),
        fallback_action=Action.REQUIRE_MFA,
    )


def replay(df: pd.DataFrame, cut, freeman: FreemanOnlineScorer, logreg: LogRegOnlineScorer) -> pd.DataFrame:
    """One row per successful test login: label, depth, Freeman score, and the
    engine's raw signals (so any band choice can be applied afterwards)."""
    rows: list[dict] = []
    df = df.sort_values(["user_id", "login_timestamp"], kind="mergesort")
    for user_id, g in df.groupby("user_id", sort=False):
        profile = ProfileState()
        for event, label in zip(g[EVENT_COLS].to_dict("records"), g[schema.LABEL].to_numpy()):
            if event["login_timestamp"] > cut and event["login_successful"]:
                features = compute_features(event, profile)
                travel = compute_travel(event, profile)
                score = freeman.predict(event, profile).risk_score
                failed = failed_login_signal(int(features["failed_logins_last_24h"]), burst_threshold=BURST, lockout_threshold=LOCKOUT)
                rows.append({
                    "user_id": user_id,
                    "ato": bool(label),
                    "depth": int(features["user_login_count"]),
                    "score": score,
                    "rule_floor": failed[1].value if failed else None,
                    "travel": bool(travel.impossible_travel or travel.vpn_or_hosting),
                    "supervised": bool(logreg.predict(features).fired),
                })
            update_profile(profile, event)
    return pd.DataFrame(rows)


def decide(row, policy: TenantPolicy, *, rules: bool) -> Action:
    _, action = apply_policy(float(row.score), policy)
    if rules:
        if isinstance(row.rule_floor, str):  # None becomes NaN in the frame
            action = escalate(action, Action(row.rule_floor))
        if row.travel or row.supervised:
            action = escalate(action, Action.REQUIRE_MFA)
    return action


def actions_for(frame: pd.DataFrame, name: str, policy: TenantPolicy) -> np.ndarray:
    if name == "password_only":
        return np.array([Action.ALLOW.value] * len(frame))
    if name == "always_mfa":
        return np.array([Action.REQUIRE_MFA.value] * len(frame))
    if name == "adaptive_no_block":
        policy = policy.model_copy(update={"level_to_action": policy.level_to_action.model_copy(update={"CRITICAL": Action.REAUTHENTICATE})})
    rules = name in ("adaptive", "adaptive_no_block")
    return np.array([decide(r, policy, rules=rules).value for r in frame.itertuples()])


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (math.nan, math.nan)
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (max(0.0, centre - half), min(1.0, centre + half))


def user_bootstrap(frame: pd.DataFrame, hit: np.ndarray, mask: np.ndarray, *, reps: int, rng) -> tuple[float, float]:
    """Percentile CI of mean(hit[mask]) resampling users with replacement."""
    per_user_hits = defaultdict(int)
    per_user_n = defaultdict(int)
    for user, h, m in zip(frame["user_id"].to_numpy(), hit, mask):
        if m:
            per_user_hits[user] += int(h)
            per_user_n[user] += 1
    users = list(per_user_n)
    if not users:
        return (math.nan, math.nan)
    hits = np.array([per_user_hits[u] for u in users])
    ns = np.array([per_user_n[u] for u in users])
    draws = rng.integers(0, len(users), size=(reps, len(users)))
    rates = hits[draws].sum(axis=1) / ns[draws].sum(axis=1)
    return (float(np.percentile(rates, 2.5)), float(np.percentile(rates, 97.5)))


def rate(frame, hit, mask, *, reps, rng) -> dict:
    k, n = int(hit[mask].sum()), int(mask.sum())
    lo, hi = wilson(k, n)
    blo, bhi = user_bootstrap(frame, hit, mask, reps=reps, rng=rng)
    return {"k": k, "n": n, "rate": k / n if n else math.nan, "wilson": [lo, hi], "bootstrap_users": [blo, bhi]}


def summarise(frame: pd.DataFrame, actions: np.ndarray, *, reps: int, rng) -> dict:
    legit, ato = ~frame["ato"].to_numpy(), frame["ato"].to_numpy()
    challenged = actions != Action.ALLOW.value
    blocked = actions == Action.BLOCK.value
    return {
        "legit_friction": rate(frame, challenged, legit, reps=reps, rng=rng),
        "legit_blocked": rate(frame, blocked, legit, reps=reps, rng=rng),
        "ato_stopped": rate(frame, challenged, ato, reps=reps, rng=rng),
        "ato_blocked": rate(frame, blocked, ato, reps=reps, rng=rng),
    }


def pct(x: float) -> str:
    return "—" if x != x else f"{100 * x:.2f} %"


def ci(r: dict, key: str) -> str:
    lo, hi = r[key]
    return f"[{pct(lo)}, {pct(hi)}]"


def main() -> None:
    parser = argparse.ArgumentParser(description="D-2: compare authentication policies on the held-out logins.")
    parser.add_argument("--data", default="data/subset/logins.parquet")
    parser.add_argument("--freeman", default="artifacts/serving/freeman-0.2.0.json")
    parser.add_argument("--logreg", default="artifacts/serving/logreg-0.1.0.json")
    parser.add_argument("--train-frac", type=float, default=0.70)
    parser.add_argument("--reps", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=6)
    parser.add_argument("--out", default="reports/policy_compare")
    parser.add_argument("--fresh", action="store_true", help="replay again instead of reusing replay.parquet")
    args = parser.parse_args()

    t0 = time.time()
    full = pd.read_parquet(args.data).reset_index(drop=True)
    last_pos = full.loc[full[schema.LABEL], "login_timestamp"].max()
    df = full[full["login_timestamp"] <= last_pos].reset_index(drop=True)
    cut = df["login_timestamp"].quantile(args.train_frac)
    freeman = FreemanOnlineScorer.from_path(Path(args.freeman))
    logreg = LogRegOnlineScorer.from_path(Path(args.logreg))
    # The replay is the slow part; its output only depends on the data and models.
    cache = Path(args.out, "replay.parquet")
    if cache.exists() and not args.fresh:
        frame = pd.read_parquet(cache)
    else:
        frame = replay(df, cut, freeman, logreg)
        os.makedirs(args.out, exist_ok=True)
        frame.to_parquet(cache, index=False)
    rng = np.random.default_rng(args.seed)

    policy = starter_policy()
    results = {name: summarise(frame, actions_for(frame, name, policy), reps=args.reps, rng=rng) for name in POLICIES}

    # How the adaptive trade-off moves with the LOW band's upper limit (the
    # console's Política "impacto" in aggregate).
    sweep = []
    for low in LOW_CUTS:
        acts = actions_for(frame, "adaptive", starter_policy(low))
        s = summarise(frame, acts, reps=200, rng=rng)
        sweep.append({"low_max": low, "legit_friction": s["legit_friction"]["rate"], "ato_stopped": s["ato_stopped"]["rate"]})

    # Where the adaptive engine stops takeovers, by history depth.
    acts = actions_for(frame, "adaptive", policy)
    frame = frame.assign(adaptive=acts)
    bins = [(-1, 0, "0"), (0, 2, "1-2"), (2, 9, "3-9"), (9, 10**9, "10+")]
    by_depth = []
    for lo, hi, name in bins:
        part = frame[(frame["depth"] > lo) & (frame["depth"] <= hi)]
        legit, ato = part[~part["ato"]], part[part["ato"]]
        by_depth.append({
            "depth": name,
            "legit_n": int(len(legit)),
            "legit_friction": float((legit["adaptive"] != "ALLOW").mean()) if len(legit) else None,
            "ato_n": int(len(ato)),
            "ato_stopped": int((ato["adaptive"] != "ALLOW").sum()),
        })

    os.makedirs(args.out, exist_ok=True)
    meta = {
        "split_at": str(cut),
        "label_window_end": str(last_pos),
        "successful_test_logins": int(len(frame)),
        "legit": int((~frame["ato"]).sum()),
        "ato": int(frame["ato"].sum()),
        "models": [freeman.artifact.model_version, logreg.artifact.model_version],
        "policy": "starter (0.30 / 0.60 / 0.80 → ALLOW / REQUIRE_MFA / REAUTHENTICATE / BLOCK), ACTIVE",
        "bootstrap_reps": args.reps,
        "seed": args.seed,
    }
    Path(args.out, "results.json").write_text(json.dumps({"meta": meta, "policies": results, "sweep_low_max": sweep, "adaptive_by_depth": by_depth}, indent=2))

    label = {"password_only": "Solo contraseña", "always_mfa": "MFA siempre", "adaptive": "Adaptativa (motor)", "adaptive_no_block": "Adaptativa sin bloqueo", "freeman_only": "Solo Freeman (ablación)"}
    lines = [
        "# D-2 — Policy comparison on the held-out logins",
        "",
        f"Successful test logins: **{meta['successful_test_logins']:,}** ({meta['legit']:,} legitimate, **{meta['ato']}** account takeovers). "
        f"Split at {cut}; models {', '.join(meta['models'])}; policy {meta['policy']}.",
        "",
        "| Policy | Legit friction | 95% CI (Wilson) | 95% CI (users) | Takeovers stopped | 95% CI (Wilson) | 95% CI (users) | Legit blocked |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for name in POLICIES:
        r = results[name]
        f, s = r["legit_friction"], r["ato_stopped"]
        lines.append(
            f"| {label[name]} | {pct(f['rate'])} | {ci(f, 'wilson')} | {ci(f, 'bootstrap_users')} | "
            f"{s['k']}/{s['n']} ({pct(s['rate'])}) | {ci(s, 'wilson')} | {ci(s, 'bootstrap_users')} | {pct(r['legit_blocked']['rate'])} |"
        )
    lines += ["", "## Adaptive trade-off by the LOW band's upper limit", "", "| LOW ≤ | Legit friction | Takeovers stopped |", "|---|---|---|"]
    for row in sweep:
        lines.append(f"| {row['low_max']:.2f} | {pct(row['legit_friction'])} | {pct(row['ato_stopped'])} |")
    lines += ["", "## Adaptive by history depth (prior successful logins)", "", "| Depth | Legit logins | Legit friction | Takeovers | Stopped |", "|---|---|---|---|---|"]
    for row in by_depth:
        lines.append(f"| {row['depth']} | {row['legit_n']:,} | {pct(row['legit_friction']) if row['legit_friction'] is not None else '—'} | {row['ato_n']} | {row['ato_stopped']} |")
    lines += ["", f"_Generated by `python -m ml.policy_compare` in {time.time() - t0:.0f} s._"]
    Path(args.out, "report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
