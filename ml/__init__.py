"""rba-ml-training: offline data + modelling pipeline for the RBA system.

Steps (Phase 1 of plans/development_plan.md):
    ingest/  -> Step 2: acquire + subset the Wiefling dataset
    eda/     -> Step 3: exploratory analysis
    (features live in the shared rba_features package)
    train    -> Step 5: baselines (Freeman, LogReg, RF, GBM) + RBA metrics
    leakage  -> Step 6: the is_attack_ip Variant A vs B leakage comparison
"""
