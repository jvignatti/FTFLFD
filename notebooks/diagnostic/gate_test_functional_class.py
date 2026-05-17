"""
gate_test_functional_class.py

3-gate acceptance test for functional_class feature.
Source: AADT files (aadt_limited.csv + aadt_other.csv) via StandardRouteCode
interval join (BeginMM <= mile_bin <= EndMM).

IMPORTANT: Gate 1 verdict is withheld pending OD-004 resolution.
Do NOT run this script until OD-004 (open_decisions.md) has been explicitly
signed off. Gate 2 and Gate 3 output is present for reference only.

METHODOLOGICAL NOTE — Gate 3 baseline:
Gate 3 trains both models on matched-only train windows (windows where
FunctionalClass is not NaN after AADT join). The baseline recall produced
here will NOT reproduce the iter_001 Phase 1 value of 0.508, which was
trained on all 6,394,710 windows. The gain metric (augmented minus baseline)
is internally valid — both models see identical training data. However, a
gain of +0.005 over a matched-only baseline does not guarantee the augmented
model beats iter_001 on the full val set. Full-val evaluation requires
resolving OD-001 (imputation strategy) after OD-004 is decided.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import spearmanr
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import recall_score, precision_score

DATA_RAW    = Path("data/raw")
DATA_SPLITS = Path("data/splits")

AADT_LIMITED   = DATA_RAW    / "aadt_limited.csv"
AADT_OTHER     = DATA_RAW    / "aadt_other.csv"
TRAIN_FEATURED = DATA_SPLITS / "train_featured.parquet"
VAL_FEATURED   = DATA_SPLITS / "val_featured.parquet"

BASELINE_FATAL_RECALL    = 0.508  # iter_001 LogReg, Phase 1 val set, full training
GATE1_THRESHOLD          = 0.95   # 95% — threshold scope is OD-004 dispute
GATE2_MI_THRESHOLD       = 0.01
GATE2_SPEARMAN_THRESHOLD = 0.05
GATE3_MIN_GAIN           = 0.005

RANDOM_SEED = 42

LOGREG_PARAMS = dict(
    C=1.0,
    class_weight="balanced",
    max_iter=1000,
    solver="lbfgs",
    random_state=RANDOM_SEED,
)

BASE_FEATURES = [
    "segment_crash_rate",
    "segment_fatal_rate",
    "segment_injury_rate",
    "segment_pdo_rate",
    "month",
    "is_winter",
    "is_weekend_heavy",
    "segment_type_is_grid",
]


# ---------------------------------------------------------------------------
# AADT loading and join
# ---------------------------------------------------------------------------

def load_aadt() -> pd.DataFrame:
    """Load aadt_limited and aadt_other, deduplicate on route+interval."""
    limited = pd.read_csv(AADT_LIMITED)
    other   = pd.read_csv(AADT_OTHER)
    aadt = pd.concat([limited, other], ignore_index=True)
    aadt = aadt.drop_duplicates(subset=["StandardRouteCode", "BeginMM", "EndMM"])
    return aadt[["StandardRouteCode", "BeginMM", "EndMM", "FunctionalClass"]].copy()


def parse_segment_id(sid: str):
    """
    Parse LRS_{LRSNUMBER}_{mile_bin} into (lrs_number, mile_bin).
    Returns (None, None) for GRID segments and malformed IDs.
    Splits from the right so LRSNUMBER can contain underscores.
    """
    if not isinstance(sid, str) or not sid.startswith("LRS_"):
        return None, None
    remainder = sid[4:]               # strip leading "LRS_"
    parts = remainder.rsplit("_", 1)  # split on last underscore only
    if len(parts) != 2:
        return None, None
    lrs_number, mile_str = parts
    try:
        return lrs_number, float(mile_str)
    except ValueError:
        return None, None


def build_fc_map(aadt: pd.DataFrame) -> dict:
    """
    Return dict {segment_id: functional_class} for all unique segment IDs
    in both splits. GRID segments and unmatched LRS segments are absent (map
    returns NaN via .map()). When multiple AADT records match a segment, take
    the minimum FunctionalClass.
    """
    train_ids = pd.read_parquet(TRAIN_FEATURED, columns=["segment_id"])["segment_id"]
    val_ids   = pd.read_parquet(VAL_FEATURED,   columns=["segment_id"])["segment_id"]
    unique_ids = pd.Series(
        pd.concat([train_ids, val_ids]).unique(), name="segment_id"
    )

    parsed = pd.DataFrame(
        [parse_segment_id(sid) for sid in unique_ids],
        columns=["lrs_number", "mile_bin"],
    )
    parsed.insert(0, "segment_id", unique_ids.values)

    lrs_only = parsed.dropna(subset=["lrs_number"]).copy()
    lrs_only["mile_bin"] = lrs_only["mile_bin"].astype(float)

    merged = lrs_only.merge(
        aadt,
        left_on="lrs_number",
        right_on="StandardRouteCode",
        how="left",
    )
    in_interval = (
        (merged["BeginMM"] <= merged["mile_bin"]) &
        (merged["mile_bin"] <= merged["EndMM"])
    )
    matched = merged.loc[in_interval]

    fc_min = (
        matched.groupby("segment_id")["FunctionalClass"]
        .min()
        .rename("functional_class")
    )
    return fc_min.to_dict()


def assign_fc(df: pd.DataFrame, fc_map: dict) -> pd.DataFrame:
    """Add functional_class column; NaN for GRID and unmatched LRS."""
    df = df.copy()
    df["functional_class"] = df["segment_id"].map(fc_map)
    return df


# ---------------------------------------------------------------------------
# Gates
# ---------------------------------------------------------------------------

def gate1(train_df: pd.DataFrame):
    """
    Gate 1: Coverage check.
    Reports both all-records and fatal-only coverage without issuing a verdict —
    OD-004 must be resolved before the threshold scope is determined.
    """
    total   = len(train_df)
    matched = train_df["functional_class"].notna().sum()
    cov_all = matched / total

    fatal_mask  = train_df["has_fatal"] == 1
    fatal_total = fatal_mask.sum()
    fatal_match = train_df.loc[fatal_mask, "functional_class"].notna().sum()
    cov_fatal   = fatal_match / fatal_total if fatal_total > 0 else 0.0

    print("=== GATE 1: AVAILABILITY ===")
    print(f"  Total train windows:        {total:,}")
    print(f"  Matched windows:            {matched:,}")
    print(f"  Coverage (all records):     {cov_all:.4f}  ({cov_all*100:.2f}%)")
    print(f"  Fatal train windows:        {fatal_total:,}")
    print(f"  Fatal matched:              {fatal_match:,}")
    print(f"  Coverage (fatal records):   {cov_fatal:.4f}  ({cov_fatal*100:.2f}%)")
    print(f"  Gate 1 threshold (written): {GATE1_THRESHOLD:.0%}")
    print()

    for label, cov in [("All-records ", cov_all), ("Fatal-only  ", cov_fatal)]:
        if cov >= GATE1_THRESHOLD:
            print(f"  {label}verdict:  PASS")
        else:
            short = (GATE1_THRESHOLD - cov) * 100
            print(f"  {label}verdict:  FAIL  ({short:.1f} pp below threshold)")

    print()
    print("  *** OD-004 is UNRESOLVED — Gate 1 verdict is not accepted until")
    print("      open_decisions.md OD-004 has been explicitly signed off. ***")
    print()
    return cov_all, cov_fatal


def gate2(train_df: pd.DataFrame) -> bool:
    """
    Signal test on training set only, matched windows only.
    Target: label (combined positive class — per CLAUDE.md Gate 2 spec).
    """
    matched = train_df.dropna(subset=["functional_class"]).copy()
    X = matched[["functional_class"]].values
    y = matched["label"].values

    mi = mutual_info_classif(X, y, random_state=RANDOM_SEED)[0]
    r, p = spearmanr(matched["functional_class"], y)

    mi_pass = mi > GATE2_MI_THRESHOLD
    sp_pass = abs(r) > GATE2_SPEARMAN_THRESHOLD
    overall = mi_pass or sp_pass

    print("=== GATE 2: SIGNAL (training set, matched windows only) ===")
    print(f"  Windows used:               {len(matched):,}")
    print(f"  Mutual information:         {mi:.6f}  "
          f"(threshold > {GATE2_MI_THRESHOLD})  {'PASS' if mi_pass else 'FAIL'}")
    print(f"  Spearman r:                 {r:.4f}   "
          f"(threshold > |{GATE2_SPEARMAN_THRESHOLD}|)  {'PASS' if sp_pass else 'FAIL'}")
    print(f"  Spearman p-value:           {p:.4e}")
    print(f"  Gate 2 overall:             {'PASS' if overall else 'FAIL'}")
    print()
    return overall


def gate3(train_df: pd.DataFrame, val_df: pd.DataFrame, threshold: float = 0.5):
    """
    Incremental recall gain on full val set.
    Both models trained on matched-only train windows (functional_class not NaN).
    Both models evaluated on the FULL val set.
    Augmented model: unmatched val windows get functional_class imputed with median
    from matched train windows — for prediction only, never for training.
    Train target: label.  Evaluation target: has_fatal.
    """
    train_m = train_df.dropna(
        subset=BASE_FEATURES + ["functional_class", "label"]
    ).copy()
    y_tr = train_m["label"].values

    val_full = val_df.dropna(subset=BASE_FEATURES + ["has_fatal"]).copy()
    y_va = val_full["has_fatal"].values

    fc_median = train_m["functional_class"].median()

    val_aug = val_full.copy()
    val_aug["functional_class"] = val_aug["functional_class"].fillna(fc_median)

    def fit_eval(features, val_data):
        model = LogisticRegression(**LOGREG_PARAMS)
        model.fit(train_m[features], y_tr)
        proba = model.predict_proba(val_data[features])[:, 1]
        pred  = (proba >= threshold).astype(int)
        return (
            recall_score(y_va, pred, zero_division=0),
            precision_score(y_va, pred, zero_division=0),
            pred.mean(),
            model,
        )

    rec_base, pre_base, flag_base, _          = fit_eval(BASE_FEATURES, val_full)
    rec_aug,  pre_aug,  flag_aug,  model_aug  = fit_eval(
        BASE_FEATURES + ["functional_class"], val_aug
    )
    gain   = rec_aug - rec_base
    passed = gain >= GATE3_MIN_GAIN

    print("=== GATE 3: INCREMENTAL GAIN (full val set) ===")
    print(f"  Matched train windows:      {len(train_m):,}")
    print(f"  Full val windows:           {len(val_full):,}")
    print(f"  Fatal val windows:          {int(y_va.sum()):,}")
    print(f"  FC median (matched train):  {fc_median:.1f}  (imputed into unmatched val)")
    print(f"  Threshold:                  {threshold}")
    print()
    print(f"  Baseline fatal recall:      {rec_base:.4f}")
    print(f"  Baseline precision:         {pre_base:.4f}")
    print(f"  Baseline flag rate:         {flag_base:.4f}")
    print(f"  (Phase 1 full-train baseline reference: {BASELINE_FATAL_RECALL})")
    print()
    print(f"  Augmented fatal recall:     {rec_aug:.4f}")
    print(f"  Augmented precision:        {pre_aug:.4f}")
    print(f"  Augmented flag rate:        {flag_aug:.4f}")
    print()
    print(f"  Recall gain:                {gain:+.4f}  "
          f"(threshold >= {GATE3_MIN_GAIN})  {'PASS' if passed else 'FAIL'}")
    print()
    print("  Augmented model coefficients:")
    for fname, coef in zip(BASE_FEATURES + ["functional_class"], model_aug.coef_[0]):
        print(f"    {fname:<30}: {coef:+.4f}")
    print()
    return passed, gain


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 62)
    print("functional_class — 3-Gate Acceptance Test")
    print("Source: AADT files via StandardRouteCode interval join")
    print()
    print("STOP: OD-004 (open_decisions.md) is UNRESOLVED.")
    print("Gate 1 verdict cannot be accepted until OD-004 is signed off.")
    print("=" * 62)
    print()

    aadt = load_aadt()
    print(f"AADT records (deduplicated): {len(aadt):,}")
    fc_vals = sorted(aadt["FunctionalClass"].dropna().unique().tolist())
    print(f"FunctionalClass values:      {fc_vals}")
    print()

    fc_map = build_fc_map(aadt)
    print(f"Segments with coverage:      {len(fc_map):,}")
    print()

    train_df = assign_fc(pd.read_parquet(TRAIN_FEATURED), fc_map)
    val_df   = assign_fc(pd.read_parquet(VAL_FEATURED),   fc_map)
    print(f"Train windows loaded:        {len(train_df):,}")
    print(f"Val windows loaded:          {len(val_df):,}")
    print()

    cov_all, cov_fatal = gate1(train_df)

    g2 = gate2(train_df)
    if not g2:
        print("Gate 2 FAILED — no signal. Stopping.")
        return

    g3, gain = gate3(train_df, val_df)

    print("=" * 62)
    print("SUMMARY")
    print("=" * 62)
    v1a = "PASS" if cov_all   >= GATE1_THRESHOLD else "FAIL"
    v1f = "PASS" if cov_fatal >= GATE1_THRESHOLD else "FAIL"
    print(f"  Gate 1 all-records  ({cov_all:.2%}):  {v1a}  — VERDICT PENDING OD-004")
    print(f"  Gate 1 fatal-only   ({cov_fatal:.2%}):  {v1f}  — VERDICT PENDING OD-004")
    print(f"  Gate 2:              {'PASS' if g2 else 'FAIL'}")
    print(f"  Gate 3:              {'PASS' if g3 else 'FAIL'}  (gain: {gain:+.4f})")
    print()
    print("  Resolve OD-004 before accepting or rejecting this feature.")
    print()


if __name__ == "__main__":
    main()
