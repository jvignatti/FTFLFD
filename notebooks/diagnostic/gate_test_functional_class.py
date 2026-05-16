"""
gate_test_functional_class.py
------------------------------
Gate 2 and Gate 3 test for functional_class feature candidate.

Gate 1 already passed (see notebooks/diagnostic/functional_class_coverage.py):
    All records:   98.72%  PASS
    Fatal records: 99.43%  PASS

Gate 2 — Signal test:
    Mutual information > 0.01 OR Spearman correlation > 0.05

Gate 3 — Incremental fatal recall:
    Retrain LogReg (same hyperparameters as iter_001) with 8 base features + FUNCL
    Fatal recall on val set must exceed iter_001 baseline (0.508) by >= 0.005
    i.e. >= 0.513

Deduplication rule (locked):
    When crash LRSNUMBER matches multiple centerline segments with different FUNCL,
    take minimum FUNCL value (highest functional class — interstate beats local).
    Grid segments (no LRSNUMBER) receive FUNCL = 0 (unclassified).

Data sources:
    Train:      data/splits/train_featured.parquet (8,815,920 windows)
    Val:        data/splits/val_featured.parquet   (124,608 windows, 253 fatal)
    Centerline: data/raw/road_centerline.csv       (78,876 segments)

Baseline (iter_001, Phase 1, threshold=0.5):
    Fatal recall: 0.508
    Flag rate:    0.143

LRS integration methodology: Owen Mosley.
See docs/technical/07_aadt_coverage_diagnosis.md for context.

Run from project root:
    python notebooks/diagnostic/gate_test_functional_class.py

Output:
    python notebooks/diagnostic/gate_test_functional_class.py > notebooks/diagnostic/output/gate_test_functional_class_output.txt
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    recall_score, precision_score, f1_score, log_loss
)
from sklearn.feature_selection import mutual_info_classif

# ---------------------------------------------------------------------------
# Configuration — locked, mirrors iter_001
# ---------------------------------------------------------------------------
BASELINE_FATAL_RECALL = 0.508
BASELINE_FLAG_RATE    = 0.143
THRESHOLD             = 0.5
GATE2_MI_MIN          = 0.01
GATE2_SPEARMAN_MIN    = 0.05
GATE3_RECALL_GAIN_MIN = 0.005
GATE3_RECALL_MIN      = BASELINE_FATAL_RECALL + GATE3_RECALL_GAIN_MIN  # 0.513

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

LOGREG_PARAMS = dict(
    C=1.0,
    class_weight="balanced",
    max_iter=1000,
    random_state=42,
    solver="lbfgs",
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
TRAIN_FEATURED = "data/splits/train_featured.parquet"
VAL_FEATURED   = "data/splits/val_featured.parquet"
CENTERLINE_CSV = "data/raw/road_centerline.csv"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def extract_lrsnumber(segment_id: str):
    """Extract LRSNUMBER from segment_id.
    Format: LRS_{LRSNUMBER}_{mile_bin} or GRID_{row}_{col}
    """
    if segment_id.startswith("LRS_"):
        parts = segment_id[4:].rsplit("_", 1)
        return parts[0] if parts else None
    return None


def build_funcl_lookup(centerline: pd.DataFrame) -> dict:
    """Build LRSNUMBER -> FUNCL lookup using min FUNCL deduplication rule.
    Min FUNCL = highest functional class (1=Interstate beats 7=Local).
    """
    lookup = (
        centerline.groupby("TWN_LR")["FUNCL"]
        .min()
        .to_dict()
    )
    return lookup


def add_funcl(df: pd.DataFrame, lookup: dict) -> pd.DataFrame:
    """Add functional_class column to windowed feature dataframe.
    Grid segments receive FUNCL = 0 (unclassified — no LRS identity).
    Unmatched LRS segments receive NaN (reported separately).
    """
    df = df.copy()
    df["lrsnumber"] = df["segment_id"].apply(extract_lrsnumber)
    df["functional_class"] = df["lrsnumber"].map(lookup)
    # Grid segments: no LRS, assign 0 (unclassified)
    grid_mask = df["segment_id"].str.startswith("GRID_")
    df.loc[grid_mask, "functional_class"] = 0
    return df


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
print("=" * 60)
print("GATE TEST — functional_class (FUNCL)")
print("=" * 60)

print("\nLoading data...")
train      = pd.read_parquet(TRAIN_FEATURED)
val        = pd.read_parquet(VAL_FEATURED)
centerline = pd.read_csv(CENTERLINE_CSV)

print(f"  Train windows: {len(train):,}  (fatal: {train['has_fatal'].sum():,})")
print(f"  Val windows:   {len(val):,}  (fatal: {val['has_fatal'].sum():,})")
print(f"  Centerline:    {len(centerline):,} segments")

# ---------------------------------------------------------------------------
# Build FUNCL lookup and add to train/val
# ---------------------------------------------------------------------------
print("\nBuilding FUNCL lookup (min rule)...")
lookup = build_funcl_lookup(centerline)
print(f"  Unique LRSNUMBER -> FUNCL mappings: {len(lookup):,}")

train = add_funcl(train, lookup)
val   = add_funcl(val, lookup)

# Coverage report
train_cov = train["functional_class"].notna().mean()
val_cov   = val["functional_class"].notna().mean()
train_fatal_cov = train[train["has_fatal"] == 1]["functional_class"].notna().mean()
val_fatal_cov   = val[val["has_fatal"] == 1]["functional_class"].notna().mean()

print(f"\n  Train coverage (all):   {train_cov:.2%}")
print(f"  Train coverage (fatal): {train_fatal_cov:.2%}")
print(f"  Val coverage (all):     {val_cov:.2%}")
print(f"  Val coverage (fatal):   {val_fatal_cov:.2%}")

# Null fill report
train_null = train["functional_class"].isna().sum()
val_null   = val["functional_class"].isna().sum()
print(f"\n  Remaining nulls after grid fill — train: {train_null:,}, val: {val_null:,}")
print(f"  Filling remaining nulls with 0 (unclassified)")
train["functional_class"] = train["functional_class"].fillna(0)
val["functional_class"]   = val["functional_class"].fillna(0)

# ---------------------------------------------------------------------------
# GATE 2 — Signal test
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("GATE 2 — SIGNAL TEST")
print("=" * 60)

# Mutual information
X_mi = train[["functional_class"]].values
y_mi = train["has_fatal"].values
mi   = mutual_info_classif(X_mi, y_mi, random_state=42)[0]
print(f"  Mutual information:     {mi:.6f}  (threshold: >{GATE2_MI_MIN})")

# Spearman correlation
rho, pval = spearmanr(train["functional_class"], train["has_fatal"])
print(f"  Spearman correlation:   {rho:.6f}  (p={pval:.4e})  (threshold: >{GATE2_SPEARMAN_MIN})")

gate2_pass = (mi > GATE2_MI_MIN) or (abs(rho) > GATE2_SPEARMAN_MIN)
print(f"\n  Gate 2 result:  {'PASS' if gate2_pass else 'FAIL'}")
print(f"  (MI > {GATE2_MI_MIN}: {mi > GATE2_MI_MIN}) OR "
      f"(|Spearman| > {GATE2_SPEARMAN_MIN}: {abs(rho) > GATE2_SPEARMAN_MIN})")

# FUNCL distribution in fatal vs non-fatal windows
print("\n  FUNCL mean — fatal windows:     "
      f"{train[train['has_fatal']==1]['functional_class'].mean():.3f}")
print(f"  FUNCL mean — non-fatal windows: "
      f"{train[train['has_fatal']==0]['functional_class'].mean():.3f}")

# ---------------------------------------------------------------------------
# GATE 3 — Incremental fatal recall
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("GATE 3 — INCREMENTAL FATAL RECALL")
print("=" * 60)
print(f"  Baseline (iter_001): fatal recall = {BASELINE_FATAL_RECALL}")
print(f"  Required minimum:    fatal recall >= {GATE3_RECALL_MIN}")

AUGMENTED_FEATURES = BASE_FEATURES + ["functional_class"]

X_train_base = train[BASE_FEATURES].values
X_train_aug  = train[AUGMENTED_FEATURES].values
y_train      = train["has_fatal"].values

X_val_base   = val[BASE_FEATURES].values
X_val_aug    = val[AUGMENTED_FEATURES].values
y_val        = val["has_fatal"].values

# Baseline model (8 features — reproduce iter_001)
print("\nTraining baseline model (8 features)...")
model_base = LogisticRegression(**LOGREG_PARAMS)
model_base.fit(X_train_base, y_train)
prob_base  = model_base.predict_proba(X_val_base)[:, 1]
pred_base  = (prob_base >= THRESHOLD).astype(int)

recall_base    = recall_score(y_val, pred_base, zero_division=0)
precision_base = precision_score(y_val, pred_base, zero_division=0)
flag_rate_base = pred_base.mean()
logloss_base   = log_loss(y_val, prob_base)

print(f"  Fatal recall:  {recall_base:.4f}")
print(f"  Precision:     {precision_base:.4f}")
print(f"  Flag rate:     {flag_rate_base:.4f}")
print(f"  Log loss:      {logloss_base:.4f}")

# Augmented model (8 features + functional_class)
print("\nTraining augmented model (8 features + functional_class)...")
model_aug = LogisticRegression(**LOGREG_PARAMS)
model_aug.fit(X_train_aug, y_train)
prob_aug  = model_aug.predict_proba(X_val_aug)[:, 1]
pred_aug  = (prob_aug >= THRESHOLD).astype(int)

recall_aug    = recall_score(y_val, pred_aug, zero_division=0)
precision_aug = precision_score(y_val, pred_aug, zero_division=0)
flag_rate_aug = pred_aug.mean()
logloss_aug   = log_loss(y_val, prob_aug)

print(f"  Fatal recall:  {recall_aug:.4f}")
print(f"  Precision:     {precision_aug:.4f}")
print(f"  Flag rate:     {flag_rate_aug:.4f}")
print(f"  Log loss:      {logloss_aug:.4f}")

# Incremental gain
recall_gain = recall_aug - recall_base
flag_delta  = flag_rate_aug - flag_rate_base

print(f"\n  Recall gain:   {recall_gain:+.4f}  (threshold: >={GATE3_RECALL_GAIN_MIN})")
print(f"  Flag delta:    {flag_delta:+.4f}")

gate3_pass = recall_aug >= GATE3_RECALL_MIN
print(f"\n  Gate 3 result: {'PASS' if gate3_pass else 'FAIL'}")

# Feature coefficients
print("\n  Augmented model coefficients:")
for fname, coef in zip(AUGMENTED_FEATURES, model_aug.coef_[0]):
    print(f"    {fname:<30}: {coef:+.4f}")

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("GATE TEST SUMMARY — functional_class")
print("=" * 60)
print(f"  Gate 1 (availability >= 95%):          PASS (98.72% all, 99.43% fatal)")
print(f"  Gate 2 (MI > {GATE2_MI_MIN} or Spearman > {GATE2_SPEARMAN_MIN}):  "
      f"{'PASS' if gate2_pass else 'FAIL'} (MI={mi:.4f}, Spearman={rho:.4f})")
print(f"  Gate 3 (recall gain >= {GATE3_RECALL_GAIN_MIN}):          "
      f"{'PASS' if gate3_pass else 'FAIL'} "
      f"({recall_base:.4f} -> {recall_aug:.4f}, gain={recall_gain:+.4f})")

all_pass = gate2_pass and gate3_pass
print(f"\n  FEATURE DECISION: {'ACCEPT' if all_pass else 'REJECT'}")
print(f"  Baseline fatal recall: {BASELINE_FATAL_RECALL}")
print(f"  Augmented fatal recall: {recall_aug:.4f}")
print(f"  Flag rate: {flag_rate_base:.4f} -> {flag_rate_aug:.4f}")

print("\nDone.")