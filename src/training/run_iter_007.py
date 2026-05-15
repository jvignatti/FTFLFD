"""
Iteration 007 — RF class weight sweep.

One hyperparameter being tuned: class_weight ratio.
Tests 20x, 25x, 30x, 35x, 40x, 50x to find the sweet spot
between 10x (too conservative) and 71x (too aggressive).

Constraints to satisfy simultaneously:
  - Fatal recall >= 0.50
  - Precision >= 0.05
  - Flag rate <= 0.30
  - Gen gap <= 0.10

Best model is selected by fatal recall among feasible candidates.
"""

import pandas as pd
import numpy as np
import json
import yaml
import joblib
from pathlib import Path
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import recall_score, precision_score, f1_score, confusion_matrix

REPO_ROOT = Path(__file__).resolve().parents[2]
SPLITS_DIR = REPO_ROOT / "data" / "splits"
EXPERIMENTS_DIR = REPO_ROOT / "experiments"

FEATURE_COLUMNS = [
    "segment_crash_rate",
    "segment_fatal_rate",
    "segment_injury_rate",
    "segment_pdo_rate",
    "month",
    "is_winter",
    "is_weekend_heavy",
    "segment_type_is_grid",
]

ITER_ID = "iter_007"
WEIGHTS_TO_TEST = [20, 25, 30, 35, 40, 50]


def evaluate_model(model, X, y, featured_df, split_name):
    y_pred = model.predict(X)
    combined_recall = recall_score(y, y_pred, zero_division=0)
    combined_precision = precision_score(y, y_pred, zero_division=0)
    combined_f1 = f1_score(y, y_pred, zero_division=0)
    flag_rate = y_pred.mean()

    fatal_mask = featured_df["has_fatal"] == 1
    injury_only_mask = (featured_df["has_injury"] == 1) & (featured_df["has_fatal"] == 0)
    fatal_recall = float(y_pred[fatal_mask.values].mean()) if fatal_mask.sum() > 0 else None
    injury_recall = float(y_pred[injury_only_mask.values].mean()) if injury_only_mask.sum() > 0 else None

    tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()

    return {
        "split": split_name,
        "recall_combined": round(combined_recall, 4),
        "recall_fatal": round(float(fatal_recall), 4) if fatal_recall is not None else None,
        "recall_injury": round(float(injury_recall), 4) if injury_recall is not None else None,
        "precision": round(combined_precision, 4),
        "f1": round(combined_f1, 4),
        "flag_rate": round(flag_rate, 4),
        "true_positives": int(tp),
        "false_positives": int(fp),
        "true_negatives": int(tn),
        "false_negatives": int(fn),
        "support_positive": int(y.sum()),
        "support_total": int(len(y)),
        "fatal_windows_total": int(fatal_mask.sum()),
    }


def check_constraints(val_metrics, gen_gap):
    checks = {
        "fatal_recall_ge_050": val_metrics["recall_fatal"] >= 0.50 if val_metrics["recall_fatal"] else False,
        "precision_ge_005": val_metrics["precision"] >= 0.05,
        "flag_rate_le_030": val_metrics["flag_rate"] <= 0.30,
        "gen_gap_le_010": gen_gap <= 0.10,
    }
    all_pass = all(checks.values())
    return checks, all_pass


def main():
    print("=" * 60)
    print("ITERATION 007 — RF CLASS WEIGHT SWEEP")
    print(f"Testing weights: {WEIGHTS_TO_TEST}")
    print("Selecting best feasible model by fatal recall")
    print("=" * 60)

    print("\nLoading featured splits...")
    train_df = pd.read_parquet(SPLITS_DIR / "train_featured.parquet")
    val_df = pd.read_parquet(SPLITS_DIR / "val_featured.parquet")

    X_train = train_df[FEATURE_COLUMNS]
    y_train = train_df["label"]
    X_val = val_df[FEATURE_COLUMNS]
    y_val = val_df["label"]

    print(f"  Train: {X_train.shape[0]:,} rows")
    print(f"  Val:   {X_val.shape[0]:,} rows")

    # Sweep
    all_results = []
    best_feasible = None
    best_fatal_recall = 0

    for weight in WEIGHTS_TO_TEST:
        print(f"\n{'='*60}")
        print(f"  Training with class_weight={{0:1, 1:{weight}}}...")

        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_leaf=50,
            class_weight={0: 1, 1: weight},
            random_state=42,
            n_jobs=-1,
        )
        model.fit(X_train, y_train)

        train_metrics = evaluate_model(model, X_train, y_train, train_df, "train")
        val_metrics = evaluate_model(model, X_val, y_val, val_df, "val")

        gen_gap = round(train_metrics["recall_combined"] - val_metrics["recall_combined"], 4)
        constraints, all_pass = check_constraints(val_metrics, gen_gap)

        status = "FEASIBLE" if all_pass else "VIOLATED"

        print(f"  Weight: {weight}x | Status: {status}")
        print(f"    Fatal recall:    {val_metrics['recall_fatal']}")
        print(f"    Combined recall: {val_metrics['recall_combined']}")
        print(f"    Precision:       {val_metrics['precision']}")
        print(f"    Flag rate:       {val_metrics['flag_rate']}")
        print(f"    F1:              {val_metrics['f1']}")
        print(f"    Gen gap:         {gen_gap}")

        if not all_pass:
            failed = [k for k, v in constraints.items() if not v]
            print(f"    Violated: {', '.join(failed)}")

        entry = {
            "weight": weight,
            "status": status,
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
            "gen_gap": gen_gap,
            "constraints": {k: bool(v) for k, v in constraints.items()},
            "all_pass": all_pass,
        }
        all_results.append(entry)

        if all_pass and val_metrics["recall_fatal"] is not None:
            if val_metrics["recall_fatal"] > best_fatal_recall:
                best_fatal_recall = val_metrics["recall_fatal"]
                best_feasible = entry
                best_model = model

    # Summary
    print(f"\n{'='*60}")
    print("SWEEP SUMMARY")
    print(f"{'='*60}")
    print(f"\n  {'Weight':>7s} {'Status':>10s} {'Fatal_R':>8s} {'Comb_R':>8s} {'Prec':>8s} {'Flag%':>8s} {'GenGap':>8s}")
    print(f"  {'-'*60}")
    for r in all_results:
        vm = r["val_metrics"]
        print(f"  {r['weight']:>5d}x  {r['status']:>10s} {vm['recall_fatal']:>8.4f} "
              f"{vm['recall_combined']:>8.4f} {vm['precision']:>8.4f} "
              f"{vm['flag_rate']:>8.4f} {r['gen_gap']:>8.4f}")

    print(f"\n  --- REFERENCE ---")
    print(f"  {'LR001':>7s} {'ACCEPTED':>10s} {'0.6600':>8s} {'0.6330':>8s} {'0.0521':>8s} {'0.1430':>8s} {'0.0642':>8s}")
    print(f"  {'TopK15':>7s} {'BASELINE':>10s} {'0.1580':>8s} {'0.3797':>8s} {'0.0720':>8s} {'0.1530':>8s} {'---':>8s}")

    if best_feasible is not None:
        w = best_feasible["weight"]
        vm = best_feasible["val_metrics"]
        print(f"\n  BEST FEASIBLE: weight={w}x")
        print(f"    Fatal recall:    {vm['recall_fatal']}")
        print(f"    Combined recall: {vm['recall_combined']}")
        print(f"    Precision:       {vm['precision']}")
        print(f"    Flag rate:       {vm['flag_rate']}")
        print(f"    F1:              {vm['f1']}")
        print(f"    Gen gap:         {best_feasible['gen_gap']}")

        # Does it beat LogReg iter_001?
        beats_lr_fatal = vm["recall_fatal"] > 0.660
        beats_lr_combined = vm["recall_combined"] > 0.633
        beats_lr_flag = vm["flag_rate"] <= 0.143
        print(f"\n    Beats LR iter_001 fatal recall (0.660)?    {'YES' if beats_lr_fatal else 'NO'}")
        print(f"    Beats LR iter_001 combined recall (0.633)? {'YES' if beats_lr_combined else 'NO'}")
        print(f"    Beats LR iter_001 flag rate (0.143)?       {'YES' if beats_lr_flag else 'NO'}")

        # Save best model
        iter_dir = EXPERIMENTS_DIR / ITER_ID
        iter_dir.mkdir(parents=True, exist_ok=True)

        all_metrics = {
            "_schema_version": "1.0",
            "iter_id": ITER_ID,
            "date": datetime.now().strftime("%Y-%m-%d"),
            "train": best_feasible["train_metrics"],
            "val": vm,
            "benchmark": {"set": None, "audit_only": True},
            "diagnostics": {
                "generalization_gap": best_feasible["gen_gap"],
                "overfitting_warning": bool(best_feasible["gen_gap"] > 0.10),
                "fatal_recall_floor_check": bool(vm["recall_fatal"] >= 0.50) if vm["recall_fatal"] is not None else None,
            },
            "sweep_results": [
                {
                    "weight": r["weight"],
                    "status": r["status"],
                    "fatal_recall": r["val_metrics"]["recall_fatal"],
                    "combined_recall": r["val_metrics"]["recall_combined"],
                    "precision": r["val_metrics"]["precision"],
                    "flag_rate": r["val_metrics"]["flag_rate"],
                    "gen_gap": r["gen_gap"],
                }
                for r in all_results
            ],
        }
        with open(iter_dir / "metrics.json", "w") as f:
            json.dump(all_metrics, f, indent=2)

        config_snapshot = {
            "iter_id": ITER_ID,
            "date": datetime.now().strftime("%Y-%m-%d"),
            "model_class": "RandomForestClassifier",
            "change": f"class_weight sweep — best feasible: {{0:1, 1:{w}}}",
            "hyperparameters": {
                "n_estimators": 200,
                "max_depth": 10,
                "min_samples_leaf": 50,
                "class_weight": f"{{0: 1, 1: {w}}}",
                "random_state": 42,
                "n_jobs": -1,
            },
            "random_seed": 42,
            "feature_set_version": "1.1",
            "active_features": FEATURE_COLUMNS,
            "threshold": 0.5,
            "splits_config_version": "1.1",
            "thresholds_config_version": "1.0",
            "predecessor": "iter_004",
            "weights_tested": WEIGHTS_TO_TEST,
        }
        with open(iter_dir / "config_snapshot.yaml", "w") as f:
            yaml.dump(config_snapshot, f, default_flow_style=False)

        joblib.dump(best_model, iter_dir / "model.pkl")

        importance = dict(zip(
            FEATURE_COLUMNS,
            [round(float(i), 4) for i in best_model.feature_importances_]
        ))
        pd.DataFrame(
            list(importance.items()),
            columns=["feature", "gini_importance"]
        ).sort_values("gini_importance", ascending=False).to_csv(
            iter_dir / "feature_importance.csv", index=False
        )

        print(f"\n  Best model saved to {iter_dir}")
    else:
        print("\n  NO FEASIBLE MODEL FOUND across all weights tested.")
        print("  Consider: different hyperparameters, different features, or different model class.")

    print("=" * 60)


if __name__ == "__main__":
    main()