#!/usr/bin/env python3
"""
Test script for extreme regularization to combat severe overfitting.
"""

import subprocess
import sys
from pathlib import Path
import json
import time


def run_extreme_test(name: str, C: float, num_kernels: int, penalty: str = "l2"):
    """Run a single test with extreme regularization."""
    print(f"\n{'=' * 60}")
    print(f"🧪 EXTREME TEST: {name}")
    print(f"{'=' * 60}")

    cmd = [
        "uv",
        "run",
        "python",
        "ml_pipeline/train.py",
        "--pre-ecg-window",
        "pre_ecg_1y",
        "--outcome-label",
        "af_recurrence_1y",
        "--model",
        "rocket_transformer",
        "--leads",
        "all",
        "--cv",
        "--n-folds",
        "3",  # Larger folds
        "--class-weight",
        "balanced",
        "--C",
        str(C),
        "--penalty",
        penalty,
        "--num-kernels",
        str(num_kernels),
        "--random-state",
        "42",
    ]

    print(f"Command: {' '.join(cmd)}")

    start_time = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    duration = time.time() - start_time

    if result.returncode != 0:
        print(f"❌ FAILED (exit code: {result.returncode})")
        print(f"STDERR: {result.stderr}")
        return False

    print(f"✅ COMPLETED in {duration:.1f}s")

    # Parse results
    lines = result.stdout.split("\n")
    for line in lines:
        if "Results saved to:" in line:
            results_path = line.split("Results saved to:")[1].strip()
            results_file = Path(results_path) / "results.json"

            if results_file.exists():
                with open(results_file, "r") as f:
                    results = json.load(f)

                if "cv_results" in results:
                    cv_auc = results["cv_results"]["mean"]["roc_auc"]
                    cv_auc_std = results["cv_results"]["std"]["roc_auc"]
                    holdout_auc = results["holdout_metrics"]["roc_auc"]

                    print(f"📊 CV ROC-AUC: {cv_auc:.3f} ± {cv_auc_std:.3f}")
                    print(f"📊 Holdout ROC-AUC: {holdout_auc:.3f}")

                    # Check overfitting
                    if "individual_folds" in results["cv_results"]:
                        train_aucs = [
                            fold["train_metrics"]["roc_auc"]
                            for fold in results["cv_results"]["individual_folds"]
                        ]
                        val_aucs = [
                            fold["val_metrics"]["roc_auc"]
                            for fold in results["cv_results"]["individual_folds"]
                        ]

                        avg_train_auc = sum(train_aucs) / len(train_aucs)
                        avg_val_auc = sum(val_aucs) / len(val_aucs)
                        overfitting_gap = avg_train_auc - avg_val_auc

                        print(f"🔍 Train AUC: {avg_train_auc:.3f}")
                        print(f"🔍 Val AUC: {avg_val_auc:.3f}")
                        print(f"🔍 Overfitting Gap: {overfitting_gap:.3f}")

                        if overfitting_gap < 0.1:
                            print("🎉 OVERFITTING UNDER CONTROL!")
                        elif overfitting_gap < 0.2:
                            print("✅ MODERATE OVERFITTING")
                        elif overfitting_gap < 0.3:
                            print("⚠️  STILL SOME OVERFITTING")
                        else:
                            print("❌ SEVERE OVERFITTING PERSISTS")

                        return {
                            "cv_auc": cv_auc,
                            "holdout_auc": holdout_auc,
                            "overfitting_gap": overfitting_gap,
                            "train_auc": avg_train_auc,
                        }

    print("⚠️  Could not find results file")
    return None


def main():
    """Run extreme regularization tests."""
    print("🚀 TESTING EXTREME REGULARIZATION")
    print("=" * 60)
    print("Goal: Reduce overfitting gap to < 0.2")

    tests = [
        {
            "name": "Ultra Strong L2 (C=0.001)",
            "C": 0.001,
            "num_kernels": 2000,
            "penalty": "l2",
        },
        {
            "name": "Extreme L2 (C=0.0001)",
            "C": 0.0001,
            "num_kernels": 2000,
            "penalty": "l2",
        },
        {
            "name": "Ultra Strong L1 (C=0.001)",
            "C": 0.001,
            "num_kernels": 2000,
            "penalty": "l1",
        },
        {
            "name": "Minimal Kernels (1000)",
            "C": 0.01,
            "num_kernels": 1000,
            "penalty": "l2",
        },
    ]

    results = []
    for test in tests:
        result = run_extreme_test(
            test["name"], test["C"], test["num_kernels"], test["penalty"]
        )
        results.append((test["name"], result))

    # Summary
    print(f"\n{'=' * 60}")
    print("📋 EXTREME REGULARIZATION SUMMARY")
    print(f"{'=' * 60}")

    best_result = None
    best_gap = float("inf")

    for name, result in results:
        if result:
            gap = result["overfitting_gap"]
            cv_auc = result["cv_auc"]
            train_auc = result["train_auc"]

            status = (
                "🎉 EXCELLENT"
                if gap < 0.1
                else "✅ GOOD"
                if gap < 0.2
                else "⚠️ MODERATE"
                if gap < 0.3
                else "❌ POOR"
            )
            print(f"{status}: {name}")
            print(
                f"   CV AUC: {cv_auc:.3f}, Train AUC: {train_auc:.3f}, Gap: {gap:.3f}"
            )

            if gap < best_gap:
                best_gap = gap
                best_result = (name, result)

    if best_result:
        print(f"\n🏆 BEST CONFIGURATION: {best_result[0]}")
        print(f"   Overfitting Gap: {best_result[1]['overfitting_gap']:.3f}")
        print(f"   CV AUC: {best_result[1]['cv_auc']:.3f}")

    return best_result is not None


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
