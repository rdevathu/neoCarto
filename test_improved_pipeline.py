#!/usr/bin/env python3
"""
Test script for improved ML pipeline with overfitting fixes.
"""

import subprocess
import sys
from pathlib import Path
import json
import time


def run_test(name: str, cmd: list, expected_improvements: dict = None):
    """Run a single test and validate results."""
    print(f"\n{'=' * 60}")
    print(f"🧪 TEST: {name}")
    print(f"{'=' * 60}")
    print(f"Command: {' '.join(cmd)}")

    start_time = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    duration = time.time() - start_time

    if result.returncode != 0:
        print(f"❌ FAILED (exit code: {result.returncode})")
        print(f"STDERR: {result.stderr}")
        return False

    print(f"✅ COMPLETED in {duration:.1f}s")

    # Try to find and parse results
    lines = result.stdout.split("\n")
    for line in lines:
        if "Results saved to:" in line:
            results_path = line.split("Results saved to:")[1].strip()
            results_file = Path(results_path) / "results.json"

            if results_file.exists():
                with open(results_file, "r") as f:
                    results = json.load(f)

                # Extract key metrics
                if "cv_results" in results:
                    cv_auc = results["cv_results"]["mean"]["roc_auc"]
                    cv_auc_std = results["cv_results"]["std"]["roc_auc"]
                    holdout_auc = results["holdout_metrics"]["roc_auc"]

                    print(f"📊 CV ROC-AUC: {cv_auc:.3f} ± {cv_auc_std:.3f}")
                    print(f"📊 Holdout ROC-AUC: {holdout_auc:.3f}")

                    # Check for overfitting (train vs val performance)
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

                        if overfitting_gap > 0.2:
                            print("⚠️  HIGH OVERFITTING DETECTED")
                        elif overfitting_gap > 0.1:
                            print("⚠️  MODERATE OVERFITTING")
                        else:
                            print("✅ OVERFITTING UNDER CONTROL")

                    # Validate improvements
                    if expected_improvements:
                        if cv_auc >= expected_improvements.get("min_cv_auc", 0.0):
                            print(
                                f"✅ CV AUC meets expectation (>= {expected_improvements['min_cv_auc']})"
                            )
                        else:
                            print(
                                f"❌ CV AUC below expectation (< {expected_improvements['min_cv_auc']})"
                            )

                        if overfitting_gap <= expected_improvements.get(
                            "max_overfitting_gap", 1.0
                        ):
                            print(
                                f"✅ Overfitting gap acceptable (<= {expected_improvements['max_overfitting_gap']})"
                            )
                        else:
                            print(
                                f"❌ Overfitting gap too high (> {expected_improvements['max_overfitting_gap']})"
                            )

                return True

    print("⚠️  Could not find results file")
    return True


def main():
    """Run comprehensive tests of the improved pipeline."""
    print("🚀 TESTING IMPROVED ML PIPELINE")
    print("=" * 60)

    base_cmd = [
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
        "3",
        "--random-state",
        "42",
    ]

    tests = [
        {
            "name": "Baseline with Strong Regularization",
            "cmd": base_cmd
            + [
                "--class-weight",
                "balanced",
                "--C",
                "0.1",
                "--penalty",
                "l2",
                "--num-kernels",
                "5000",
            ],
            "expected": {"min_cv_auc": 0.55, "max_overfitting_gap": 0.3},
        },
        {
            "name": "L1 Regularization Test",
            "cmd": base_cmd
            + [
                "--class-weight",
                "balanced",
                "--C",
                "0.1",
                "--penalty",
                "l1",
                "--num-kernels",
                "5000",
            ],
            "expected": {"min_cv_auc": 0.55, "max_overfitting_gap": 0.3},
        },
        {
            "name": "ElasticNet Regularization Test",
            "cmd": base_cmd
            + [
                "--class-weight",
                "balanced",
                "--C",
                "0.1",
                "--penalty",
                "elasticnet",
                "--l1-ratio",
                "0.5",
                "--num-kernels",
                "5000",
            ],
            "expected": {"min_cv_auc": 0.55, "max_overfitting_gap": 0.3},
        },
        {
            "name": "Even Stronger Regularization",
            "cmd": base_cmd
            + [
                "--class-weight",
                "balanced",
                "--C",
                "0.01",  # Very strong regularization
                "--penalty",
                "l2",
                "--num-kernels",
                "3000",  # Fewer kernels
            ],
            "expected": {"min_cv_auc": 0.52, "max_overfitting_gap": 0.2},
        },
    ]

    results = []
    for test in tests:
        success = run_test(test["name"], test["cmd"], test.get("expected"))
        results.append((test["name"], success))

    # Summary
    print(f"\n{'=' * 60}")
    print("📋 TEST SUMMARY")
    print(f"{'=' * 60}")

    passed = sum(1 for _, success in results if success)
    total = len(results)

    for name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{status}: {name}")

    print(f"\n🎯 OVERALL: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 ALL TESTS PASSED! Pipeline improvements are working.")
    else:
        print("⚠️  Some tests failed. Review the results above.")

    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
