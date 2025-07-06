#!/usr/bin/env python
"""
Test optimal production settings from hyperparameter search.
"""

import subprocess
import sys
from pathlib import Path


def run_test():
    """Run production test with optimal settings."""
    print("🚀 TESTING OPTIMAL PRODUCTION SETTINGS")
    print("=" * 60)
    print("Configuration: C=0.0001, kernels=1000, L2 penalty")
    print("Expected: CV AUC ~0.528, Overfitting Gap ~0.353")
    print()

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
        "--augment",
        "--oversample",
        "--class-weight",
        "balanced",
        "--C",
        "0.0001",
        "--num-kernels",
        "1000",
        "--penalty",
        "l2",
        "--cv",
        "--n-folds",
        "3",
        "--results-dir",
        "test_optimal_results",
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)

        print("STDOUT:")
        print(result.stdout)
        print("\nSTDERR:")
        print(result.stderr)
        print(f"\nReturn code: {result.returncode}")

        if result.returncode == 0:
            print("\n✅ TEST SUCCESSFUL!")
            print("Check test_optimal_results/ for detailed results")
        else:
            print("\n❌ TEST FAILED!")

    except subprocess.TimeoutExpired:
        print("\n⏰ TEST TIMED OUT (10 minutes)")
    except Exception as e:
        print(f"\n💥 ERROR: {e}")


if __name__ == "__main__":
    run_test()
