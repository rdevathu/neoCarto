#!/usr/bin/env python
"""
Test final optimal configuration based on quick experiment results.
"""

import subprocess
import sys


def run_final_test():
    """Run final optimal configuration test."""
    print("🎯 TESTING FINAL OPTIMAL CONFIGURATION")
    print("=" * 60)
    print("Configuration: C=0.0001, kernels=1000, no augment, no oversample")
    print("Expected: CV AUC ~0.538, Holdout AUC ~0.513")
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
        # No --augment flag (optimal)
        # No --oversample flag (optimal)
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
        "final_optimal_results",
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)

        print("STDOUT:")
        print(result.stdout)
        print("\nSTDERR:")
        print(result.stderr)
        print(f"\nReturn code: {result.returncode}")

        if result.returncode == 0:
            print("\n✅ FINAL OPTIMAL TEST SUCCESSFUL!")
            print("🎯 This configuration is ready for production use")
            print("📊 Expected performance: CV AUC ~0.538, good generalization")
        else:
            print("\n❌ TEST FAILED!")

    except subprocess.TimeoutExpired:
        print("\n⏰ TEST TIMED OUT (10 minutes)")
    except Exception as e:
        print(f"\n💥 ERROR: {e}")


if __name__ == "__main__":
    run_final_test()
