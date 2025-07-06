#!/usr/bin/env python3
"""
Hyperparameter optimization for AF recurrence prediction.
Uses RandomizedSearchCV to find optimal parameters.
"""

import subprocess
import sys
from pathlib import Path
import json
import time
import numpy as np
from sklearn.model_selection import ParameterGrid


def run_single_config(config, base_dir="hp_search_results"):
    """Run a single hyperparameter configuration."""
    timestamp = int(time.time() * 1000)  # Unique timestamp

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
        "3",
        "--class-weight",
        "balanced",
        "--results-dir",
        f"{base_dir}/config_{timestamp}",
        "--random-state",
        "42",
    ]

    # Add configuration parameters
    for key, value in config.items():
        if key == "augmentation" and value:
            cmd.append("--augment")
        elif key == "oversampling" and value:
            cmd.append("--oversample")
        elif key in ["C", "num_kernels", "penalty"]:
            cmd.extend([f"--{key.replace('_', '-')}", str(value)])

    print(
        f"🧪 Testing: C={config['C']}, kernels={config['num_kernels']}, penalty={config['penalty']}"
    )

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

        if result.returncode != 0:
            print(f"❌ FAILED: {result.stderr[:200]}")
            return None

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

                        # Calculate overfitting gap
                        train_aucs = [
                            fold["train_metrics"]["roc_auc"]
                            for fold in results["cv_results"]["individual_folds"]
                        ]
                        val_aucs = [
                            fold["val_metrics"]["roc_auc"]
                            for fold in results["cv_results"]["individual_folds"]
                        ]

                        avg_train_auc = np.mean(train_aucs)
                        overfitting_gap = avg_train_auc - cv_auc

                        return {
                            "config": config,
                            "cv_auc": cv_auc,
                            "cv_auc_std": cv_auc_std,
                            "holdout_auc": holdout_auc,
                            "train_auc": avg_train_auc,
                            "overfitting_gap": overfitting_gap,
                            "results_path": str(results_path),
                        }

        print("⚠️  Could not parse results")
        return None

    except subprocess.TimeoutExpired:
        print("⏰ TIMEOUT")
        return None
    except Exception as e:
        print(f"❌ ERROR: {str(e)}")
        return None


def main():
    """Run hyperparameter optimization."""
    print("🚀 HYPERPARAMETER OPTIMIZATION")
    print("=" * 60)

    # Define parameter grid
    param_grid = {
        "C": [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05],
        "num_kernels": [1000, 1500, 2000, 2500],
        "penalty": ["l2"],
        "augmentation": [True],
        "oversampling": [True],
    }

    # Generate all combinations
    configs = list(ParameterGrid(param_grid))
    print(f"Testing {len(configs)} configurations...")

    # Randomize order to avoid systematic bias
    np.random.seed(42)
    np.random.shuffle(configs)

    # Run experiments
    results = []
    best_score = 0
    best_config = None

    for i, config in enumerate(configs[:12]):  # Limit to 12 configs for speed
        print(f"\n[{i + 1}/{min(12, len(configs))}] ", end="")

        result = run_single_config(config)
        if result is not None:
            results.append(result)

            cv_auc = result["cv_auc"]
            gap = result["overfitting_gap"]

            # Score = CV AUC - penalty for overfitting
            score = cv_auc - max(0, gap - 0.2) * 0.5

            print(f"✅ CV AUC: {cv_auc:.3f}, Gap: {gap:.3f}, Score: {score:.3f}")

            if score > best_score:
                best_score = score
                best_config = result
                print("🏆 NEW BEST!")
        else:
            print("❌ Failed")

    # Summary
    print(f"\n{'=' * 60}")
    print("📊 HYPERPARAMETER OPTIMIZATION RESULTS")
    print(f"{'=' * 60}")

    if not results:
        print("❌ No successful experiments")
        return False

    # Sort by score
    scored_results = []
    for result in results:
        cv_auc = result["cv_auc"]
        gap = result["overfitting_gap"]
        score = cv_auc - max(0, gap - 0.2) * 0.5
        scored_results.append((score, result))

    scored_results.sort(reverse=True, key=lambda x: x[0])

    print("\n🏆 TOP 5 CONFIGURATIONS:")
    for i, (score, result) in enumerate(scored_results[:5]):
        config = result["config"]
        print(f"\n{i + 1}. Score: {score:.3f}")
        print(f"   C={config['C']}, kernels={config['num_kernels']}")
        print(f"   CV AUC: {result['cv_auc']:.3f} ± {result['cv_auc_std']:.3f}")
        print(f"   Holdout AUC: {result['holdout_auc']:.3f}")
        print(f"   Overfitting Gap: {result['overfitting_gap']:.3f}")

    # Best configuration
    if best_config:
        print(f"\n🎯 RECOMMENDED CONFIGURATION:")
        config = best_config["config"]
        print(f"   --C {config['C']}")
        print(f"   --num-kernels {config['num_kernels']}")
        print(f"   --penalty {config['penalty']}")
        print(f"   --class-weight balanced")
        if config["augmentation"]:
            print(f"   --augment")
        if config["oversampling"]:
            print(f"   --oversample")

        print(f"\n📈 Expected Performance:")
        print(f"   CV AUC: {best_config['cv_auc']:.3f}")
        print(f"   Overfitting Gap: {best_config['overfitting_gap']:.3f}")

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
