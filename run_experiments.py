#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Batch runner for AF recurrence prediction experiments.
Runs multiple experiments with different configurations.
"""

import os
import sys
import subprocess
import itertools
from pathlib import Path
from datetime import datetime
import pandas as pd
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn

console = Console()

# Experiment configurations
EXPERIMENT_CONFIGS = {
    # Three clinically-relevant windows only
    "pre_ecg_windows": ["pre_ecg_1y", "pre_ecg_3y", "pre_ecg_5y"],
    # All 12 outcome labels remain – performance is outcome-dependent
    "outcome_labels": [
        "af_recurrence_1y",
        "af_recurrence_3y",
        "af_recurrence_5y",
        "af_recurrence_any",
        "at_recurrence_1y",
        "at_recurrence_3y",
        "at_recurrence_5y",
        "at_recurrence_any",
        "af_at_recurrence_1y",
        "af_at_recurrence_3y",
        "af_at_recurrence_5y",
        "af_at_recurrence_any",
    ],
    # 🔥 Only the best-performing model for small data
    "models": ["rocket_transformer"],
    # 12-lead ECGs consistently outperform single-lead
    "lead_configs": ["all"],
    # Augmentation & oversampling were shown to hurt – fix to False
    "augmentation": [False],
    "oversampling": [False],
    # Patient bootstrapping delivered +32% ROC-AUC – always enable
    "bootstrap_patients": [True],
    # Hyper-params already tuned – keep single value
    "regularization": ["l2"],
}


def run_single_experiment(config, results_base_dir):
    """Run a single experiment with given configuration."""
    cmd = [
        "uv",
        "run",
        "python",
        "ml_pipeline/train.py",
        "--pre-ecg-window",
        config["pre_ecg_window"],
        "--outcome-label",
        config["outcome_label"],
        "--model",
        config["model"],
        "--leads",
        config["lead_config"],
        "--results-dir",
        results_base_dir,
        "--cv",  # Always use CV for comprehensive evaluation
        "--n-folds",
        "3",
    ]

    if config["augmentation"]:
        cmd.append("--augment")

    if config["oversampling"]:
        cmd.append("--oversample")

    if config["bootstrap_patients"]:
        cmd.append("--bootstrap-patients")

    # Add class weight and regularization for rocket_transformer
    if config["model"] == "rocket_transformer":
        cmd.extend(["--class-weight", "balanced"])
        cmd.extend(["--penalty", config["regularization"]])
        cmd.extend(["--C", "0.0001"])  # Optimal hyperparameter from search
        cmd.extend(["--num-kernels", "1000"])  # Optimal kernel count from search
        # Note: Augmentation and oversampling handled by config flags above

        # Enable calibration for rocket_transformer by default
        cmd.extend(["--enable-calibration"])
        cmd.extend(["--threshold-metric", "f1"])
        cmd.extend(["--calibration-method", "platt"])

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=3600
        )  # 1 hour timeout
        return {
            "config": config,
            "success": result.returncode == 0,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "returncode": result.returncode,
        }
    except subprocess.TimeoutExpired:
        return {
            "config": config,
            "success": False,
            "error": "Timeout (1 hour)",
            "returncode": -1,
        }
    except Exception as e:
        return {"config": config, "success": False, "error": str(e), "returncode": -1}


def generate_experiment_configs():
    """Generate all experiment configurations."""
    configs = []

    for (
        window,
        outcome,
        model,
        leads,
        aug,
        oversample,
        bootstrap,
        reg,
    ) in itertools.product(
        EXPERIMENT_CONFIGS["pre_ecg_windows"],
        EXPERIMENT_CONFIGS["outcome_labels"],
        EXPERIMENT_CONFIGS["models"],
        EXPERIMENT_CONFIGS["lead_configs"],
        EXPERIMENT_CONFIGS["augmentation"],
        EXPERIMENT_CONFIGS["oversampling"],
        EXPERIMENT_CONFIGS["bootstrap_patients"],
        EXPERIMENT_CONFIGS["regularization"],
    ):
        configs.append(
            {
                "pre_ecg_window": window,
                "outcome_label": outcome,
                "model": model,
                "lead_config": leads,
                "augmentation": aug,
                "oversampling": oversample,
                "bootstrap_patients": bootstrap,
                "regularization": reg,
            }
        )

    return configs


def filter_configs(configs, quick_run=False):
    """Filter configurations for testing purposes."""
    if quick_run:
        # For quick testing, run a subset
        filtered = []
        for config in configs:
            if (
                config["pre_ecg_window"] == "pre_ecg_1y"
                and config["outcome_label"]
                in ["af_recurrence_1y"]  # Focus on AF only first
                and config["model"] == "rocket_transformer"
                and config["lead_config"] == "all"  # Only all leads
            ):
                filtered.append(config)
        return filtered[:4]  # Just first 4 for quick test

    return configs


def create_experiment_summary(results):
    """Create a summary table of all experiments."""
    summary_data = []

    for result in results:
        config = result["config"]
        summary_data.append(
            {
                "Window": config["pre_ecg_window"],
                "Outcome": config["outcome_label"],
                "Model": config["model"],
                "Leads": config["lead_config"],
                "Augment": "Yes" if config["augmentation"] else "No",
                "Oversample": "Yes" if config["oversampling"] else "No",
                "Bootstrap": "Yes" if config["bootstrap_patients"] else "No",
                "Success": "Yes" if result["success"] else "No",
                "Error": result.get("error", ""),
            }
        )

    return pd.DataFrame(summary_data)


def load_completed_experiments(results_root: str) -> set:
    """Return a set of config keys that already finished successfully.

    The key is a tuple matching the same fields we generate later so we can
    de-duplicate work across runs. It scans *all* existing batch_* directories
    under the given `results_root` for `experiment_summary*.csv` files.
    """
    completed = set()
    root_path = Path(results_root)
    if not root_path.exists():
        return completed

    csv_files = root_path.glob("batch_*/experiment_summary*.csv")
    for csv_path in csv_files:
        try:
            df = pd.read_csv(csv_path)
        except Exception:
            # Corrupted or partial file – skip
            continue

        # Normalise optional columns that older runs might miss
        for col, default in [
            ("Augment", "No"),
            ("Oversample", "No"),
            ("Bootstrap", "No"),
        ]:
            if col not in df.columns:
                df[col] = default

        for _, row in df.iterrows():
            if str(row.get("Success", "")).lower() != "yes":
                continue  # Only finished runs
            key = (
                row["Window"],
                row["Outcome"],
                row["Model"],
                row["Leads"],
                row["Augment"].lower() == "yes",
                row["Oversample"].lower() == "yes",
                row["Bootstrap"].lower() == "yes",
            )
            completed.add(key)
    return completed


def main():
    """Main function to run all experiments."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Run batch AF recurrence prediction experiments (deduplicated)"
    )
    parser.add_argument(
        "--quick", action="store_true", help="Run quick test with subset of configs"
    )
    parser.add_argument(
        "--results-dir", default="batch_results", help="Base directory for results"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Print configs without running"
    )

    args = parser.parse_args()

    # Load completed experiments so we don't redo them
    completed_keys = load_completed_experiments(args.results_dir)

    # Generate experiment configurations
    all_configs = generate_experiment_configs()
    configs = filter_configs(all_configs, args.quick)

    # De-duplicate against past successful runs
    pre_filter_count = len(configs)
    configs = [
        cfg
        for cfg in configs
        if (
            cfg["pre_ecg_window"],
            cfg["outcome_label"],
            cfg["model"],
            cfg["lead_config"],
            cfg["augmentation"],
            cfg["oversampling"],
            cfg["bootstrap_patients"],
        )
        not in completed_keys
    ]

    skipped = pre_filter_count - len(configs)
    console.print(
        f"[bold]{skipped} configs skipped because they were already completed successfully[/bold]"
    )
    console.print(f"[bold]Running {len(configs)} new experiment configurations[/bold]")

    if args.dry_run or not configs:
        # If --dry-run OR nothing new to run, just list what would be executed
        table = Table(title="Experiment Configurations (new)")
        table.add_column("Window")
        table.add_column("Outcome")
        table.add_column("Model")
        table.add_column("Leads")
        table.add_column("Bootstrap")

        for cfg in configs:
            table.add_row(
                cfg["pre_ecg_window"],
                cfg["outcome_label"],
                cfg["model"],
                cfg["lead_config"],
                "Yes" if cfg["bootstrap_patients"] else "No",
            )
        console.print(table)
        return

    # Create results directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path(args.results_dir) / f"batch_{timestamp}"
    results_dir.mkdir(parents=True, exist_ok=True)

    console.print(f"[bold]Starting batch experiments[/bold]")
    console.print(f"Results will be saved to: {results_dir}")

    # Run experiments
    results = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Running experiments...", total=len(configs))

        for i, config in enumerate(configs):
            progress.update(
                task,
                description=f"[{i + 1}/{len(configs)}] {config['pre_ecg_window']} -> {config['outcome_label']} ({config['model']}, {config['lead_config']})",
            )

            result = run_single_experiment(config, str(results_dir))
            results.append(result)

            if result["success"]:
                console.print(
                    f"[green]✓[/green] Experiment {i + 1} completed successfully"
                )
            else:
                console.print(
                    f"[red]✗[/red] Experiment {i + 1} failed: {result.get('error', 'Unknown error')}"
                )

            # Save incremental summary after each experiment (for crash recovery)
            if (i + 1) % 3 == 0 or i == len(
                configs
            ) - 1:  # Every 3 experiments or at the end
                summary_df = create_experiment_summary(results)
                summary_path = results_dir / "experiment_summary_partial.csv"
                summary_df.to_csv(summary_path, index=False)

            progress.advance(task)

    # Create summary
    summary_df = create_experiment_summary(results)
    summary_path = results_dir / "experiment_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    # Print summary
    console.print("\n[bold]Experiment Summary[/bold]")

    success_count = sum(1 for r in results if r["success"])
    console.print(f"Successful experiments: {success_count}/{len(results)}")

    if success_count < len(results):
        console.print("\n[red]Failed experiments:[/red]")
        for result in results:
            if not result["success"]:
                config = result["config"]
                console.print(
                    f"  {config['pre_ecg_window']} -> {config['outcome_label']} ({config['model']}): {result.get('error', 'Unknown error')}"
                )

    console.print(f"\nDetailed summary saved to: {summary_path}")
    console.print(f"All results saved to: {results_dir}")


if __name__ == "__main__":
    main()
