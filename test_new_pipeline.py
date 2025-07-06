#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test script for the new rocket_transformer pipeline.
Quick validation before running full experiments.
"""

import sys
import logging
from pathlib import Path
import numpy as np
import pandas as pd
from rich.console import Console

# Add ml_pipeline to path
sys.path.append(str(Path(__file__).parent / "ml_pipeline"))

from data_loading import load_ecg_data
from preprocess import (
    filter_sinus_rhythm,
    create_pre_ecg_labels,
    create_outcome_labels,
    get_cohort_for_window_and_outcome,
)
from splitter import PatientLevelSplitter
from train import train_single_fold

console = Console()
logger = logging.getLogger(__name__)


def test_rocket_transformer():
    """Test the new rocket_transformer pipeline."""
    logging.basicConfig(level=logging.INFO)

    console.print("[bold green]Testing Rocket Transformer Pipeline[/bold green]")

    try:
        # Load and preprocess data
        console.print("[bold]Loading data...[/bold]")
        metadata, waveforms = load_ecg_data()

        console.print("[bold]Preprocessing data...[/bold]")
        sinus_df = filter_sinus_rhythm(metadata)
        processed_df = create_pre_ecg_labels(sinus_df)
        processed_df = create_outcome_labels(processed_df)

        # Get a small cohort for testing
        console.print("[bold]Getting test cohort...[/bold]")
        cohort_df, cohort_waveforms = get_cohort_for_window_and_outcome(
            processed_df, waveforms, "pre_ecg_1y", "af_recurrence_1y"
        )

        if len(cohort_df) == 0:
            console.print("[red]No data available for testing[/red]")
            return False

        console.print(
            f"Test cohort: {len(cohort_df)} ECGs, {cohort_df['mrn'].nunique()} patients"
        )

        # Create simple train/val split
        splitter = PatientLevelSplitter(random_state=42)
        splits = splitter.create_splits(
            cohort_df, cohort_waveforms, "af_recurrence_1y", use_cv=False, n_folds=3
        )

        # Test different configurations
        test_configs = [
            {
                "name": "rocket_transformer_basic",
                "model_name": "rocket_transformer",
                "use_oversampling": False,
                "use_sample_weights": False,
                "class_weight": None,
                "model_kwargs": {"num_kernels": 1000, "C": 1.0},
            },
            {
                "name": "rocket_transformer_class_weight",
                "model_name": "rocket_transformer",
                "use_oversampling": False,
                "use_sample_weights": False,
                "class_weight": "balanced",
                "model_kwargs": {"num_kernels": 1000, "C": 1.0},
            },
            {
                "name": "rocket_transformer_oversampling",
                "model_name": "rocket_transformer",
                "use_oversampling": True,
                "use_sample_weights": False,
                "class_weight": None,
                "model_kwargs": {"num_kernels": 1000, "C": 1.0},
            },
        ]

        for config in test_configs:
            console.print(f"\n[bold]Testing: {config['name']}[/bold]")

            try:
                pipeline, train_metrics, val_metrics = train_single_fold(
                    splits["train"]["metadata"],
                    splits["train"]["waveforms"],
                    splits["val"]["metadata"],
                    splits["val"]["waveforms"],
                    "af_recurrence_1y",
                    config["model_name"],
                    "lead1",  # Use single lead for faster testing
                    False,  # No augmentation for testing
                    config["use_oversampling"],
                    config["use_sample_weights"],
                    config["class_weight"],
                    42,
                    **config["model_kwargs"],
                )

                console.print(f"  ✓ Training completed successfully")
                console.print(f"  Train ROC-AUC: {train_metrics['roc_auc']:.3f}")
                console.print(f"  Val ROC-AUC: {val_metrics['roc_auc']:.3f}")
                console.print(f"  Train PR-AUC: {train_metrics['pr_auc']:.3f}")
                console.print(f"  Val PR-AUC: {val_metrics['pr_auc']:.3f}")

                # Check if we're getting reasonable predictions
                if val_metrics["roc_auc"] > 0.4 and val_metrics["roc_auc"] < 1.0:
                    console.print(f"  ✓ ROC-AUC in reasonable range")
                else:
                    console.print(f"  ⚠ ROC-AUC outside expected range")

            except Exception as e:
                console.print(f"  ✗ Failed: {str(e)}")
                logger.exception(f"Test {config['name']} failed")
                return False

        console.print("\n[bold green]All tests passed! ✓[/bold green]")
        return True

    except Exception as e:
        console.print(f"[red]Test failed: {str(e)}[/red]")
        logger.exception("Pipeline test failed")
        return False


def test_class_balance_script():
    """Test the class balance analysis script."""
    console.print("\n[bold]Testing class balance analysis...[/bold]")

    try:
        import subprocess

        result = subprocess.run(
            ["python", "scripts/analyse_class_balance.py"],
            capture_output=True,
            text=True,
            timeout=300,  # 5 minute timeout
        )

        if result.returncode == 0:
            console.print("  ✓ Class balance analysis completed successfully")
            return True
        else:
            console.print(f"  ✗ Class balance analysis failed: {result.stderr}")
            return False

    except Exception as e:
        console.print(f"  ✗ Could not run class balance analysis: {str(e)}")
        return False


if __name__ == "__main__":
    success = True

    # Test the new pipeline
    success &= test_rocket_transformer()

    # Test class balance script
    success &= test_class_balance_script()

    if success:
        console.print(
            "\n[bold green]🎉 All tests passed! Ready for full experiments.[/bold green]"
        )
        sys.exit(0)
    else:
        console.print(
            "\n[bold red]❌ Some tests failed. Please fix issues before proceeding.[/bold red]"
        )
        sys.exit(1)
