#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Main training script for AF recurrence prediction.
Orchestrates the entire ML pipeline with CLI interface.
"""

import os
import sys
import json
import logging
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    classification_report,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline as ImbPipeline
import joblib
from rich.console import Console
from rich.table import Table
from rich.progress import track

# Import our modules
from data_loading import load_ecg_data, get_data_summary
from preprocess import (
    filter_sinus_rhythm,
    create_pre_ecg_labels,
    create_outcome_labels,
    get_cohort_for_window_and_outcome,
    get_cohort_summary,
)
from splitter import PatientLevelSplitter
from augment import ECGAugmenter

console = Console()
logger = logging.getLogger(__name__)


def setup_logging(log_level: str = "INFO") -> None:
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def create_results_dir(base_dir: str = "results") -> Path:
    """Create timestamped results directory."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path(base_dir) / timestamp
    results_dir.mkdir(parents=True, exist_ok=True)
    return results_dir


def get_model(model_name: str, n_features: int, random_state: int = 42):
    """
    Get model instance based on name.

    Args:
        model_name: Name of the model ('rocket' or 'resnet')
        n_features: Number of features (for ResNet)
        random_state: Random seed

    Returns:
        Model instance
    """
    if model_name == "rocket":
        try:
            from sktime.classification.kernel_based import RocketClassifier

            return RocketClassifier(random_state=random_state)
        except ImportError:
            console.print(
                "[red]Error: sktime not installed or RocketClassifier not available[/red]"
            )
            raise

    elif model_name == "resnet":
        try:
            from sktime.classification.deep_learning.resnet import ResNetClassifier

            return ResNetClassifier(random_state=random_state, n_epochs=50)
        except ImportError:
            console.print(
                "[red]Error: sktime-dl not installed or ResNetClassifier not available[/red]"
            )
            console.print("[yellow]Falling back to RocketClassifier[/yellow]")
            from sktime.classification.kernel_based import RocketClassifier

            return RocketClassifier(random_state=random_state)

    else:
        raise ValueError(f"Unknown model: {model_name}")


def create_pipeline(
    model_name: str,
    n_features: int,
    use_oversampling: bool = True,
    random_state: int = 42,
) -> Pipeline:
    """
    Create ML pipeline with preprocessing and model.

    Args:
        model_name: Name of the model
        n_features: Number of features
        use_oversampling: Whether to use oversampling for class imbalance
        random_state: Random seed

    Returns:
        Sklearn pipeline
    """
    model = get_model(model_name, n_features, random_state)

    # Time series classifiers (ROCKET, ResNet) work with 3D data and cannot use
    # traditional oversampling which expects 2D data. We skip oversampling for these.
    if model_name in ["rocket", "resnet"] and use_oversampling:
        logger.warning(
            f"Oversampling not supported with {model_name} (3D time series data) - skipping oversampling"
        )
        use_oversampling = False

    steps = []

    # Add oversampling if requested and supported
    if use_oversampling:
        steps.append(("oversampler", RandomOverSampler(random_state=random_state)))

    # Add model
    steps.append(("classifier", model))

    if use_oversampling:
        return ImbPipeline(steps)
    else:
        return Pipeline(steps)


def evaluate_model(
    y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray
) -> Dict[str, float]:
    """
    Evaluate model performance.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_prob: Predicted probabilities

    Returns:
        Dictionary of metrics
    """
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="binary", zero_division=0),
        "recall": recall_score(y_true, y_pred, average="binary", zero_division=0),
        "f1": f1_score(y_true, y_pred, average="binary", zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.0,
        "pr_auc": average_precision_score(y_true, y_prob)
        if len(np.unique(y_true)) > 1
        else 0.0,
    }

    return metrics


def train_single_fold(
    train_metadata: pd.DataFrame,
    train_waveforms: np.ndarray,
    val_metadata: pd.DataFrame,
    val_waveforms: np.ndarray,
    outcome_label: str,
    model_name: str,
    lead_config: str,
    use_augmentation: bool,
    use_oversampling: bool,
    random_state: int,
) -> Tuple[Pipeline, Dict[str, float], Dict[str, float]]:
    """
    Train a single fold.

    Args:
        train_metadata: Training metadata
        train_waveforms: Training waveforms
        val_metadata: Validation metadata
        val_waveforms: Validation waveforms
        outcome_label: Outcome column name
        model_name: Model name
        lead_config: Lead configuration
        use_augmentation: Whether to use augmentation
        use_oversampling: Whether to use oversampling
        random_state: Random seed

    Returns:
        Tuple of (trained_pipeline, train_metrics, val_metrics)
    """
    # Initialize augmenter
    augmenter = ECGAugmenter()

    # Prepare lead configuration
    train_X = augmenter.prepare_lead_configuration(train_waveforms, lead_config)
    val_X = augmenter.prepare_lead_configuration(val_waveforms, lead_config)

    # Apply augmentation to training data only
    if use_augmentation:
        train_metadata_aug, train_X_aug = augmenter.augment_training_data(
            train_metadata, train_X, enable_augmentation=True
        )
        train_y = train_metadata_aug[outcome_label].values
        train_X = train_X_aug
    else:
        train_y = train_metadata[outcome_label].values

    val_y = val_metadata[outcome_label].values

    # Create pipeline
    n_features = train_X.shape[-1]  # Number of leads
    pipeline = create_pipeline(model_name, n_features, use_oversampling, random_state)

    # Train model
    logger.info(f"Training {model_name} with {len(train_X)} samples")
    pipeline.fit(train_X, train_y)

    # Evaluate on training set
    train_pred = pipeline.predict(train_X)
    train_prob = pipeline.predict_proba(train_X)[:, 1]
    train_metrics = evaluate_model(train_y, train_pred, train_prob)

    # Evaluate on validation set
    val_pred = pipeline.predict(val_X)
    val_prob = pipeline.predict_proba(val_X)[:, 1]
    val_metrics = evaluate_model(val_y, val_pred, val_prob)

    return pipeline, train_metrics, val_metrics


def run_experiment(
    metadata_df: pd.DataFrame,
    waveforms: np.ndarray,
    pre_ecg_window: str,
    outcome_label: str,
    model_name: str,
    lead_config: str,
    use_augmentation: bool,
    use_oversampling: bool,
    use_cv: bool,
    n_folds: int,
    random_state: int,
    results_dir: Path,
) -> Dict[str, Any]:
    """
    Run a complete experiment.

    Args:
        metadata_df: Full metadata DataFrame
        waveforms: Full waveform array
        pre_ecg_window: Pre-ECG window column name
        outcome_label: Outcome label column name
        model_name: Model name
        lead_config: Lead configuration
        use_augmentation: Whether to use augmentation
        use_oversampling: Whether to use oversampling
        use_cv: Whether to use cross-validation
        n_folds: Number of CV folds
        random_state: Random seed
        results_dir: Results directory

    Returns:
        Dictionary with experiment results
    """
    logger.info(f"Running experiment: {pre_ecg_window} -> {outcome_label}")

    # Get cohort for this window and outcome
    cohort_df, cohort_waveforms = get_cohort_for_window_and_outcome(
        metadata_df, waveforms, pre_ecg_window, outcome_label
    )

    if len(cohort_df) == 0:
        logger.warning(f"No data for {pre_ecg_window} -> {outcome_label}")
        return {"error": "No data available"}

    # Create splits
    splitter = PatientLevelSplitter(random_state=random_state)
    splits = splitter.create_splits(
        cohort_df, cohort_waveforms, outcome_label, use_cv=use_cv, n_folds=n_folds
    )

    results = {
        "experiment_config": {
            "pre_ecg_window": pre_ecg_window,
            "outcome_label": outcome_label,
            "model_name": model_name,
            "lead_config": lead_config,
            "use_augmentation": use_augmentation,
            "use_oversampling": use_oversampling,
            "use_cv": use_cv,
            "n_folds": n_folds,
            "random_state": random_state,
        },
        "cohort_size": len(cohort_df),
        "n_patients": cohort_df["mrn"].nunique(),
        "class_distribution": cohort_df[outcome_label].value_counts().to_dict(),
        "holdout_size": len(splits["holdout"]["metadata"]),
    }

    if use_cv:
        # Cross-validation
        cv_results = []
        fold_models = []

        for fold_data in track(splits["cv_folds"], description="Training CV folds"):
            fold_idx = fold_data["fold"]

            pipeline, train_metrics, val_metrics = train_single_fold(
                fold_data["train"]["metadata"],
                fold_data["train"]["waveforms"],
                fold_data["val"]["metadata"],
                fold_data["val"]["waveforms"],
                outcome_label,
                model_name,
                lead_config,
                use_augmentation,
                use_oversampling,
                random_state,
            )

            cv_results.append(
                {
                    "fold": fold_idx,
                    "train_metrics": train_metrics,
                    "val_metrics": val_metrics,
                    "train_size": len(fold_data["train"]["metadata"]),
                    "val_size": len(fold_data["val"]["metadata"]),
                }
            )

            fold_models.append(pipeline)

        # Calculate CV summary statistics
        val_metrics_df = pd.DataFrame([r["val_metrics"] for r in cv_results])
        cv_summary = {
            "mean": val_metrics_df.mean().to_dict(),
            "std": val_metrics_df.std().to_dict(),
            "individual_folds": cv_results,
        }

        results["cv_results"] = cv_summary

        # Save best model (highest validation ROC-AUC)
        best_fold_idx = val_metrics_df["roc_auc"].idxmax()
        best_model = fold_models[best_fold_idx]

    else:
        # Simple train/val split
        pipeline, train_metrics, val_metrics = train_single_fold(
            splits["train"]["metadata"],
            splits["train"]["waveforms"],
            splits["val"]["metadata"],
            splits["val"]["waveforms"],
            outcome_label,
            model_name,
            lead_config,
            use_augmentation,
            use_oversampling,
            random_state,
        )

        results["train_metrics"] = train_metrics
        results["val_metrics"] = val_metrics
        results["train_size"] = len(splits["train"]["metadata"])
        results["val_size"] = len(splits["val"]["metadata"])

        best_model = pipeline

    # Evaluate on holdout set
    holdout_X = ECGAugmenter().prepare_lead_configuration(
        splits["holdout"]["waveforms"], lead_config
    )
    holdout_y = splits["holdout"]["metadata"][outcome_label].values

    holdout_pred = best_model.predict(holdout_X)
    holdout_prob = best_model.predict_proba(holdout_X)[:, 1]
    holdout_metrics = evaluate_model(holdout_y, holdout_pred, holdout_prob)

    results["holdout_metrics"] = holdout_metrics

    # Save model and results
    experiment_name = f"{pre_ecg_window}_{outcome_label}_{model_name}_{lead_config}"
    if use_augmentation:
        experiment_name += "_aug"
    if use_oversampling:
        experiment_name += "_oversample"

    model_path = results_dir / f"{experiment_name}_model.joblib"
    results_path = results_dir / f"{experiment_name}_results.json"

    joblib.dump(best_model, model_path)

    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    logger.info(f"Saved model to {model_path}")
    logger.info(f"Saved results to {results_path}")

    return results


def print_results_table(results: Dict[str, Any]) -> None:
    """Print results in a nice table format."""
    table = Table(title="Experiment Results")
    table.add_column("Metric", style="cyan")
    table.add_column("Train", style="green")
    table.add_column("Validation", style="yellow")
    table.add_column("Holdout", style="red")

    if "cv_results" in results:
        # CV results
        cv_mean = results["cv_results"]["mean"]
        cv_std = results["cv_results"]["std"]

        for metric in ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]:
            table.add_row(
                metric.upper(),
                "-",
                f"{cv_mean[metric]:.3f} ± {cv_std[metric]:.3f}",
                f"{results['holdout_metrics'][metric]:.3f}",
            )
    else:
        # Simple train/val results
        for metric in ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]:
            table.add_row(
                metric.upper(),
                f"{results['train_metrics'][metric]:.3f}",
                f"{results['val_metrics'][metric]:.3f}",
                f"{results['holdout_metrics'][metric]:.3f}",
            )

    console.print(table)


def main():
    """Main function with CLI interface."""
    parser = argparse.ArgumentParser(
        description="Train AF recurrence prediction models"
    )

    # Data arguments
    parser.add_argument(
        "--metadata-path",
        default="carto_ecg_metadata_FULL.csv",
        help="Path to metadata CSV file",
    )
    parser.add_argument(
        "--waveforms-path",
        default="carto_ecg_waveforms.npy",
        help="Path to waveforms NPY file",
    )

    # Experiment arguments
    parser.add_argument(
        "--pre-ecg-window", default="pre_ecg_1y", help="Pre-ECG window column name"
    )
    parser.add_argument(
        "--outcome-label", default="af_recurrence_1y", help="Outcome label column name"
    )
    parser.add_argument(
        "--model", choices=["rocket", "resnet"], default="rocket", help="Model to use"
    )
    parser.add_argument(
        "--leads", choices=["lead1", "all"], default="all", help="Lead configuration"
    )

    # Training arguments
    parser.add_argument(
        "--augment", action="store_true", help="Use sliding window augmentation"
    )
    parser.add_argument(
        "--oversample", action="store_true", help="Use oversampling for class imbalance"
    )
    parser.add_argument("--cv", action="store_true", help="Use cross-validation")
    parser.add_argument("--n-folds", type=int, default=5, help="Number of CV folds")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed")

    # Output arguments
    parser.add_argument(
        "--results-dir", default="results", help="Base directory for results"
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )

    args = parser.parse_args()

    # Setup
    setup_logging(args.log_level)
    results_dir = create_results_dir(args.results_dir)

    console.print(
        f"[bold green]Starting AF Recurrence Prediction Training[/bold green]"
    )
    console.print(f"Results will be saved to: {results_dir}")

    try:
        # Load data
        console.print("[bold]Loading data...[/bold]")
        metadata, waveforms = load_ecg_data(args.metadata_path, args.waveforms_path)

        # Preprocess
        console.print("[bold]Preprocessing data...[/bold]")
        sinus_df = filter_sinus_rhythm(metadata)
        processed_df = create_pre_ecg_labels(sinus_df)
        processed_df = create_outcome_labels(processed_df)

        # Run experiment
        console.print("[bold]Running experiment...[/bold]")
        results = run_experiment(
            processed_df,
            waveforms,
            args.pre_ecg_window,
            args.outcome_label,
            args.model,
            args.leads,
            args.augment,
            args.oversample,
            args.cv,
            args.n_folds,
            args.random_state,
            results_dir,
        )

        if "error" in results:
            console.print(f"[red]Experiment failed: {results['error']}[/red]")
            return

        # Print results
        print_results_table(results)

        console.print(f"[bold green]Experiment completed successfully![/bold green]")
        console.print(f"Results saved to: {results_dir}")

    except Exception as e:
        console.print(f"[red]Error: {str(e)}[/red]")
        logger.exception("Experiment failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
