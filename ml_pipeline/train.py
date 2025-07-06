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
from splitter import PatientLevelSplitter, PatientBootstrapper
from augment import ECGAugmenter
from calibration import (
    ThresholdOptimizer,
    ProbabilityCalibrator,
    CalibratedModel,
    CalibrationEvaluator,
    optimize_model_calibration,
)

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


def get_model(model_name: str, n_features: int, random_state: int = 42, **kwargs):
    """
    Get model instance based on name.

    Args:
        model_name: Name of the model ('rocket' or 'resnet')
        n_features: Number of features (for ResNet)
        random_state: Random seed
        **kwargs: Additional model parameters

    Returns:
        Model instance
    """
    if model_name == "rocket":
        try:
            from sktime.classification.kernel_based import RocketClassifier

            return RocketClassifier(random_state=random_state, n_jobs=-1)
        except ImportError:
            console.print(
                "[red]Error: sktime not installed or RocketClassifier not available[/red]"
            )
            raise

    elif model_name == "rocket_transformer":
        try:
            from sktime.transformations.panel.rocket import Rocket
            from sklearn.linear_model import LogisticRegression

            # Extract parameters
            num_kernels = kwargs.get("num_kernels", 5000)
            C = kwargs.get("C", 0.1)
            class_weight = kwargs.get("class_weight", None)
            max_iter = kwargs.get("max_iter", 2000)
            penalty = kwargs.get("penalty", "l2")
            l1_ratio = kwargs.get("l1_ratio", 0.5)

            # Create ROCKET transformer
            rocket_transformer = Rocket(
                num_kernels=num_kernels,
                random_state=random_state,
                n_jobs=-1,
            )

            # Create logistic regression classifier with appropriate solver
            solver = (
                "liblinear"
                if penalty == "l1"
                else "saga"
                if penalty == "elasticnet"
                else "lbfgs"
            )

            classifier_kwargs = {
                "C": C,
                "class_weight": class_weight,
                "max_iter": max_iter,
                "random_state": random_state,
                "n_jobs": -1
                if solver != "liblinear"
                else 1,  # liblinear doesn't support n_jobs
                "solver": solver,
                "penalty": penalty,
            }

            # Add l1_ratio for elasticnet
            if penalty == "elasticnet":
                classifier_kwargs["l1_ratio"] = l1_ratio

            classifier = LogisticRegression(**classifier_kwargs)

            return rocket_transformer, classifier
        except ImportError:
            console.print(
                "[red]Error: sktime not installed or Rocket transformer not available[/red]"
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
    use_sample_weights: bool = False,
    class_weight: str = None,
    random_state: int = 42,
    **model_kwargs,
) -> Pipeline:
    """
    Create ML pipeline with preprocessing and model.

    Args:
        model_name: Name of the model
        n_features: Number of features
        use_oversampling: Whether to use oversampling for class imbalance
        use_sample_weights: Whether to use sample weights instead of oversampling
        class_weight: Class weight strategy for models that support it
        random_state: Random seed
        **model_kwargs: Additional model parameters

    Returns:
        Sklearn pipeline
    """
    # Handle class weighting for supported models
    if class_weight and model_name == "rocket_transformer":
        model_kwargs["class_weight"] = class_weight

    # Get model components
    model_components = get_model(model_name, n_features, random_state, **model_kwargs)

    # Handle different model types
    if model_name == "rocket_transformer":
        # ROCKET transformer + classifier pipeline
        transformer, classifier = model_components

        steps = [("rocket_transformer", transformer)]

        # Add oversampling after transformation (now 2D)
        if use_oversampling and not use_sample_weights:
            steps.append(("oversampler", RandomOverSampler(random_state=random_state)))

        steps.append(("classifier", classifier))

        if use_oversampling and not use_sample_weights:
            return ImbPipeline(steps)
        else:
            return Pipeline(steps)

    else:
        # Traditional approach for other models
        model = model_components

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
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    include_calibration: bool = False,
) -> Dict[str, float]:
    """
    Evaluate model performance.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_prob: Predicted probabilities
        include_calibration: Whether to include calibration metrics

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

    # Add calibration metrics if requested
    if include_calibration and len(np.unique(y_true)) > 1:
        try:
            calibration_metrics = CalibrationEvaluator.calibration_metrics(
                y_true, y_prob
            )
            metrics.update(calibration_metrics)
        except Exception as e:
            logger.warning(f"Could not calculate calibration metrics: {e}")

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
    use_sample_weights: bool,
    class_weight: str,
    random_state: int,
    **model_kwargs,
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
        use_sample_weights: Whether to use sample weights
        class_weight: Class weight strategy
        random_state: Random seed
        **model_kwargs: Additional model parameters

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
    pipeline = create_pipeline(
        model_name,
        n_features,
        use_oversampling,
        use_sample_weights,
        class_weight,
        random_state,
        **model_kwargs,
    )

    # Train model
    logger.info(f"Training {model_name} with {len(train_X)} samples")

    # Handle sample weights if requested
    if (
        use_sample_weights
        and not use_oversampling
        and model_name == "rocket_transformer"
    ):
        # Import here to avoid circular imports
        import sys
        from pathlib import Path

        utils_path = Path(__file__).parent.parent / "utils"
        if str(utils_path) not in sys.path:
            sys.path.append(str(utils_path))

        try:
            from cv_helpers import calculate_sample_weights

            sample_weights = calculate_sample_weights(
                train_y, method=class_weight or "balanced"
            )

            # For rocket_transformer, pass sample weights to the final classifier
            pipeline.fit(train_X, train_y, classifier__sample_weight=sample_weights)
        except ImportError as e:
            logger.warning(
                f"Could not import cv_helpers: {e}, falling back to regular fit"
            )
            pipeline.fit(train_X, train_y)
    else:
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
    use_sample_weights: bool,
    class_weight: str,
    use_cv: bool,
    n_folds: int,
    random_state: int,
    results_dir: Path,
    bootstrap_patients: bool = False,
    enable_calibration: bool = False,
    threshold_metric: str = "f1",
    calibration_method: str = "platt",
    calibration_cv: int = 3,
    **model_kwargs,
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
        use_sample_weights: Whether to use sample weights
        class_weight: Class weight strategy
        use_cv: Whether to use cross-validation
        n_folds: Number of CV folds
        random_state: Random seed
        results_dir: Results directory
        **model_kwargs: Additional model parameters

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

    # Apply patient bootstrapping if requested
    if bootstrap_patients:
        from splitter import PatientBootstrapper

        logger.info("Applying patient-level bootstrapping...")

        bootstrapper = PatientBootstrapper(random_state=random_state)
        cohort_df, cohort_waveforms = bootstrapper.bootstrap_minority_patients(
            cohort_df, cohort_waveforms, outcome_label
        )

        # Validate bootstrap integrity
        original_cohort_df, _ = get_cohort_for_window_and_outcome(
            metadata_df, waveforms, pre_ecg_window, outcome_label
        )
        if not bootstrapper.validate_bootstrap_integrity(
            original_cohort_df, cohort_df, outcome_label
        ):
            logger.error("Bootstrap integrity validation failed")
            return {"error": "Bootstrap integrity validation failed"}

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
            "use_sample_weights": use_sample_weights,
            "class_weight": class_weight,
            "bootstrap_patients": bootstrap_patients,
            "use_cv": use_cv,
            "n_folds": n_folds,
            "random_state": random_state,
            "model_kwargs": model_kwargs,
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
                use_sample_weights,
                class_weight,
                random_state,
                **model_kwargs,
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
            use_sample_weights,
            class_weight,
            random_state,
            **model_kwargs,
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
    holdout_metrics = evaluate_model(
        holdout_y, holdout_pred, holdout_prob, include_calibration=True
    )

    results["holdout_metrics"] = holdout_metrics

    # Apply calibration optimization if enabled and using CV (we need validation data)
    calibrated_model = None
    if enable_calibration and use_cv and len(splits["cv_folds"]) > 0:
        logger.info("Applying threshold optimization and probability calibration...")

        # Use validation data from best fold for calibration
        best_fold_data = splits["cv_folds"][best_fold_idx]
        val_X = ECGAugmenter().prepare_lead_configuration(
            best_fold_data["val"]["waveforms"], lead_config
        )
        val_y = best_fold_data["val"]["metadata"][outcome_label].values

        try:
            # Optimize calibration
            calibrated_model = optimize_model_calibration(
                best_model,
                val_X,
                val_y,
                threshold_metric=threshold_metric,
                calibration_method=calibration_method,
                cv_folds=calibration_cv,
            )

            # Evaluate calibrated model on holdout
            holdout_pred_cal = calibrated_model.predict(holdout_X)
            holdout_prob_cal = calibrated_model.predict_proba(holdout_X)[:, 1]
            holdout_metrics_cal = evaluate_model(
                holdout_y, holdout_pred_cal, holdout_prob_cal, include_calibration=True
            )

            results["holdout_metrics_calibrated"] = holdout_metrics_cal

            # Log calibration improvements
            roc_improvement = (
                holdout_metrics_cal["roc_auc"] - holdout_metrics["roc_auc"]
            )
            f1_improvement = holdout_metrics_cal["f1"] - holdout_metrics["f1"]

            logger.info(
                f"Calibration improvements - ROC-AUC: {roc_improvement:+.3f}, F1: {f1_improvement:+.3f}"
            )

        except Exception as e:
            logger.warning(f"Calibration optimization failed: {e}")
            calibrated_model = None

    # Save model and results
    experiment_name = f"{pre_ecg_window}_{outcome_label}_{model_name}_{lead_config}"
    if use_augmentation:
        experiment_name += "_aug"
    if use_oversampling:
        experiment_name += "_oversample"
    if bootstrap_patients:
        experiment_name += "_bootstrap"

    model_path = results_dir / f"{experiment_name}_model.joblib"
    results_path = results_dir / f"{experiment_name}_results.json"

    # Save original model
    joblib.dump(best_model, model_path)

    # Save calibrated model if available
    if calibrated_model is not None:
        calibrated_model_path = (
            results_dir / f"{experiment_name}_calibrated_model.joblib"
        )
        calibrated_model.save(calibrated_model_path)
        logger.info(f"Saved calibrated model to {calibrated_model_path}")

    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    logger.info(f"Saved model to {model_path}")
    logger.info(f"Saved results to {results_path}")

    return results


def print_results_table(results: Dict[str, Any]) -> None:
    """Print results in a nice table format."""
    # Check if we have calibrated results
    has_calibrated = "holdout_metrics_calibrated" in results

    if has_calibrated:
        table = Table(title="Experiment Results (Original vs Calibrated)")
        table.add_column("Metric", style="cyan")
        table.add_column("Train", style="green")
        table.add_column("Validation", style="yellow")
        table.add_column("Holdout", style="red")
        table.add_column("Holdout (Cal.)", style="bold red")
    else:
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
            row = [
                metric.upper(),
                "-",
                f"{cv_mean[metric]:.3f} ± {cv_std[metric]:.3f}",
                f"{results['holdout_metrics'][metric]:.3f}",
            ]

            if has_calibrated:
                cal_value = results["holdout_metrics_calibrated"][metric]
                orig_value = results["holdout_metrics"][metric]
                improvement = cal_value - orig_value
                row.append(f"{cal_value:.3f} ({improvement:+.3f})")

            table.add_row(*row)
    else:
        # Simple train/val results
        for metric in ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]:
            row = [
                metric.upper(),
                f"{results['train_metrics'][metric]:.3f}",
                f"{results['val_metrics'][metric]:.3f}",
                f"{results['holdout_metrics'][metric]:.3f}",
            ]

            if has_calibrated:
                cal_value = results["holdout_metrics_calibrated"][metric]
                orig_value = results["holdout_metrics"][metric]
                improvement = cal_value - orig_value
                row.append(f"{cal_value:.3f} ({improvement:+.3f})")

            table.add_row(*row)

    console.print(table)

    # Print calibration-specific metrics if available
    if has_calibrated and "brier_score" in results["holdout_metrics_calibrated"]:
        cal_table = Table(title="Calibration Quality Metrics")
        cal_table.add_column("Metric", style="cyan")
        cal_table.add_column("Value", style="green")
        cal_table.add_column("Description", style="yellow")

        cal_metrics = results["holdout_metrics_calibrated"]

        cal_table.add_row(
            "Brier Score", f"{cal_metrics.get('brier_score', 0):.4f}", "Lower is better"
        )
        cal_table.add_row(
            "Log Loss", f"{cal_metrics.get('log_loss', 0):.4f}", "Lower is better"
        )
        cal_table.add_row(
            "ECE", f"{cal_metrics.get('ece', 0):.4f}", "Expected Calibration Error"
        )
        cal_table.add_row(
            "MCE", f"{cal_metrics.get('mce', 0):.4f}", "Maximum Calibration Error"
        )

        console.print(cal_table)


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
        "--model",
        choices=["rocket", "rocket_transformer", "resnet"],
        default="rocket_transformer",
        help="Model to use",
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
    parser.add_argument(
        "--use-sample-weights",
        action="store_true",
        help="Use sample weights instead of oversampling",
    )
    parser.add_argument(
        "--class-weight",
        default=None,
        help="Class weight strategy (balanced, inverse, or ratio like 1:3)",
    )
    parser.add_argument(
        "--bootstrap-patients",
        action="store_true",
        help="Bootstrap minority class patients to improve balance",
    )

    # Model-specific arguments
    parser.add_argument(
        "--num-kernels",
        type=int,
        default=1000,
        help="Number of ROCKET kernels (optimized for small datasets)",
    )
    parser.add_argument(
        "--C",
        type=float,
        default=0.0001,
        help="Logistic regression regularization parameter (optimized for small datasets)",
    )
    parser.add_argument(
        "--penalty",
        choices=["l1", "l2", "elasticnet"],
        default="l2",
        help="Regularization penalty type",
    )
    parser.add_argument(
        "--l1-ratio",
        type=float,
        default=0.5,
        help="L1 ratio for elasticnet penalty (0=L2, 1=L1)",
    )
    parser.add_argument("--cv", action="store_true", help="Use cross-validation")
    parser.add_argument(
        "--n-folds",
        type=int,
        default=3,
        help="Number of CV folds (reduced for small datasets)",
    )
    parser.add_argument("--random-state", type=int, default=42, help="Random seed")

    # Calibration arguments
    parser.add_argument(
        "--enable-calibration",
        action="store_true",
        help="Enable threshold optimization and probability calibration",
    )
    parser.add_argument(
        "--threshold-metric",
        choices=["f1", "f_beta", "youden", "cost_sensitive"],
        default="f1",
        help="Metric for threshold optimization",
    )
    parser.add_argument(
        "--calibration-method",
        choices=["platt", "isotonic", "beta"],
        default="platt",
        help="Method for probability calibration",
    )
    parser.add_argument(
        "--calibration-cv",
        type=int,
        default=3,
        help="Number of CV folds for calibration",
    )

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
        # Prepare model kwargs
        model_kwargs = {
            "num_kernels": args.num_kernels,
            "C": args.C,
            "penalty": args.penalty,
            "l1_ratio": args.l1_ratio,
        }

        results = run_experiment(
            processed_df,
            waveforms,
            args.pre_ecg_window,
            args.outcome_label,
            args.model,
            args.leads,
            args.augment,
            args.oversample,
            args.use_sample_weights,
            args.class_weight,
            args.cv,
            args.n_folds,
            args.random_state,
            results_dir,
            bootstrap_patients=args.bootstrap_patients,
            enable_calibration=args.enable_calibration,
            threshold_metric=args.threshold_metric,
            calibration_method=args.calibration_method,
            calibration_cv=args.calibration_cv,
            **model_kwargs,
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
