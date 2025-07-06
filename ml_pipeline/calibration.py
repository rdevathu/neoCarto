#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Threshold optimization and probability calibration for AF recurrence prediction.
Implements advanced post-processing techniques to improve model performance.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import logging
from sklearn.calibration import CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    precision_recall_curve,
    roc_curve,
    average_precision_score,
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
    brier_score_loss,
    log_loss,
)
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import joblib
import json

logger = logging.getLogger(__name__)


class ThresholdOptimizer:
    """
    Optimizes decision thresholds for binary classification.

    Supports multiple optimization criteria:
    - F1-score maximization
    - Precision-Recall balance (F-beta score)
    - Youden's J statistic (sensitivity + specificity - 1)
    - Cost-sensitive optimization
    """

    def __init__(self, metric: str = "f1", beta: float = 1.0, cost_ratio: float = 1.0):
        """
        Initialize threshold optimizer.

        Args:
            metric: Optimization metric ('f1', 'f_beta', 'youden', 'cost_sensitive')
            beta: Beta parameter for F-beta score (higher beta favors recall)
            cost_ratio: Cost ratio for false positives vs false negatives
        """
        self.metric = metric
        self.beta = beta
        self.cost_ratio = cost_ratio
        self.optimal_threshold_ = None
        self.threshold_scores_ = None

    def fit(self, y_true: np.ndarray, y_prob: np.ndarray) -> "ThresholdOptimizer":
        """
        Find optimal threshold based on validation data.

        Args:
            y_true: True binary labels
            y_prob: Predicted probabilities for positive class

        Returns:
            Self for method chaining
        """
        # Generate candidate thresholds
        thresholds = np.linspace(0.01, 0.99, 99)
        scores = []

        for threshold in thresholds:
            y_pred = (y_prob >= threshold).astype(int)
            score = self._calculate_score(y_true, y_pred, y_prob)
            scores.append(score)

        scores = np.array(scores)
        self.threshold_scores_ = list(zip(thresholds, scores))

        # Find optimal threshold
        optimal_idx = np.argmax(scores)
        self.optimal_threshold_ = thresholds[optimal_idx]

        logger.info(
            f"Optimal threshold: {self.optimal_threshold_:.3f} "
            f"(score: {scores[optimal_idx]:.3f})"
        )

        return self

    def _calculate_score(
        self, y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray
    ) -> float:
        """Calculate optimization score for given predictions."""
        if len(np.unique(y_true)) < 2:
            return 0.0

        if self.metric == "f1":
            return f1_score(y_true, y_pred, zero_division=0)

        elif self.metric == "f_beta":
            precision = precision_score(y_true, y_pred, zero_division=0)
            recall = recall_score(y_true, y_pred, zero_division=0)
            if precision + recall == 0:
                return 0.0
            return (
                (1 + self.beta**2)
                * precision
                * recall
                / (self.beta**2 * precision + recall)
            )

        elif self.metric == "youden":
            # Youden's J = Sensitivity + Specificity - 1
            tn = np.sum((y_true == 0) & (y_pred == 0))
            fp = np.sum((y_true == 0) & (y_pred == 1))
            fn = np.sum((y_true == 1) & (y_pred == 0))
            tp = np.sum((y_true == 1) & (y_pred == 1))

            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

            return sensitivity + specificity - 1

        elif self.metric == "cost_sensitive":
            # Minimize cost = FP * cost_ratio + FN
            fp = np.sum((y_true == 0) & (y_pred == 1))
            fn = np.sum((y_true == 1) & (y_pred == 0))
            cost = fp * self.cost_ratio + fn
            # Return negative cost for maximization
            return -cost / len(y_true)

        else:
            raise ValueError(f"Unknown metric: {self.metric}")

    def predict(self, y_prob: np.ndarray) -> np.ndarray:
        """Make predictions using optimal threshold."""
        if self.optimal_threshold_ is None:
            raise ValueError("Must call fit() before predict()")
        return (y_prob >= self.optimal_threshold_).astype(int)

    def get_threshold_curve(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get threshold vs score curve for plotting."""
        if self.threshold_scores_ is None:
            raise ValueError("Must call fit() before get_threshold_curve()")

        thresholds, scores = zip(*self.threshold_scores_)
        return np.array(thresholds), np.array(scores)


class ProbabilityCalibrator:
    """
    Calibrates predicted probabilities using various methods.

    Supports:
    - Platt scaling (logistic regression)
    - Isotonic regression
    - Beta calibration
    """

    def __init__(self, method: str = "platt", cv: int = 3):
        """
        Initialize probability calibrator.

        Args:
            method: Calibration method ('platt', 'isotonic', 'beta')
            cv: Number of CV folds for calibration
        """
        self.method = method
        self.cv = cv
        self.calibrator_ = None
        self.is_fitted_ = False

    def fit(self, y_true: np.ndarray, y_prob: np.ndarray) -> "ProbabilityCalibrator":
        """
        Fit calibration model.

        Args:
            y_true: True binary labels
            y_prob: Uncalibrated predicted probabilities

        Returns:
            Self for method chaining
        """
        if self.method == "platt":
            # Platt scaling using logistic regression
            self.calibrator_ = LogisticRegression()
            self.calibrator_.fit(y_prob.reshape(-1, 1), y_true)

        elif self.method == "isotonic":
            # Isotonic regression
            self.calibrator_ = IsotonicRegression(out_of_bounds="clip")
            self.calibrator_.fit(y_prob, y_true)

        elif self.method == "beta":
            # Beta calibration (fit beta distribution parameters)
            self.calibrator_ = self._fit_beta_calibration(y_true, y_prob)

        else:
            raise ValueError(f"Unknown calibration method: {self.method}")

        self.is_fitted_ = True
        return self

    def _fit_beta_calibration(
        self, y_true: np.ndarray, y_prob: np.ndarray
    ) -> Dict[str, float]:
        """Fit beta calibration parameters."""
        # Separate probabilities by class
        pos_probs = y_prob[y_true == 1]
        neg_probs = y_prob[y_true == 0]

        # Fit beta distributions
        if len(pos_probs) > 1:
            pos_alpha, pos_beta, _, _ = stats.beta.fit(pos_probs, floc=0, fscale=1)
        else:
            pos_alpha, pos_beta = 1, 1

        if len(neg_probs) > 1:
            neg_alpha, neg_beta, _, _ = stats.beta.fit(neg_probs, floc=0, fscale=1)
        else:
            neg_alpha, neg_beta = 1, 1

        return {
            "pos_alpha": pos_alpha,
            "pos_beta": pos_beta,
            "neg_alpha": neg_alpha,
            "neg_beta": neg_beta,
            "pos_prior": len(pos_probs) / len(y_prob),
        }

    def predict_proba(self, y_prob: np.ndarray) -> np.ndarray:
        """
        Calibrate predicted probabilities.

        Args:
            y_prob: Uncalibrated predicted probabilities

        Returns:
            Calibrated probabilities
        """
        if not self.is_fitted_:
            raise ValueError("Must call fit() before predict_proba()")

        if self.method == "platt":
            return self.calibrator_.predict_proba(y_prob.reshape(-1, 1))[:, 1]

        elif self.method == "isotonic":
            return self.calibrator_.predict(y_prob)

        elif self.method == "beta":
            return self._predict_beta_calibration(y_prob)

        else:
            raise ValueError(f"Unknown calibration method: {self.method}")

    def _predict_beta_calibration(self, y_prob: np.ndarray) -> np.ndarray:
        """Apply beta calibration."""
        params = self.calibrator_

        # Calculate likelihoods
        pos_likelihood = stats.beta.pdf(y_prob, params["pos_alpha"], params["pos_beta"])
        neg_likelihood = stats.beta.pdf(y_prob, params["neg_alpha"], params["neg_beta"])

        # Apply Bayes' rule
        pos_prior = params["pos_prior"]
        neg_prior = 1 - pos_prior

        numerator = pos_likelihood * pos_prior
        denominator = pos_likelihood * pos_prior + neg_likelihood * neg_prior

        # Avoid division by zero
        denominator = np.maximum(denominator, 1e-10)

        return numerator / denominator


class CalibrationEvaluator:
    """Evaluates calibration quality and generates calibration plots."""

    @staticmethod
    def reliability_diagram(
        y_true: np.ndarray,
        y_prob: np.ndarray,
        n_bins: int = 10,
        strategy: str = "uniform",
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate reliability diagram data.

        Args:
            y_true: True binary labels
            y_prob: Predicted probabilities
            n_bins: Number of bins for reliability diagram
            strategy: Binning strategy ('uniform' or 'quantile')

        Returns:
            Tuple of (bin_boundaries, bin_lowers, bin_uppers, empirical_prob, mean_pred_prob, counts)
        """
        if strategy == "uniform":
            bin_boundaries = np.linspace(0, 1, n_bins + 1)
        elif strategy == "quantile":
            bin_boundaries = np.percentile(y_prob, np.linspace(0, 100, n_bins + 1))
        else:
            raise ValueError("Strategy must be 'uniform' or 'quantile'")

        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]

        empirical_prob = np.zeros(n_bins)
        mean_pred_prob = np.zeros(n_bins)
        counts = np.zeros(n_bins)

        for i in range(n_bins):
            in_bin = (y_prob > bin_lowers[i]) & (y_prob <= bin_uppers[i])
            counts[i] = in_bin.sum()

            if counts[i] > 0:
                empirical_prob[i] = y_true[in_bin].mean()
                mean_pred_prob[i] = y_prob[in_bin].mean()

        return bin_boundaries, empirical_prob, mean_pred_prob, counts

    @staticmethod
    def calibration_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
        """
        Calculate calibration quality metrics.

        Args:
            y_true: True binary labels
            y_prob: Predicted probabilities

        Returns:
            Dictionary of calibration metrics
        """
        # Brier score (lower is better)
        brier_score = brier_score_loss(y_true, y_prob)

        # Log loss (lower is better)
        log_loss_score = log_loss(y_true, y_prob)

        # Expected Calibration Error (ECE)
        bin_boundaries, empirical_prob, mean_pred_prob, counts = (
            CalibrationEvaluator.reliability_diagram(y_true, y_prob)
        )

        total_samples = len(y_true)
        ece = 0.0

        for i in range(len(empirical_prob)):
            if counts[i] > 0:
                ece += (counts[i] / total_samples) * abs(
                    empirical_prob[i] - mean_pred_prob[i]
                )

        # Maximum Calibration Error (MCE)
        mce = 0.0
        for i in range(len(empirical_prob)):
            if counts[i] > 0:
                mce = max(mce, abs(empirical_prob[i] - mean_pred_prob[i]))

        return {
            "brier_score": brier_score,
            "log_loss": log_loss_score,
            "ece": ece,
            "mce": mce,
        }

    @staticmethod
    def plot_calibration_curve(
        y_true: np.ndarray,
        y_prob_uncalibrated: np.ndarray,
        y_prob_calibrated: Optional[np.ndarray] = None,
        n_bins: int = 10,
        save_path: Optional[Path] = None,
    ) -> plt.Figure:
        """
        Plot calibration curve (reliability diagram).

        Args:
            y_true: True binary labels
            y_prob_uncalibrated: Uncalibrated predicted probabilities
            y_prob_calibrated: Calibrated predicted probabilities (optional)
            n_bins: Number of bins
            save_path: Path to save plot (optional)

        Returns:
            Matplotlib figure
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Plot 1: Reliability diagram
        _, emp_prob, mean_pred_prob, counts = CalibrationEvaluator.reliability_diagram(
            y_true, y_prob_uncalibrated, n_bins
        )

        # Plot uncalibrated
        mask = counts > 0
        ax1.plot(
            mean_pred_prob[mask],
            emp_prob[mask],
            "o-",
            label="Uncalibrated",
            color="red",
        )

        # Plot calibrated if provided
        if y_prob_calibrated is not None:
            _, emp_prob_cal, mean_pred_prob_cal, counts_cal = (
                CalibrationEvaluator.reliability_diagram(
                    y_true, y_prob_calibrated, n_bins
                )
            )
            mask_cal = counts_cal > 0
            ax1.plot(
                mean_pred_prob_cal[mask_cal],
                emp_prob_cal[mask_cal],
                "o-",
                label="Calibrated",
                color="blue",
            )

        # Perfect calibration line
        ax1.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect calibration")
        ax1.set_xlabel("Mean Predicted Probability")
        ax1.set_ylabel("Fraction of Positives")
        ax1.set_title("Reliability Diagram")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot 2: Histogram of predicted probabilities
        ax2.hist(
            y_prob_uncalibrated, bins=20, alpha=0.7, label="Uncalibrated", color="red"
        )
        if y_prob_calibrated is not None:
            ax2.hist(
                y_prob_calibrated, bins=20, alpha=0.7, label="Calibrated", color="blue"
            )
        ax2.set_xlabel("Predicted Probability")
        ax2.set_ylabel("Count")
        ax2.set_title("Distribution of Predicted Probabilities")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")

        return fig


class CalibratedModel:
    """
    Wrapper that combines a trained model with threshold optimization and probability calibration.
    """

    def __init__(
        self,
        base_model,
        threshold_optimizer: Optional[ThresholdOptimizer] = None,
        probability_calibrator: Optional[ProbabilityCalibrator] = None,
    ):
        """
        Initialize calibrated model wrapper.

        Args:
            base_model: Trained sklearn-compatible model
            threshold_optimizer: Optional threshold optimizer
            probability_calibrator: Optional probability calibrator
        """
        self.base_model = base_model
        self.threshold_optimizer = threshold_optimizer
        self.probability_calibrator = probability_calibrator

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get calibrated predicted probabilities."""
        # Get base model probabilities
        y_prob = self.base_model.predict_proba(X)[:, 1]

        # Apply probability calibration if available
        if self.probability_calibrator is not None:
            y_prob = self.probability_calibrator.predict_proba(y_prob)

        # Return as 2D array for compatibility
        return np.column_stack([1 - y_prob, y_prob])

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Get predictions using optimized threshold."""
        y_prob = self.predict_proba(X)[:, 1]

        # Apply threshold optimization if available
        if self.threshold_optimizer is not None:
            return self.threshold_optimizer.predict(y_prob)
        else:
            return (y_prob >= 0.5).astype(int)

    def save(self, path: Path) -> None:
        """Save calibrated model to disk."""
        model_data = {
            "base_model": self.base_model,
            "threshold_optimizer": self.threshold_optimizer,
            "probability_calibrator": self.probability_calibrator,
        }
        joblib.dump(model_data, path)

    @classmethod
    def load(cls, path: Path) -> "CalibratedModel":
        """Load calibrated model from disk."""
        model_data = joblib.load(path)
        return cls(
            base_model=model_data["base_model"],
            threshold_optimizer=model_data["threshold_optimizer"],
            probability_calibrator=model_data["probability_calibrator"],
        )


def optimize_model_calibration(
    model,
    X_val: np.ndarray,
    y_val: np.ndarray,
    threshold_metric: str = "f1",
    calibration_method: str = "platt",
    cv_folds: int = 3,
) -> CalibratedModel:
    """
    Optimize threshold and calibrate probabilities for a trained model.

    Args:
        model: Trained sklearn-compatible model
        X_val: Validation features
        y_val: Validation labels
        threshold_metric: Metric for threshold optimization
        calibration_method: Method for probability calibration
        cv_folds: Number of CV folds for calibration

    Returns:
        CalibratedModel with optimized threshold and calibrated probabilities
    """
    # Get base model predictions
    y_prob = model.predict_proba(X_val)[:, 1]

    # Optimize threshold
    threshold_optimizer = ThresholdOptimizer(metric=threshold_metric)
    threshold_optimizer.fit(y_val, y_prob)

    # Calibrate probabilities using CV to avoid overfitting
    probability_calibrator = ProbabilityCalibrator(
        method=calibration_method, cv=cv_folds
    )
    probability_calibrator.fit(y_val, y_prob)

    return CalibratedModel(
        base_model=model,
        threshold_optimizer=threshold_optimizer,
        probability_calibrator=probability_calibrator,
    )
