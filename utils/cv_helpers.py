#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Cross-validation helpers for AF recurrence prediction.
Provides improved CV strategies for small, imbalanced datasets.
"""

import numpy as np
import pandas as pd
import logging
from typing import List, Tuple, Generator
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.utils import check_random_state

logger = logging.getLogger(__name__)


class StratifiedGroupKFoldWrapper:
    """
    Wrapper for StratifiedGroupKFold to handle patient-level stratification.
    """

    def __init__(self, n_splits: int = 5, shuffle: bool = True, random_state: int = 42):
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state
        self.cv = StratifiedGroupKFold(
            n_splits=n_splits, shuffle=shuffle, random_state=random_state
        )

    def split(
        self, metadata_df: pd.DataFrame, outcome_label: str
    ) -> Generator[Tuple[pd.DataFrame, pd.DataFrame], None, None]:
        """
        Split data using patient-level stratification.

        Args:
            metadata_df: ECG metadata DataFrame
            outcome_label: Column name for outcome to stratify on

        Yields:
            Tuple of (train_df, val_df) for each fold
        """
        if "mrn" not in metadata_df.columns:
            raise ValueError("mrn column required for patient-level splitting")

        # Create patient-level aggregation for stratification
        patient_outcomes = metadata_df.groupby("mrn")[outcome_label].max().reset_index()
        patient_outcomes.columns = ["mrn", "patient_outcome"]

        # Merge back to get groups and stratification labels
        df_with_patient_outcome = metadata_df.merge(patient_outcomes, on="mrn")

        # Use patient MRN as groups and patient-level outcome for stratification
        X = df_with_patient_outcome.index.values  # Just use indices as features
        y = df_with_patient_outcome["patient_outcome"].values
        groups = df_with_patient_outcome["mrn"].values

        fold_idx = 0
        for train_idx, val_idx in self.cv.split(X, y, groups):
            train_df = metadata_df.iloc[train_idx].copy()
            val_df = metadata_df.iloc[val_idx].copy()

            # Validate no patient leakage
            train_mrns = set(train_df["mrn"].unique())
            val_mrns = set(val_df["mrn"].unique())

            if train_mrns & val_mrns:
                raise ValueError(f"Patient leakage detected in fold {fold_idx}")

            logger.info(
                f"Fold {fold_idx}: {len(train_df)} train ECGs ({len(train_mrns)} patients), "
                f"{len(val_df)} val ECGs ({len(val_mrns)} patients)"
            )

            yield train_df, val_df
            fold_idx += 1


def create_balanced_folds(
    metadata_df: pd.DataFrame,
    outcome_label: str,
    n_splits: int = 5,
    random_state: int = 42,
) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Create balanced CV folds with patient-level stratification.

    Args:
        metadata_df: ECG metadata DataFrame
        outcome_label: Column name for outcome to stratify on
        n_splits: Number of folds
        random_state: Random seed

    Returns:
        List of (train_df, val_df) tuples
    """
    splitter = StratifiedGroupKFoldWrapper(
        n_splits=n_splits, shuffle=True, random_state=random_state
    )

    folds = list(splitter.split(metadata_df, outcome_label))

    # Log fold statistics
    logger.info(f"Created {len(folds)} balanced folds")

    for i, (train_df, val_df) in enumerate(folds):
        train_pos = train_df[outcome_label].sum()
        val_pos = val_df[outcome_label].sum()
        train_rate = train_pos / len(train_df) if len(train_df) > 0 else 0
        val_rate = val_pos / len(val_df) if len(val_df) > 0 else 0

        logger.info(
            f"  Fold {i}: Train {train_rate:.1%} positive, Val {val_rate:.1%} positive"
        )

    return folds


def calculate_sample_weights(y: np.ndarray, method: str = "balanced") -> np.ndarray:
    """
    Calculate sample weights for imbalanced datasets.

    Args:
        y: Target labels
        method: Weighting method ('balanced', 'inverse', or custom ratio like '1:3')

    Returns:
        Array of sample weights
    """
    unique_classes, counts = np.unique(y, return_counts=True)

    if method == "balanced":
        # Standard balanced weighting
        class_weights = len(y) / (len(unique_classes) * counts)
        weight_dict = dict(zip(unique_classes, class_weights))
    elif method == "inverse":
        # Inverse frequency weighting
        class_weights = 1.0 / counts
        weight_dict = dict(zip(unique_classes, class_weights))
    elif ":" in method:
        # Custom ratio like "1:3" for negative:positive
        try:
            neg_weight, pos_weight = map(float, method.split(":"))
            weight_dict = {0: neg_weight, 1: pos_weight}
        except ValueError:
            raise ValueError(f"Invalid custom ratio format: {method}")
    else:
        raise ValueError(f"Unknown weighting method: {method}")

    # Map weights to samples
    sample_weights = np.array([weight_dict[label] for label in y])

    logger.info(f"Sample weights calculated using {method} method")
    logger.info(f"  Class distribution: {dict(zip(unique_classes, counts))}")
    logger.info(f"  Class weights: {weight_dict}")

    return sample_weights


def validate_cv_splits(
    folds: List[Tuple[pd.DataFrame, pd.DataFrame]],
    outcome_label: str,
    min_positive_per_fold: int = 5,
) -> bool:
    """
    Validate that CV folds have sufficient positive examples.

    Args:
        folds: List of (train_df, val_df) tuples
        outcome_label: Column name for outcome
        min_positive_per_fold: Minimum positive examples required per fold

    Returns:
        True if all folds are valid
    """
    valid = True

    for i, (train_df, val_df) in enumerate(folds):
        train_pos = train_df[outcome_label].sum()
        val_pos = val_df[outcome_label].sum()

        if train_pos < min_positive_per_fold:
            logger.warning(f"Fold {i} train set has only {train_pos} positive examples")
            valid = False

        if val_pos < min_positive_per_fold:
            logger.warning(f"Fold {i} val set has only {val_pos} positive examples")
            valid = False

    if valid:
        logger.info("All CV folds passed validation")
    else:
        logger.warning("Some CV folds have insufficient positive examples")

    return valid


def get_fold_statistics(
    folds: List[Tuple[pd.DataFrame, pd.DataFrame]], outcome_label: str
) -> pd.DataFrame:
    """
    Get statistics for each CV fold.

    Args:
        folds: List of (train_df, val_df) tuples
        outcome_label: Column name for outcome

    Returns:
        DataFrame with fold statistics
    """
    stats = []

    for i, (train_df, val_df) in enumerate(folds):
        train_pos = train_df[outcome_label].sum()
        val_pos = val_df[outcome_label].sum()

        stats.append(
            {
                "fold": i,
                "train_size": len(train_df),
                "val_size": len(val_df),
                "train_positive": train_pos,
                "val_positive": val_pos,
                "train_positive_rate": train_pos / len(train_df)
                if len(train_df) > 0
                else 0,
                "val_positive_rate": val_pos / len(val_df) if len(val_df) > 0 else 0,
                "train_patients": train_df["mrn"].nunique(),
                "val_patients": val_df["mrn"].nunique(),
            }
        )

    return pd.DataFrame(stats)


def bootstrap_minority_patients(
    metadata_df: pd.DataFrame,
    waveforms: np.ndarray,
    outcome_label: str,
    target_ratio: float = 0.3,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Bootstrap minority class patients to improve class balance.

    Args:
        metadata_df: ECG metadata DataFrame
        waveforms: ECG waveform array
        outcome_label: Column name for outcome
        target_ratio: Target positive class ratio
        random_state: Random seed

    Returns:
        Tuple of (augmented_metadata, augmented_waveforms)
    """
    np.random.seed(random_state)

    # Get current class distribution
    pos_count = metadata_df[outcome_label].sum()
    total_count = len(metadata_df)
    current_ratio = pos_count / total_count

    logger.info(f"Current positive ratio: {current_ratio:.3f}")
    logger.info(f"Target positive ratio: {target_ratio:.3f}")

    if current_ratio >= target_ratio:
        logger.info("Already at target ratio, no bootstrapping needed")
        return metadata_df, waveforms

    # Calculate how many positive samples we need
    target_pos_count = int(total_count * target_ratio / (1 - target_ratio))
    additional_pos_needed = target_pos_count - pos_count

    if additional_pos_needed <= 0:
        return metadata_df, waveforms

    # Get positive class patients
    pos_patients = metadata_df[metadata_df[outcome_label] == 1]["mrn"].unique()

    # Bootstrap patients (sample with replacement)
    n_patients_to_bootstrap = min(additional_pos_needed // 2, len(pos_patients))
    bootstrapped_patients = np.random.choice(
        pos_patients, size=n_patients_to_bootstrap, replace=True
    )

    # Collect all ECGs from bootstrapped patients
    bootstrap_rows = []
    bootstrap_indices = []

    for patient in bootstrapped_patients:
        patient_ecgs = metadata_df[metadata_df["mrn"] == patient]
        for _, row in patient_ecgs.iterrows():
            bootstrap_rows.append(row.copy())
            bootstrap_indices.append(row.name)

    if len(bootstrap_rows) == 0:
        return metadata_df, waveforms

    # Create augmented dataset
    bootstrap_df = pd.DataFrame(bootstrap_rows)
    bootstrap_waveforms = waveforms[bootstrap_indices]

    # Combine with original data
    augmented_metadata = pd.concat([metadata_df, bootstrap_df], ignore_index=True)
    augmented_waveforms = np.concatenate([waveforms, bootstrap_waveforms], axis=0)

    # Log results
    new_pos_count = augmented_metadata[outcome_label].sum()
    new_total_count = len(augmented_metadata)
    new_ratio = new_pos_count / new_total_count

    logger.info(
        f"Added {len(bootstrap_df)} ECGs from {len(set(bootstrapped_patients))} patients"
    )
    logger.info(f"New positive ratio: {new_ratio:.3f}")
    logger.info(f"New dataset size: {new_total_count} ECGs")

    return augmented_metadata, augmented_waveforms


if __name__ == "__main__":
    # Test the CV helpers
    import sys
    from pathlib import Path

    # Add ml_pipeline to path
    sys.path.append(str(Path(__file__).parent.parent / "ml_pipeline"))

    from data_loading import load_ecg_data
    from preprocess import (
        filter_sinus_rhythm,
        create_pre_ecg_labels,
        create_outcome_labels,
        get_cohort_for_window_and_outcome,
    )

    logging.basicConfig(level=logging.INFO)

    # Load and preprocess data
    metadata, waveforms = load_ecg_data()
    sinus_df = filter_sinus_rhythm(metadata)
    processed_df = create_pre_ecg_labels(sinus_df)
    processed_df = create_outcome_labels(processed_df)

    # Get a test cohort
    cohort_df, _ = get_cohort_for_window_and_outcome(
        processed_df, waveforms, "pre_ecg_1y", "af_recurrence_1y"
    )

    # Test balanced folds
    folds = create_balanced_folds(cohort_df, "af_recurrence_1y", n_splits=3)

    # Validate folds
    validate_cv_splits(folds, "af_recurrence_1y")

    # Get statistics
    stats_df = get_fold_statistics(folds, "af_recurrence_1y")
    print("\nFold Statistics:")
    print(stats_df.to_string(index=False))

    # Test sample weights
    y = cohort_df["af_recurrence_1y"].values
    weights = calculate_sample_weights(y, method="balanced")
    print(f"\nSample weights summary: min={weights.min():.3f}, max={weights.max():.3f}")
