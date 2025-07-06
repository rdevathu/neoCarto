#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Splitting module for AF recurrence prediction.
Handles patient-level stratified splitting to prevent data leakage.
"""

import pandas as pd
import numpy as np
import logging
from typing import Tuple, List, Dict, Generator
from sklearn.model_selection import StratifiedKFold, train_test_split
from collections import defaultdict

logger = logging.getLogger(__name__)


def create_patient_level_holdout(
    metadata_df: pd.DataFrame,
    outcome_label: str,
    test_size: float = 0.2,
    random_state: int = 42,
    stratify_by_mrn: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create patient-level holdout set ensuring no patient leakage.

    Args:
        metadata_df: ECG metadata DataFrame
        outcome_label: Column name for outcome to stratify on
        test_size: Proportion of patients to hold out
        random_state: Random seed for reproducibility
        stratify_by_mrn: Whether to stratify by patient-level outcomes

    Returns:
        Tuple of (train_df, holdout_df)
    """
    if "mrn" not in metadata_df.columns:
        raise ValueError("mrn column required for patient-level splitting")

    # Create patient-level aggregation for stratification
    if stratify_by_mrn:
        # Aggregate outcomes at patient level (any positive ECG = positive patient)
        patient_outcomes = metadata_df.groupby("mrn")[outcome_label].max().reset_index()
        patient_outcomes.columns = ["mrn", "patient_outcome"]

        # Split patients, not ECGs
        train_mrns, holdout_mrns = train_test_split(
            patient_outcomes["mrn"],
            test_size=test_size,
            random_state=random_state,
            stratify=patient_outcomes["patient_outcome"],
        )
    else:
        # Simple random split of patients
        unique_mrns = metadata_df["mrn"].unique()
        train_mrns, holdout_mrns = train_test_split(
            unique_mrns, test_size=test_size, random_state=random_state
        )

    # Split ECGs based on patient assignment
    train_df = metadata_df[metadata_df["mrn"].isin(train_mrns)].copy()
    holdout_df = metadata_df[metadata_df["mrn"].isin(holdout_mrns)].copy()

    logger.info(f"Patient-level holdout split:")
    logger.info(f"  Train: {len(train_df)} ECGs from {len(train_mrns)} patients")
    logger.info(f"  Holdout: {len(holdout_df)} ECGs from {len(holdout_mrns)} patients")

    # Log outcome distribution
    if outcome_label in train_df.columns:
        train_pos = train_df[outcome_label].sum()
        holdout_pos = holdout_df[outcome_label].sum()
        logger.info(
            f"  Train positive rate: {train_pos}/{len(train_df)} ({train_pos / len(train_df) * 100:.1f}%)"
        )
        logger.info(
            f"  Holdout positive rate: {holdout_pos}/{len(holdout_df)} ({holdout_pos / len(holdout_df) * 100:.1f}%)"
        )

    return train_df, holdout_df


def create_patient_level_cv_folds(
    metadata_df: pd.DataFrame,
    outcome_label: str,
    n_folds: int = 5,
    random_state: int = 42,
) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Create patient-level cross-validation folds.

    Args:
        metadata_df: ECG metadata DataFrame
        outcome_label: Column name for outcome to stratify on
        n_folds: Number of CV folds
        random_state: Random seed for reproducibility

    Returns:
        List of (train_df, val_df) tuples for each fold
    """
    if "mrn" not in metadata_df.columns:
        raise ValueError("mrn column required for patient-level splitting")

    # Create patient-level aggregation for stratification
    patient_outcomes = metadata_df.groupby("mrn")[outcome_label].max().reset_index()
    patient_outcomes.columns = ["mrn", "patient_outcome"]

    # Create stratified folds at patient level
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)

    folds = []
    for fold_idx, (train_patient_idx, val_patient_idx) in enumerate(
        skf.split(patient_outcomes["mrn"], patient_outcomes["patient_outcome"])
    ):
        train_mrns = patient_outcomes.iloc[train_patient_idx]["mrn"].values
        val_mrns = patient_outcomes.iloc[val_patient_idx]["mrn"].values

        # Get ECGs for each fold
        train_df = metadata_df[metadata_df["mrn"].isin(train_mrns)].copy()
        val_df = metadata_df[metadata_df["mrn"].isin(val_mrns)].copy()

        folds.append((train_df, val_df))

        logger.info(
            f"Fold {fold_idx + 1}: {len(train_df)} train ECGs, {len(val_df)} val ECGs"
        )

    return folds


def split_train_val(
    metadata_df: pd.DataFrame,
    outcome_label: str,
    val_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Simple train/validation split at patient level.

    Args:
        metadata_df: ECG metadata DataFrame
        outcome_label: Column name for outcome to stratify on
        val_size: Proportion of patients for validation
        random_state: Random seed for reproducibility

    Returns:
        Tuple of (train_df, val_df)
    """
    if "mrn" not in metadata_df.columns:
        raise ValueError("mrn column required for patient-level splitting")

    # Create patient-level aggregation for stratification
    patient_outcomes = metadata_df.groupby("mrn")[outcome_label].max().reset_index()
    patient_outcomes.columns = ["mrn", "patient_outcome"]

    # Split patients
    train_mrns, val_mrns = train_test_split(
        patient_outcomes["mrn"],
        test_size=val_size,
        random_state=random_state,
        stratify=patient_outcomes["patient_outcome"],
    )

    # Split ECGs based on patient assignment
    train_df = metadata_df[metadata_df["mrn"].isin(train_mrns)].copy()
    val_df = metadata_df[metadata_df["mrn"].isin(val_mrns)].copy()

    logger.info(f"Train/Val split:")
    logger.info(f"  Train: {len(train_df)} ECGs from {len(train_mrns)} patients")
    logger.info(f"  Val: {len(val_df)} ECGs from {len(val_mrns)} patients")

    return train_df, val_df


def get_waveforms_for_split(
    metadata_df: pd.DataFrame, waveforms: np.ndarray, index_col: str = "index"
) -> np.ndarray:
    """
    Get waveforms corresponding to a metadata split.

    Args:
        metadata_df: ECG metadata DataFrame (after splitting)
        waveforms: Cohort-filtered waveform array
        index_col: Column name containing waveform indices

    Returns:
        Filtered waveform array
    """
    # For cohort-filtered data, use the DataFrame index (which should be 0-based after reset_index)
    # rather than the original 'index' column values
    row_indices = metadata_df.index.values
    return waveforms[row_indices]


def validate_no_patient_leakage(
    train_df: pd.DataFrame, val_df: pd.DataFrame, patient_col: str = "mrn"
) -> bool:
    """
    Validate that there's no patient leakage between train and validation sets.

    Args:
        train_df: Training set DataFrame
        val_df: Validation set DataFrame
        patient_col: Column name for patient identifier

    Returns:
        True if no leakage, False otherwise
    """
    train_patients = set(train_df[patient_col].unique())
    val_patients = set(val_df[patient_col].unique())

    overlap = train_patients.intersection(val_patients)

    if overlap:
        logger.error(
            f"Patient leakage detected! {len(overlap)} patients in both train and val sets"
        )
        logger.error(f"Overlapping patients: {list(overlap)[:10]}...")  # Show first 10
        return False
    else:
        logger.info("No patient leakage detected - validation passed")
        return True


class PatientLevelSplitter:
    """
    Class to handle all patient-level splitting operations.
    """

    def __init__(self, random_state: int = 42):
        self.random_state = random_state

    def create_splits(
        self,
        metadata_df: pd.DataFrame,
        waveforms: np.ndarray,
        outcome_label: str,
        holdout_size: float = 0.2,
        val_size: float = 0.2,
        use_cv: bool = False,
        n_folds: int = 5,
    ) -> Dict:
        """
        Create all necessary splits for the ML pipeline.

        Args:
            metadata_df: ECG metadata DataFrame
            waveforms: ECG waveform array
            outcome_label: Column name for outcome to stratify on
            holdout_size: Proportion for holdout set
            val_size: Proportion for validation (of remaining data)
            use_cv: Whether to create CV folds
            n_folds: Number of CV folds if use_cv=True

        Returns:
            Dictionary with all splits
        """
        logger.info(f"Creating splits for outcome: {outcome_label}")

        # Create holdout set (20% of all data)
        train_val_df, holdout_df = create_patient_level_holdout(
            metadata_df, outcome_label, holdout_size, self.random_state
        )

        # Get corresponding waveforms
        train_val_waveforms = get_waveforms_for_split(train_val_df, waveforms)
        holdout_waveforms = get_waveforms_for_split(holdout_df, waveforms)

        splits = {"holdout": {"metadata": holdout_df, "waveforms": holdout_waveforms}}

        if use_cv:
            # Create CV folds from remaining data
            cv_folds = create_patient_level_cv_folds(
                train_val_df, outcome_label, n_folds, self.random_state
            )

            splits["cv_folds"] = []
            for fold_idx, (train_df, val_df) in enumerate(cv_folds):
                train_waveforms = get_waveforms_for_split(train_df, waveforms)
                val_waveforms = get_waveforms_for_split(val_df, waveforms)

                # Validate no leakage
                validate_no_patient_leakage(train_df, val_df)

                splits["cv_folds"].append(
                    {
                        "fold": fold_idx,
                        "train": {"metadata": train_df, "waveforms": train_waveforms},
                        "val": {"metadata": val_df, "waveforms": val_waveforms},
                    }
                )
        else:
            # Simple train/val split
            train_df, val_df = split_train_val(
                train_val_df, outcome_label, val_size, self.random_state
            )

            train_waveforms = get_waveforms_for_split(train_df, waveforms)
            val_waveforms = get_waveforms_for_split(val_df, waveforms)

            # Validate no leakage
            validate_no_patient_leakage(train_df, val_df)

            splits["train"] = {"metadata": train_df, "waveforms": train_waveforms}
            splits["val"] = {"metadata": val_df, "waveforms": val_waveforms}

        return splits


class PatientBootstrapper:
    """
    Class to handle patient-level bootstrapping for minority class patients.
    Ensures no data leakage by bootstrapping at the patient level.
    """

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        np.random.seed(random_state)

    def bootstrap_minority_patients(
        self,
        metadata_df: pd.DataFrame,
        waveforms: np.ndarray,
        outcome_label: str,
        target_balance_ratio: float = 0.3,
        min_multiplier: int = 2,
        max_multiplier: int = 5,
    ) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Bootstrap minority class patients to improve class balance.

        Args:
            metadata_df: ECG metadata DataFrame
            waveforms: ECG waveform array
            outcome_label: Column name for outcome to bootstrap on
            target_balance_ratio: Target positive class ratio (0.3 = 30%)
            min_multiplier: Minimum number of times to duplicate minority patients
            max_multiplier: Maximum number of times to duplicate minority patients

        Returns:
            Tuple of (bootstrapped_metadata, bootstrapped_waveforms)
        """
        if "mrn" not in metadata_df.columns:
            raise ValueError("mrn column required for patient-level bootstrapping")

        # Calculate current class distribution
        current_positive_rate = metadata_df[outcome_label].mean()
        n_total = len(metadata_df)
        n_positive = metadata_df[outcome_label].sum()
        n_negative = n_total - n_positive

        logger.info(f"Current class distribution:")
        logger.info(f"  Positive: {n_positive}/{n_total} ({current_positive_rate:.3f})")
        logger.info(
            f"  Negative: {n_negative}/{n_total} ({1 - current_positive_rate:.3f})"
        )

        # If already balanced enough, return original data
        if current_positive_rate >= target_balance_ratio:
            logger.info(
                f"Already balanced (≥{target_balance_ratio:.3f}), skipping bootstrap"
            )
            return metadata_df, waveforms

        # Get minority class patients (those with positive outcomes)
        minority_patients = self._get_minority_patients(metadata_df, outcome_label)

        if len(minority_patients) == 0:
            logger.warning("No minority class patients found, skipping bootstrap")
            return metadata_df, waveforms

        # Calculate how many additional positive samples we need
        target_positive_count = int(
            n_negative * target_balance_ratio / (1 - target_balance_ratio)
        )
        additional_positive_needed = max(0, target_positive_count - n_positive)

        # Calculate bootstrap multiplier
        if additional_positive_needed == 0:
            logger.info("No additional positive samples needed")
            return metadata_df, waveforms

        multiplier = min(
            max_multiplier,
            max(
                min_multiplier,
                int(np.ceil(additional_positive_needed / len(minority_patients))),
            ),
        )

        logger.info(f"Bootstrapping strategy:")
        logger.info(f"  Target positive ratio: {target_balance_ratio:.3f}")
        logger.info(
            f"  Additional positive samples needed: {additional_positive_needed}"
        )
        logger.info(f"  Minority patients to bootstrap: {len(minority_patients)}")
        logger.info(f"  Bootstrap multiplier: {multiplier}")

        # Bootstrap minority patients
        bootstrapped_metadata, bootstrapped_waveforms = self._bootstrap_patients(
            metadata_df, waveforms, minority_patients, multiplier
        )

        # Validate result
        final_positive_rate = bootstrapped_metadata[outcome_label].mean()
        final_n_total = len(bootstrapped_metadata)
        final_n_positive = bootstrapped_metadata[outcome_label].sum()

        logger.info(f"After bootstrapping:")
        logger.info(f"  Total samples: {final_n_total} (+{final_n_total - n_total})")
        logger.info(
            f"  Positive: {final_n_positive}/{final_n_total} ({final_positive_rate:.3f})"
        )
        logger.info(
            f"  Improvement: {final_positive_rate - current_positive_rate:+.3f}"
        )

        return bootstrapped_metadata, bootstrapped_waveforms

    def _get_minority_patients(
        self, metadata_df: pd.DataFrame, outcome_label: str
    ) -> List[str]:
        """Get list of minority class patients (those with positive outcomes)."""
        # Get patients with at least one positive outcome
        patient_outcomes = metadata_df.groupby("mrn")[outcome_label].max()
        minority_patients = patient_outcomes[patient_outcomes == 1].index.tolist()

        logger.info(f"Identified {len(minority_patients)} minority class patients")
        return minority_patients

    def _bootstrap_patients(
        self,
        metadata_df: pd.DataFrame,
        waveforms: np.ndarray,
        minority_patients: List[str],
        multiplier: int,
    ) -> Tuple[pd.DataFrame, np.ndarray]:
        """Bootstrap specific patients by duplicating all their ECGs."""
        # Start with original data
        bootstrapped_metadata = metadata_df.copy()
        bootstrapped_waveforms = waveforms.copy()

        # For each minority patient, duplicate their ECGs (multiplier-1) times
        for patient_mrn in minority_patients:
            patient_ecgs = metadata_df[metadata_df["mrn"] == patient_mrn]
            patient_indices = patient_ecgs.index.values
            patient_waveforms = waveforms[patient_indices]

            # Duplicate this patient's data (multiplier-1) times
            for duplication_idx in range(multiplier - 1):
                # Create new metadata with modified indices
                duplicated_metadata = patient_ecgs.copy()

                # Add suffix to distinguish duplicated entries (for debugging)
                duplicated_metadata["mrn"] = (
                    duplicated_metadata["mrn"].astype(str)
                    + f"_dup_{duplication_idx + 1}"
                )

                # Reset index to avoid conflicts
                duplicated_metadata = duplicated_metadata.reset_index(drop=True)

                # Append to bootstrapped data
                bootstrapped_metadata = pd.concat(
                    [bootstrapped_metadata, duplicated_metadata], ignore_index=True
                )
                bootstrapped_waveforms = np.vstack(
                    [bootstrapped_waveforms, patient_waveforms]
                )

        logger.info(f"Bootstrap complete: {len(bootstrapped_metadata)} total ECGs")
        return bootstrapped_metadata, bootstrapped_waveforms

    def validate_bootstrap_integrity(
        self,
        original_metadata: pd.DataFrame,
        bootstrapped_metadata: pd.DataFrame,
        outcome_label: str,
    ) -> bool:
        """
        Validate that bootstrapping maintains patient integrity.

        Args:
            original_metadata: Original metadata before bootstrapping
            bootstrapped_metadata: Metadata after bootstrapping
            outcome_label: Outcome column to validate

        Returns:
            True if validation passes, False otherwise
        """
        logger.info("Validating bootstrap integrity...")

        # Check 1: No new unique patients should be created (only duplicates)
        original_base_mrns = set(
            str(mrn).split("_dup_")[0] for mrn in original_metadata["mrn"].unique()
        )
        bootstrapped_base_mrns = set(
            str(mrn).split("_dup_")[0] for mrn in bootstrapped_metadata["mrn"].unique()
        )

        if original_base_mrns != bootstrapped_base_mrns:
            logger.error("Bootstrap integrity check failed: New patients detected")
            return False

        # Check 2: All original ECGs should be preserved
        original_count = len(original_metadata)
        original_ecgs_in_bootstrap = len(
            bootstrapped_metadata[
                ~bootstrapped_metadata["mrn"]
                .astype(str)
                .str.contains("_dup_", na=False)
            ]
        )

        if original_count != original_ecgs_in_bootstrap:
            logger.error(
                "Bootstrap integrity check failed: Original ECGs not preserved"
            )
            return False

        # Check 3: Only minority class patients should be duplicated
        duplicated_mrns = (
            bootstrapped_metadata[
                bootstrapped_metadata["mrn"].astype(str).str.contains("_dup_", na=False)
            ]["mrn"]
            .apply(lambda x: str(x).split("_dup_")[0])
            .unique()
        )

        for mrn in duplicated_mrns:
            # Convert mrn to appropriate type for comparison
            patient_outcomes = original_metadata[
                original_metadata["mrn"].astype(str) == str(mrn)
            ][outcome_label]
            if len(patient_outcomes) > 0 and patient_outcomes.max() != 1:
                logger.error(
                    f"Bootstrap integrity check failed: Non-minority patient {mrn} was duplicated"
                )
                return False

        logger.info("Bootstrap integrity validation passed")
        return True


if __name__ == "__main__":
    # Test splitting
    logging.basicConfig(level=logging.INFO)

    from data_loading import load_ecg_data
    from preprocess import (
        filter_sinus_rhythm,
        create_pre_ecg_labels,
        create_outcome_labels,
    )

    # Load and preprocess data
    metadata, waveforms = load_ecg_data()
    sinus_df = filter_sinus_rhythm(metadata)
    processed_df = create_pre_ecg_labels(sinus_df)
    processed_df = create_outcome_labels(processed_df)

    # Test splitting
    splitter = PatientLevelSplitter(random_state=42)

    # Get a cohort for testing
    from preprocess import get_cohort_for_window_and_outcome

    cohort_df, cohort_waveforms = get_cohort_for_window_and_outcome(
        processed_df, waveforms, "pre_ecg_1y", "af_recurrence_1y"
    )

    # Test bootstrapping
    bootstrapper = PatientBootstrapper(random_state=42)
    boot_df, boot_waveforms = bootstrapper.bootstrap_minority_patients(
        cohort_df, cohort_waveforms, "af_recurrence_1y"
    )

    # Validate bootstrap
    bootstrapper.validate_bootstrap_integrity(cohort_df, boot_df, "af_recurrence_1y")

    # Create splits
    splits = splitter.create_splits(
        boot_df, boot_waveforms, "af_recurrence_1y", use_cv=True, n_folds=3
    )

    print(f"Created splits with {len(splits['cv_folds'])} CV folds")
    print(f"Holdout set: {len(splits['holdout']['metadata'])} ECGs")
