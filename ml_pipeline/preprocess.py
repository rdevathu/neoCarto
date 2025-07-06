#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Preprocessing module for AF recurrence prediction.
Applies cohort selection criteria and generates outcome labels.
"""

import pandas as pd
import numpy as np
import logging
from typing import List, Dict, Tuple, Optional
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


def filter_sinus_rhythm(metadata_df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter ECGs to only include sinus rhythm recordings.

    Args:
        metadata_df: ECG metadata DataFrame

    Returns:
        Filtered DataFrame with only sinus rhythm ECGs
    """
    if "sinus_rhythm" not in metadata_df.columns:
        raise ValueError("sinus_rhythm column not found in metadata")

    initial_count = len(metadata_df)
    sinus_df = metadata_df[metadata_df["sinus_rhythm"] == 1].copy()

    logger.info(
        f"Filtered to sinus rhythm: {len(sinus_df)} / {initial_count} ECGs ({len(sinus_df) / initial_count * 100:.1f}%)"
    )

    return sinus_df


def create_pre_ecg_labels(
    metadata_df: pd.DataFrame,
    time_windows: List[int] = [365, 1095, 1825],  # 1y, 3y, 5y in days
    include_same_day: bool = True,
) -> pd.DataFrame:
    """
    Create pre-procedure ECG labels for different time windows.

    Args:
        metadata_df: ECG metadata DataFrame
        time_windows: List of time windows in days before procedure
        include_same_day: Whether to include ECGs from the same day as procedure

    Returns:
        DataFrame with additional pre_ecg_* columns
    """
    df = metadata_df.copy()

    # Check required columns
    required_cols = ["mrn", "acquisition_date", "procedure_date"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Remove rows with missing procedure dates
    initial_count = len(df)
    df = df.dropna(subset=["procedure_date"])
    logger.info(f"Removed {initial_count - len(df)} ECGs with missing procedure dates")

    # Create pre-ECG labels for each time window
    for window_days in time_windows:
        window_years = window_days / 365.25
        col_name = f"pre_ecg_{window_years:.0f}y"

        # Calculate time difference
        df["time_diff_days"] = (df["procedure_date"] - df["acquisition_date"]).dt.days

        # Mark ECGs as pre-procedure if they are:
        # 1. Before or on the procedure date (time_diff_days >= 0)
        # 2. Within the specified time window (time_diff_days <= window_days)
        if include_same_day:
            df[col_name] = (
                (df["time_diff_days"] >= 0) & (df["time_diff_days"] <= window_days)
            ).astype(int)
        else:
            df[col_name] = (
                (df["time_diff_days"] > 0) & (df["time_diff_days"] <= window_days)
            ).astype(int)

        logger.info(
            f"Created {col_name}: {df[col_name].sum()} ECGs marked as pre-procedure"
        )

    # Clean up temporary column
    df = df.drop("time_diff_days", axis=1)

    return df


def create_outcome_labels(
    metadata_df: pd.DataFrame,
    outcome_types: List[str] = ["af_recurrence", "at_recurrence", "af_at_recurrence"],
    time_thresholds: List[int] = [365, 1095, 1825],  # 1y, 3y, 5y in days
    include_any: bool = True,
) -> pd.DataFrame:
    """
    Create binary outcome labels for different time thresholds.

    Args:
        metadata_df: ECG metadata DataFrame
        outcome_types: List of outcome types to process
        time_thresholds: List of time thresholds in days
        include_any: Whether to include "any" time threshold (no time limit)

    Returns:
        DataFrame with additional outcome_* columns
    """
    df = metadata_df.copy()

    for outcome_type in outcome_types:
        outcome_col = outcome_type
        days_col = f"days_till_{outcome_type}"

        if outcome_col not in df.columns:
            logger.warning(f"Outcome column {outcome_col} not found, skipping")
            continue

        # Create labels for each time threshold
        for threshold_days in time_thresholds:
            threshold_years = threshold_days / 365.25
            label_col = f"{outcome_type}_{threshold_years:.0f}y"

            if days_col in df.columns:
                # Use days_till_* column if available
                df[label_col] = (
                    (df[outcome_col].fillna(0) == 1)
                    & (df[days_col].fillna(float("inf")) <= threshold_days)
                ).astype(int)
            else:
                # Fallback to just the outcome column
                df[label_col] = df[outcome_col].fillna(0).astype(int)

            logger.info(f"Created {label_col}: {df[label_col].sum()} positive cases")

        # Create "any" time threshold if requested
        if include_any:
            any_col = f"{outcome_type}_any"
            df[any_col] = df[outcome_col].fillna(0).astype(int)
            logger.info(f"Created {any_col}: {df[any_col].sum()} positive cases")

    return df


def get_cohort_for_window_and_outcome(
    metadata_df: pd.DataFrame,
    waveforms: np.ndarray,
    pre_ecg_window: str,
    outcome_label: str,
    min_ecgs_per_mrn: int = 1,
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Get cohort for a specific pre-ECG window and outcome label.

    Args:
        metadata_df: ECG metadata DataFrame
        waveforms: ECG waveform array
        pre_ecg_window: Pre-ECG window column name (e.g., 'pre_ecg_1y')
        outcome_label: Outcome label column name (e.g., 'af_recurrence_1y')
        min_ecgs_per_mrn: Minimum number of ECGs per MRN to include

    Returns:
        Tuple of (filtered_metadata, filtered_waveforms)
    """
    # Check required columns
    required_cols = [pre_ecg_window, outcome_label, "index"]
    missing_cols = [col for col in required_cols if col not in metadata_df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Filter to pre-ECGs
    pre_ecg_df = metadata_df[metadata_df[pre_ecg_window] == 1].copy()
    logger.info(f"Filtered to {pre_ecg_window}: {len(pre_ecg_df)} ECGs")

    # Filter out ECGs with missing outcome labels
    pre_ecg_df = pre_ecg_df.dropna(subset=[outcome_label])
    logger.info(f"After removing missing outcomes: {len(pre_ecg_df)} ECGs")

    # Apply minimum ECGs per MRN filter if specified
    if min_ecgs_per_mrn > 1:
        mrn_counts = pre_ecg_df["mrn"].value_counts()
        valid_mrns = mrn_counts[mrn_counts >= min_ecgs_per_mrn].index
        pre_ecg_df = pre_ecg_df[pre_ecg_df["mrn"].isin(valid_mrns)]
        logger.info(
            f"After min ECGs per MRN filter: {len(pre_ecg_df)} ECGs from {len(valid_mrns)} MRNs"
        )

    # Get corresponding waveforms
    ecg_indices = pre_ecg_df["index"].values
    filtered_waveforms = waveforms[ecg_indices]

    # Reset index to create clean 0-based indexing for the filtered cohort
    pre_ecg_df = pre_ecg_df.reset_index(drop=True)

    # Print class distribution
    class_counts = pre_ecg_df[outcome_label].value_counts().sort_index()
    logger.info(f"Class distribution for {outcome_label}: {dict(class_counts)}")

    return pre_ecg_df, filtered_waveforms


def get_cohort_summary(
    metadata_df: pd.DataFrame, pre_ecg_windows: List[str], outcome_labels: List[str]
) -> pd.DataFrame:
    """
    Get summary of cohort sizes for all combinations of windows and outcomes.

    Args:
        metadata_df: ECG metadata DataFrame
        pre_ecg_windows: List of pre-ECG window column names
        outcome_labels: List of outcome label column names

    Returns:
        DataFrame with cohort summary statistics
    """
    summary_data = []

    for window in pre_ecg_windows:
        for outcome in outcome_labels:
            if window in metadata_df.columns and outcome in metadata_df.columns:
                # Get cohort
                cohort_df = metadata_df[metadata_df[window] == 1]
                cohort_df = cohort_df.dropna(subset=[outcome])

                if len(cohort_df) > 0:
                    n_ecgs = len(cohort_df)
                    n_mrns = cohort_df["mrn"].nunique()
                    n_positive = cohort_df[outcome].sum()
                    positive_rate = n_positive / n_ecgs if n_ecgs > 0 else 0

                    summary_data.append(
                        {
                            "window": window,
                            "outcome": outcome,
                            "n_ecgs": n_ecgs,
                            "n_mrns": n_mrns,
                            "n_positive": n_positive,
                            "positive_rate": positive_rate,
                        }
                    )

    return pd.DataFrame(summary_data)


if __name__ == "__main__":
    # Test preprocessing
    logging.basicConfig(level=logging.INFO)

    from data_loading import load_ecg_data

    metadata, waveforms = load_ecg_data()

    # Filter to sinus rhythm
    sinus_df = filter_sinus_rhythm(metadata)

    # Create pre-ECG labels
    processed_df = create_pre_ecg_labels(sinus_df)

    # Create outcome labels
    processed_df = create_outcome_labels(processed_df)

    # Get cohort summary
    pre_ecg_windows = [
        col for col in processed_df.columns if col.startswith("pre_ecg_")
    ]
    outcome_labels = [
        col
        for col in processed_df.columns
        if col.endswith(("_1y", "_3y", "_5y", "_any"))
    ]

    summary = get_cohort_summary(processed_df, pre_ecg_windows, outcome_labels)
    print("\nCohort Summary:")
    print(summary.to_string(index=False))
