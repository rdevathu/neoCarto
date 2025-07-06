#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Data loading module for AF recurrence prediction.
Loads and merges ECG metadata with waveform data.
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Tuple, Optional

logger = logging.getLogger(__name__)


def load_ecg_data(
    metadata_path: str = "carto_ecg_metadata_FULL.csv",
    waveforms_path: str = "carto_ecg_waveforms.npy",
    validate_indices: bool = True,
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Load ECG metadata and waveform data.

    Args:
        metadata_path: Path to CSV file with ECG metadata
        waveforms_path: Path to NPY file with waveform data
        validate_indices: Whether to validate that indices match

    Returns:
        Tuple of (metadata_df, waveforms_array)
    """
    logger.info(f"Loading metadata from {metadata_path}")
    metadata_df = pd.read_csv(metadata_path)

    logger.info(f"Loading waveforms from {waveforms_path}")
    waveforms = np.load(waveforms_path)

    logger.info(
        f"Loaded {len(metadata_df)} metadata records and {len(waveforms)} waveforms"
    )

    # Convert acquisition_date to datetime
    metadata_df["acquisition_date"] = pd.to_datetime(metadata_df["acquisition_date"])

    # Convert procedure_date to datetime if it exists and is not null
    if "procedure_date" in metadata_df.columns:
        metadata_df["procedure_date"] = pd.to_datetime(
            metadata_df["procedure_date"], errors="coerce"
        )

    if validate_indices:
        # Validate that indices match
        if len(metadata_df) != len(waveforms):
            raise ValueError(
                f"Mismatch: {len(metadata_df)} metadata records vs {len(waveforms)} waveforms"
            )

        # Check that index column matches array indices
        if "index" in metadata_df.columns:
            max_index = metadata_df["index"].max()
            if max_index >= len(waveforms):
                raise ValueError(
                    f"Index {max_index} exceeds waveform array length {len(waveforms)}"
                )

    logger.info(
        f"Data validation passed. Shape: metadata={metadata_df.shape}, waveforms={waveforms.shape}"
    )

    return metadata_df, waveforms


def get_data_summary(metadata_df: pd.DataFrame, waveforms: np.ndarray) -> dict:
    """
    Get summary statistics about the loaded data.

    Args:
        metadata_df: ECG metadata DataFrame
        waveforms: ECG waveform array

    Returns:
        Dictionary with summary statistics
    """
    summary = {
        "total_ecgs": len(metadata_df),
        "waveform_shape": waveforms.shape,
        "unique_mrns": metadata_df["mrn"].nunique()
        if "mrn" in metadata_df.columns
        else 0,
        "sinus_rhythm_count": metadata_df["sinus_rhythm"].sum()
        if "sinus_rhythm" in metadata_df.columns
        else 0,
        "date_range": {
            "min": metadata_df["acquisition_date"].min(),
            "max": metadata_df["acquisition_date"].max(),
        },
    }

    # Add outcome statistics if available
    outcome_cols = ["af_recurrence", "at_recurrence", "af_at_recurrence"]
    for col in outcome_cols:
        if col in metadata_df.columns:
            summary[f"{col}_count"] = metadata_df[col].sum()

    return summary


if __name__ == "__main__":
    # Test the data loading
    logging.basicConfig(level=logging.INFO)

    metadata, waveforms = load_ecg_data()
    summary = get_data_summary(metadata, waveforms)

    print("Data Summary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")
