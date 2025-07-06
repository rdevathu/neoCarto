#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Augmentation module for AF recurrence prediction.
Handles sliding window augmentation of ECG data.
"""

import numpy as np
import pandas as pd
import logging
from typing import Tuple, List, Optional

logger = logging.getLogger(__name__)


def create_sliding_windows(
    waveforms: np.ndarray,
    window_size: int = 1250,  # 5 seconds at 250 Hz
    overlap: int = 1000,  # 4 seconds overlap
    sampling_rate: int = 250,
) -> Tuple[np.ndarray, List[int]]:
    """
    Create sliding windows from ECG waveforms.

    Args:
        waveforms: ECG waveform array of shape (n_ecgs, n_samples, n_leads)
        window_size: Size of each window in samples
        overlap: Overlap between windows in samples
        sampling_rate: Sampling rate in Hz (for validation)

    Returns:
        Tuple of (windowed_waveforms, original_indices)
        - windowed_waveforms: shape (n_windows, window_size, n_leads)
        - original_indices: list mapping each window back to original ECG index
    """
    n_ecgs, n_samples, n_leads = waveforms.shape

    # Validate input parameters
    if window_size > n_samples:
        raise ValueError(
            f"Window size {window_size} larger than signal length {n_samples}"
        )

    if overlap >= window_size:
        raise ValueError(
            f"Overlap {overlap} must be less than window size {window_size}"
        )

    # Calculate step size and number of windows per ECG
    step_size = window_size - overlap
    n_windows_per_ecg = (n_samples - window_size) // step_size + 1

    logger.info(f"Creating sliding windows:")
    logger.info(
        f"  Window size: {window_size} samples ({window_size / sampling_rate:.1f} seconds)"
    )
    logger.info(f"  Overlap: {overlap} samples ({overlap / sampling_rate:.1f} seconds)")
    logger.info(
        f"  Step size: {step_size} samples ({step_size / sampling_rate:.1f} seconds)"
    )
    logger.info(f"  Windows per ECG: {n_windows_per_ecg}")

    # Pre-allocate arrays
    total_windows = n_ecgs * n_windows_per_ecg
    windowed_waveforms = np.zeros(
        (total_windows, window_size, n_leads), dtype=waveforms.dtype
    )
    original_indices = []

    # Create windows
    window_idx = 0
    for ecg_idx in range(n_ecgs):
        for win_start in range(0, n_samples - window_size + 1, step_size):
            win_end = win_start + window_size
            windowed_waveforms[window_idx] = waveforms[ecg_idx, win_start:win_end, :]
            original_indices.append(ecg_idx)
            window_idx += 1

    logger.info(f"Created {len(windowed_waveforms)} windows from {n_ecgs} ECGs")

    return windowed_waveforms, original_indices


def augment_metadata(
    metadata_df: pd.DataFrame,
    original_indices: List[int],
    window_suffix: str = "_window",
) -> pd.DataFrame:
    """
    Augment metadata to match windowed waveforms.

    Args:
        metadata_df: Original metadata DataFrame
        original_indices: List mapping each window back to original ECG index
        window_suffix: Suffix to add to identify windowed samples

    Returns:
        Augmented metadata DataFrame
    """
    # Create new metadata by replicating rows based on original_indices
    augmented_rows = []

    for window_idx, orig_idx in enumerate(original_indices):
        # Get original row
        orig_row = metadata_df.iloc[orig_idx].copy()

        # Add window information
        orig_row["window_idx"] = window_idx
        orig_row["original_idx"] = orig_idx
        orig_row["is_augmented"] = True

        # Modify filename to indicate windowing
        if "filename" in orig_row:
            orig_row["filename"] = f"{orig_row['filename']}{window_suffix}_{window_idx}"

        augmented_rows.append(orig_row)

    augmented_df = pd.DataFrame(augmented_rows)

    logger.info(
        f"Augmented metadata: {len(augmented_df)} rows from {len(metadata_df)} original ECGs"
    )

    return augmented_df


def apply_augmentation(
    metadata_df: pd.DataFrame,
    waveforms: np.ndarray,
    window_size: int = 1250,
    overlap: int = 1000,
    sampling_rate: int = 250,
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Apply sliding window augmentation to both metadata and waveforms.

    Args:
        metadata_df: ECG metadata DataFrame
        waveforms: ECG waveform array
        window_size: Size of each window in samples
        overlap: Overlap between windows in samples
        sampling_rate: Sampling rate in Hz

    Returns:
        Tuple of (augmented_metadata, augmented_waveforms)
    """
    # Create sliding windows
    windowed_waveforms, original_indices = create_sliding_windows(
        waveforms, window_size, overlap, sampling_rate
    )

    # Augment metadata
    augmented_metadata = augment_metadata(metadata_df, original_indices)

    # Validate shapes match
    if len(augmented_metadata) != len(windowed_waveforms):
        raise ValueError(
            f"Metadata and waveform lengths don't match: {len(augmented_metadata)} vs {len(windowed_waveforms)}"
        )

    return augmented_metadata, windowed_waveforms


def select_lead_subset(waveforms: np.ndarray, lead_indices: List[int]) -> np.ndarray:
    """
    Select a subset of leads from the waveform data.

    Args:
        waveforms: ECG waveform array of shape (n_ecgs, n_samples, n_leads)
        lead_indices: List of lead indices to select (0-based)

    Returns:
        Filtered waveform array of shape (n_ecgs, n_samples, len(lead_indices))
    """
    if max(lead_indices) >= waveforms.shape[2]:
        raise ValueError(
            f"Lead index {max(lead_indices)} exceeds available leads {waveforms.shape[2]}"
        )

    selected_waveforms = waveforms[:, :, lead_indices]

    logger.info(
        f"Selected leads {lead_indices}: shape {waveforms.shape} -> {selected_waveforms.shape}"
    )

    return selected_waveforms


def get_lead_1_only(waveforms: np.ndarray) -> np.ndarray:
    """
    Extract only lead I (index 0) from the waveform data.

    Args:
        waveforms: ECG waveform array of shape (n_ecgs, n_samples, n_leads)

    Returns:
        Lead I waveform array of shape (n_ecgs, n_samples, 1)
    """
    return select_lead_subset(waveforms, [0])


def get_all_leads(waveforms: np.ndarray) -> np.ndarray:
    """
    Return all 12 leads (no filtering).

    Args:
        waveforms: ECG waveform array of shape (n_ecgs, n_samples, 12)

    Returns:
        Same waveform array
    """
    if waveforms.shape[2] != 12:
        logger.warning(f"Expected 12 leads, got {waveforms.shape[2]}")

    return waveforms


class ECGAugmenter:
    """
    Class to handle ECG data augmentation.
    """

    def __init__(
        self,
        window_size: int = 1250,  # 5 seconds at 250 Hz
        overlap: int = 1000,  # 4 seconds overlap
        sampling_rate: int = 250,
    ):
        self.window_size = window_size
        self.overlap = overlap
        self.sampling_rate = sampling_rate

        # Validate parameters
        if self.overlap >= self.window_size:
            raise ValueError("Overlap must be less than window size")

    def augment_training_data(
        self,
        train_metadata: pd.DataFrame,
        train_waveforms: np.ndarray,
        enable_augmentation: bool = True,
    ) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Augment training data with sliding windows.

        Args:
            train_metadata: Training metadata DataFrame
            train_waveforms: Training waveform array
            enable_augmentation: Whether to apply augmentation

        Returns:
            Tuple of (augmented_metadata, augmented_waveforms)
        """
        if not enable_augmentation:
            logger.info("Augmentation disabled - returning original data")
            return train_metadata, train_waveforms

        logger.info("Applying sliding window augmentation to training data")

        return apply_augmentation(
            train_metadata,
            train_waveforms,
            self.window_size,
            self.overlap,
            self.sampling_rate,
        )

    def prepare_lead_configuration(
        self, waveforms: np.ndarray, lead_config: str = "all"
    ) -> np.ndarray:
        """
        Prepare waveforms for specific lead configuration.

        Args:
            waveforms: ECG waveform array
            lead_config: Either "lead1" or "all"

        Returns:
            Configured waveform array
        """
        if lead_config == "lead1":
            return get_lead_1_only(waveforms)
        elif lead_config == "all":
            return get_all_leads(waveforms)
        else:
            raise ValueError(f"Unknown lead configuration: {lead_config}")


if __name__ == "__main__":
    # Test augmentation
    logging.basicConfig(level=logging.INFO)

    # Create dummy data for testing
    n_ecgs, n_samples, n_leads = 10, 2500, 12
    dummy_waveforms = np.random.randn(n_ecgs, n_samples, n_leads)
    dummy_metadata = pd.DataFrame(
        {
            "filename": [f"ecg_{i}.xml" for i in range(n_ecgs)],
            "mrn": [f"mrn_{i}" for i in range(n_ecgs)],
            "outcome": np.random.randint(0, 2, n_ecgs),
        }
    )

    # Test augmentation
    augmenter = ECGAugmenter()

    # Test sliding windows
    aug_metadata, aug_waveforms = augmenter.augment_training_data(
        dummy_metadata, dummy_waveforms, apply_augmentation=True
    )

    print(f"Original: {dummy_waveforms.shape}")
    print(f"Augmented: {aug_waveforms.shape}")
    print(f"Metadata: {len(dummy_metadata)} -> {len(aug_metadata)}")

    # Test lead configurations
    lead1_waveforms = augmenter.prepare_lead_configuration(dummy_waveforms, "lead1")
    all_leads_waveforms = augmenter.prepare_lead_configuration(dummy_waveforms, "all")

    print(f"Lead 1 only: {lead1_waveforms.shape}")
    print(f"All leads: {all_leads_waveforms.shape}")
