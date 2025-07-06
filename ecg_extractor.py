#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to extract metadata and waveform data from ECGs in the carto folder.
This script extracts metadata (acquisition_date, interpretation, sinus_rhythm) and
waveform data from ECG XML files, storing metadata in a CSV file and waveforms in a NPY file.
"""

import os
import pandas as pd
import numpy as np
import logging
from datetime import datetime
import glob
import xml.etree.ElementTree as ET
import sierraecg as se
import neurokit2 as nk
from tqdm import tqdm
from pathlib import Path
from typing import List, Tuple, Optional
import concurrent.futures

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def log_progress(message):
    """Log progress to a Progress.txt file"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if not os.path.exists("Progress.txt"):
        with open("Progress.txt", "w") as f:
            f.write(f"{timestamp} - Initial creation of Progress.txt file\n---\n")

    with open("Progress.txt", "a") as f:
        f.write(f"{timestamp} - {message}\n---\n")


def get_ecg_files():
    """Get a list of the ECG files in the carto directory."""
    ecg_files_path = os.path.join("carto", "*.xml")
    ecg_files = glob.glob(ecg_files_path)

    if not ecg_files:
        logger.warning("No ECG files found in the carto directory.")
    else:
        logger.info(f"Found {len(ecg_files)} ECG files in the carto directory.")

    return ecg_files


def extract_ecg_metadata(file_path):
    """
    Extract metadata from an ECG XML file.

    Args:
        file_path (str): Path to the ECG XML file

    Returns:
        tuple: (acquisition_date, interpretation, sinus_rhythm)
    """
    try:
        tree = ET.parse(file_path)
        root = tree.getroot()

        # Handle namespaces in tags
        namespace = ""
        if root.tag.startswith("{"):
            namespace = root.tag.split("}")[0] + "}"

        # Extract acquisition date
        acquisition_elem = root.find("./{}dataacquisition".format(namespace))
        acquisition_date = None
        if acquisition_elem is not None:
            date_str = acquisition_elem.get("date")
            time_str = acquisition_elem.get("time")
            if date_str and time_str:
                acquisition_date = f"{date_str} {time_str}"

        # Extract interpretation
        interpretations_elem = root.find("./{}interpretations".format(namespace))
        interpretation_text = ""
        sinus_rhythm = False

        if interpretations_elem is not None:
            # Convert interpretations_elem to string to search for sinus rhythm
            interpretations_str = ET.tostring(
                interpretations_elem, encoding="unicode", method="xml"
            )

            # Check for sinus rhythm patterns in the full text
            sinus_rhythm = any(
                pattern.lower() in interpretations_str.lower()
                for pattern in [
                    "sinus rhythm",
                    "sinus tachycardia",
                    "sinus bradycardia",
                ]
            )

            # Now extract clean interpretation statements
            try:
                # Find all statement elements
                statements = interpretations_elem.findall(
                    ".//{}statement".format(namespace)
                )
                statements_list = []

                for stmt in statements:
                    left = stmt.find("{}leftstatement".format(namespace))
                    right = stmt.find("{}rightstatement".format(namespace))

                    left_text = left.text if left is not None and left.text else ""
                    right_text = right.text if right is not None and right.text else ""

                    if left_text is None or right_text is None:
                        left_text = ""
                        right_text = ""

                    statement = (left_text + " " + right_text).strip()
                    if statement:
                        statements_list.append(statement)

                interpretation_text = "\n".join(statements_list)
            except Exception as e:
                logger.error(f"Error extracting statements from {file_path}: {str(e)}")

        return acquisition_date, interpretation_text, sinus_rhythm

    except Exception as e:
        logger.error(f"Error extracting metadata from {file_path}: {str(e)}")
        return None, None, False


def process_ecg_waveform(
    file_path, target_sampling_rate=250, target_duration_seconds=10, padding_value=0.0
):
    """
    Process a single ECG file to extract waveform data.

    Args:
        file_path (str): Path to the ECG XML file
        target_sampling_rate (int): Desired sampling rate in Hz
        target_duration_seconds (int): Desired duration of each ECG in seconds
        padding_value (float): Value to use for padding when resampling

    Returns:
        tuple: (waveform_data, original_sampling_rate) or (None, None) if error
    """
    try:
        # Read ECG with sierraecg
        ecg = se.read_file(file_path)

        # Store original sampling rate
        original_rate = ecg.leads[0].sampling_freq

        # Calculate target samples
        target_samples = target_sampling_rate * target_duration_seconds

        # Initialize array for this ECG's 12 leads
        ecg_leads = np.full((target_samples, 12), padding_value)

        # Process each lead
        for i in range(12):
            lead_data = ecg.leads[i].samples
            # Resample each lead to target sampling rate
            resampled_lead = nk.signal_resample(
                lead_data,
                sampling_rate=ecg.leads[i].sampling_freq,
                desired_sampling_rate=target_sampling_rate,
            )
            # Ensure we get exactly target_samples, padding if needed
            if len(resampled_lead) >= target_samples:
                ecg_leads[:, i] = resampled_lead[:target_samples]
            else:
                ecg_leads[: len(resampled_lead), i] = resampled_lead

        return ecg_leads, original_rate

    except Exception as e:
        logger.error(f"Error processing waveform from {file_path}: {str(e)}")
        return None, None


def process_single_ecg(file_path):
    """
    Process a single ECG file: extract metadata and waveform.
    Returns a tuple (metadata_dict, waveform, original_rate) or None if error.
    """
    try:
        filename = os.path.basename(file_path)
        acquisition_date, interpretation, sinus_rhythm = extract_ecg_metadata(file_path)
        waveform, original_rate = process_ecg_waveform(file_path)
        if waveform is not None and acquisition_date is not None:
            metadata = {
                "filename": filename,
                "acquisition_date": acquisition_date,
                "interpretation": interpretation,
                "sinus_rhythm": 1 if sinus_rhythm else 0,
            }
            return (metadata, waveform, original_rate)
        else:
            return None
    except Exception as e:
        return None


def process_carto_ecgs():
    """
    Process all ECGs in the carto folder, extracting metadata and waveform data in parallel.
    Saves metadata to CSV and waveforms to NPY file with matching indices.
    """
    logger.info("Starting CARTO ECG processing")
    log_progress("Starting CARTO ECG processing")

    ecg_files = get_ecg_files()
    if not ecg_files:
        logger.error("No ECG files found to process")
        log_progress("ERROR: No ECG files found to process")
        return

    metadata_list = []
    waveform_data = []
    original_sampling_rates = []

    logger.info(f"Processing {len(ecg_files)} ECG files in parallel...")

    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = list(
            tqdm(
                executor.map(process_single_ecg, ecg_files),
                total=len(ecg_files),
                desc="Processing ECGs",
            )
        )

    for i, result in enumerate(results):
        if result is not None:
            metadata, waveform, original_rate = result
            metadata["index"] = i
            metadata_list.append(metadata)
            waveform_data.append(waveform)
            original_sampling_rates.append(original_rate)

    if waveform_data:
        waveform_array = np.array(waveform_data)
        metadata_df = pd.DataFrame(metadata_list)
        metadata_csv_path = "carto_ecg_metadata.csv"
        metadata_df.to_csv(metadata_csv_path, index=False)
        logger.info(f"Saved metadata to {metadata_csv_path}")
        log_progress(
            f"Saved metadata for {len(metadata_list)} ECGs to {metadata_csv_path}"
        )
        waveform_npy_path = "carto_ecg_waveforms.npy"
        np.save(waveform_npy_path, waveform_array)
        logger.info(f"Saved waveform data to {waveform_npy_path}")
        log_progress(
            f"Saved waveform data for {len(waveform_data)} ECGs to {waveform_npy_path}"
        )
        sampling_rates_path = "carto_ecg_sampling_rates.npy"
        np.save(sampling_rates_path, np.array(original_sampling_rates))
        logger.info(f"Saved original sampling rates to {sampling_rates_path}")
        logger.info(
            f"Successfully processed {len(metadata_list)} out of {len(ecg_files)} ECG files"
        )
        log_progress(
            f"Successfully processed {len(metadata_list)} out of {len(ecg_files)} ECG files"
        )
    else:
        logger.error("No ECG data was successfully processed")
        log_progress("ERROR: No ECG data was successfully processed")


def main():
    """Main function to process CARTO ECGs"""
    logger.info("Starting CARTO ECG processing")
    log_progress("Starting CARTO ECG processing")

    # Check if carto directory exists
    if not os.path.exists("carto"):
        error_msg = "carto directory not found"
        logger.error(error_msg)
        log_progress(f"ERROR: {error_msg}")
        raise FileNotFoundError(error_msg)

    # Process the ECGs
    process_carto_ecgs()

    logger.info("CARTO ECG processing completed")
    log_progress("CARTO ECG processing completed")


if __name__ == "__main__":
    main()
