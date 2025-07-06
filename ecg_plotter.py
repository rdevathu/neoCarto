#!/usr/bin/env python3
"""
ECG Plotter using Sierra ECG Library
Plots 12-lead ECGs in standard format for sinus and non-sinus rhythm examples
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sierraecg as se
from pathlib import Path


def plot_12_lead_ecg(ecg_data, title="12-Lead ECG", figsize=(20, 12)):
    """
    Plot 12-lead ECG in standard format

    Parameters:
    -----------
    ecg_data : SierraEcgFile
        ECG data loaded from sierraecg
    title : str
        Title for the plot
    figsize : tuple
        Figure size (width, height)
    """

    # Standard 12-lead ECG layout (3 rows, 4 columns)
    # Row 1: I, aVR, V1, V4
    # Row 2: II, aVL, V2, V5
    # Row 3: III, aVF, V3, V6

    fig, axes = plt.subplots(3, 4, figsize=figsize)
    fig.suptitle(title, fontsize=16, fontweight="bold")

    # Lead labels in correct standard order for 3x4 layout
    lead_labels = [
        "I",
        "aVR",
        "V1",
        "V4",
        "II",
        "aVL",
        "V2",
        "V5",
        "III",
        "aVF",
        "V3",
        "V6",
    ]

    # Plot each lead
    for i, (ax, lead_label) in enumerate(zip(axes.flat, lead_labels)):
        # Find the lead data
        lead_data = None
        for lead in ecg_data.leads:
            if lead.label == lead_label:
                lead_data = lead
                break

        if lead_data is not None:
            # Convert samples to time axis
            time_axis = np.arange(len(lead_data.samples)) / lead_data.sampling_freq

            # Plot the ECG signal
            ax.plot(time_axis, lead_data.samples, "b-", linewidth=0.8)
            ax.set_title(f"Lead {lead_label}", fontweight="bold")
            ax.grid(True, alpha=0.3)
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Amplitude")

            # Set reasonable y-axis limits
            y_min, y_max = np.percentile(lead_data.samples, [1, 99])
            y_range = y_max - y_min
            ax.set_ylim(y_min - 0.1 * y_range, y_max + 0.1 * y_range)

            # Set x-axis to show about 3-4 seconds
            ax.set_xlim(0, min(4, time_axis[-1]))
        else:
            ax.text(
                0.5,
                0.5,
                f"Lead {lead_label}\nNot Available",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_title(f"Lead {lead_label}")

    plt.tight_layout()
    return fig


def main():
    """Main function to demonstrate ECG plotting"""

    # Load metadata
    carto_metadata = pd.read_csv(
        "carto_ecg_metadata.csv", parse_dates=["acquisition_date"]
    )

    # Find examples of sinus and non-sinus rhythm
    sinus_examples = carto_metadata[carto_metadata["sinus_rhythm"] == 1].head(1)
    non_sinus_examples = carto_metadata[carto_metadata["sinus_rhythm"] == 0].head(1)

    print("Loading ECG examples...")

    # Plot sinus rhythm example
    if not sinus_examples.empty:
        sinus_file = sinus_examples.iloc[0]["filename"]
        sinus_path = f"carto/{sinus_file}"

        if Path(sinus_path).exists():
            print(f"Loading sinus rhythm example: {sinus_file}")
            try:
                sinus_ecg = se.read_file(sinus_path)
                sinus_title = f"Sinus Rhythm Example\n{sinus_file}\n{sinus_examples.iloc[0]['interpretation'][:100]}..."

                fig1 = plot_12_lead_ecg(sinus_ecg, title=sinus_title)
                plt.savefig("sinus_rhythm_ecg.png", dpi=300, bbox_inches="tight")
                plt.show()
                print("✓ Sinus rhythm ECG saved as 'sinus_rhythm_ecg.png'")

            except Exception as e:
                print(f"Error loading sinus rhythm example: {e}")
        else:
            print(f"File not found: {sinus_path}")

    # Plot non-sinus rhythm example
    if not non_sinus_examples.empty:
        non_sinus_file = non_sinus_examples.iloc[0]["filename"]
        non_sinus_path = f"carto/{non_sinus_file}"

        if Path(non_sinus_path).exists():
            print(f"Loading non-sinus rhythm example: {non_sinus_file}")
            try:
                non_sinus_ecg = se.read_file(non_sinus_path)
                non_sinus_title = f"Non-Sinus Rhythm Example\n{non_sinus_file}\n{non_sinus_examples.iloc[0]['interpretation'][:100]}..."

                fig2 = plot_12_lead_ecg(non_sinus_ecg, title=non_sinus_title)
                plt.savefig("non_sinus_rhythm_ecg.png", dpi=300, bbox_inches="tight")
                plt.show()
                print("✓ Non-sinus rhythm ECG saved as 'non_sinus_rhythm_ecg.png'")

            except Exception as e:
                print(f"Error loading non-sinus rhythm example: {e}")
        else:
            print(f"File not found: {non_sinus_path}")

    # Also try to load a specific atrial flutter example
    atrial_flutter_examples = carto_metadata[
        carto_metadata["interpretation"].str.contains(
            "atrial flutter", case=False, na=False
        )
    ]

    if not atrial_flutter_examples.empty:
        af_file = atrial_flutter_examples.iloc[0]["filename"]
        af_path = f"carto/{af_file}"

        if Path(af_path).exists():
            print(f"Loading atrial flutter example: {af_file}")
            try:
                af_ecg = se.read_file(af_path)
                af_title = f"Atrial Flutter Example\n{af_file}\n{atrial_flutter_examples.iloc[0]['interpretation'][:100]}..."

                fig3 = plot_12_lead_ecg(af_ecg, title=af_title)
                plt.savefig("atrial_flutter_ecg.png", dpi=300, bbox_inches="tight")
                plt.show()
                print("✓ Atrial flutter ECG saved as 'atrial_flutter_ecg.png'")

            except Exception as e:
                print(f"Error loading atrial flutter example: {e}")
        else:
            print(f"File not found: {af_path}")


if __name__ == "__main__":
    main()
