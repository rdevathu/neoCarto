#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Class balance analysis script for AF recurrence prediction.
Analyzes positive rates across all window/outcome combinations.
"""

import sys
import logging
from pathlib import Path
import pandas as pd
import numpy as np
from rich.console import Console
from rich.table import Table

# Add ml_pipeline to path
sys.path.append(str(Path(__file__).parent.parent / "ml_pipeline"))

from data_loading import load_ecg_data
from preprocess import (
    filter_sinus_rhythm,
    create_pre_ecg_labels,
    create_outcome_labels,
    get_cohort_summary,
)

console = Console()
logger = logging.getLogger(__name__)


def analyze_class_balance(metadata_df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyze class balance across all window/outcome combinations.

    Args:
        metadata_df: Processed metadata DataFrame

    Returns:
        DataFrame with detailed balance analysis
    """
    # Get all window and outcome columns
    pre_ecg_windows = [col for col in metadata_df.columns if col.startswith("pre_ecg_")]
    outcome_labels = [
        col
        for col in metadata_df.columns
        if col.endswith(("_1y", "_3y", "_5y", "_any"))
    ]

    # Get cohort summary
    summary_df = get_cohort_summary(metadata_df, pre_ecg_windows, outcome_labels)

    # Add additional analysis
    summary_df["imbalance_ratio"] = summary_df["n_ecgs"] / summary_df["n_positive"]
    summary_df["minority_class_size"] = summary_df["n_positive"]
    summary_df["majority_class_size"] = summary_df["n_ecgs"] - summary_df["n_positive"]

    # Categorize imbalance severity
    def categorize_imbalance(ratio):
        if ratio <= 2:
            return "Balanced"
        elif ratio <= 5:
            return "Mild imbalance"
        elif ratio <= 10:
            return "Moderate imbalance"
        elif ratio <= 20:
            return "Severe imbalance"
        else:
            return "Extreme imbalance"

    summary_df["imbalance_category"] = summary_df["imbalance_ratio"].apply(
        categorize_imbalance
    )

    # Sort by imbalance ratio
    summary_df = summary_df.sort_values("imbalance_ratio")

    return summary_df


def print_balance_summary(balance_df: pd.DataFrame):
    """Print a summary table of class balance."""
    table = Table(title="Class Balance Analysis")
    table.add_column("Window", style="cyan")
    table.add_column("Outcome", style="yellow")
    table.add_column("Total ECGs", justify="right")
    table.add_column("Positive", justify="right")
    table.add_column("Positive %", justify="right")
    table.add_column("Imbalance Ratio", justify="right")
    table.add_column("Category", style="red")

    for _, row in balance_df.iterrows():
        table.add_row(
            row["window"],
            row["outcome"],
            f"{row['n_ecgs']:,}",
            f"{row['n_positive']:,}",
            f"{row['positive_rate']:.1%}",
            f"{row['imbalance_ratio']:.1f}:1",
            row["imbalance_category"],
        )

    console.print(table)


def print_balance_statistics(balance_df: pd.DataFrame):
    """Print overall statistics about class balance."""
    console.print("\n[bold]Class Balance Statistics:[/bold]")

    # Overall statistics
    total_combinations = len(balance_df)
    avg_positive_rate = balance_df["positive_rate"].mean()
    median_imbalance = balance_df["imbalance_ratio"].median()

    console.print(f"Total window/outcome combinations: {total_combinations}")
    console.print(f"Average positive rate: {avg_positive_rate:.1%}")
    console.print(f"Median imbalance ratio: {median_imbalance:.1f}:1")

    # Imbalance category distribution
    console.print("\n[bold]Imbalance Category Distribution:[/bold]")
    category_counts = balance_df["imbalance_category"].value_counts()
    for category, count in category_counts.items():
        pct = count / total_combinations * 100
        console.print(f"  {category}: {count} ({pct:.1f}%)")

    # Identify most problematic combinations
    console.print("\n[bold]Most Imbalanced Combinations:[/bold]")
    worst_5 = balance_df.nlargest(5, "imbalance_ratio")
    for _, row in worst_5.iterrows():
        console.print(
            f"  {row['window']} → {row['outcome']}: {row['imbalance_ratio']:.1f}:1 ({row['n_positive']} positive)"
        )

    # Best balanced combinations
    console.print("\n[bold]Best Balanced Combinations:[/bold]")
    best_5 = balance_df.nsmallest(5, "imbalance_ratio")
    for _, row in best_5.iterrows():
        console.print(
            f"  {row['window']} → {row['outcome']}: {row['imbalance_ratio']:.1f}:1 ({row['n_positive']} positive)"
        )


def main():
    """Main function."""
    logging.basicConfig(level=logging.INFO)

    console.print(
        "[bold green]Class Balance Analysis for AF Recurrence Prediction[/bold green]"
    )

    try:
        # Load data
        console.print("[bold]Loading data...[/bold]")
        metadata, waveforms = load_ecg_data()

        # Preprocess
        console.print("[bold]Preprocessing data...[/bold]")
        sinus_df = filter_sinus_rhythm(metadata)
        processed_df = create_pre_ecg_labels(sinus_df)
        processed_df = create_outcome_labels(processed_df)

        # Analyze class balance
        console.print("[bold]Analyzing class balance...[/bold]")
        balance_df = analyze_class_balance(processed_df)

        # Print results
        print_balance_summary(balance_df)
        print_balance_statistics(balance_df)

        # Save results
        output_path = Path("class_balance_analysis.csv")
        balance_df.to_csv(output_path, index=False)
        console.print(f"\n[bold]Results saved to {output_path}[/bold]")

    except Exception as e:
        console.print(f"[red]Error: {str(e)}[/red]")
        logger.exception("Analysis failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
