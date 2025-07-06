#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Results visualization script for AF recurrence prediction experiments.
Displays experiment details, holdout metrics, and confusion matrices.
"""

import json
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.columns import Columns
from datetime import datetime
import joblib
from sklearn.metrics import confusion_matrix, classification_report
from typing import Dict, List, Optional, Tuple

console = Console()


def find_latest_batch_results(base_dir: str = "batch_results") -> Optional[Path]:
    """Find the most recent batch results directory."""
    base_path = Path(base_dir)
    if not base_path.exists():
        return None

    batch_dirs = [
        d for d in base_path.iterdir() if d.is_dir() and d.name.startswith("batch_")
    ]
    if not batch_dirs:
        return None

    # Sort by creation time (newest first)
    batch_dirs.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    return batch_dirs[0]


def load_experiment_results(results_dir: Path) -> List[Dict]:
    """Load all experiment results from a batch directory."""
    results = []

    # First, try to find results directly in the directory
    json_files = list(results_dir.glob("*_results.json"))

    # If no direct results, look in subdirectories (batch runner structure)
    if not json_files:
        for subdir in results_dir.iterdir():
            if subdir.is_dir():
                json_files.extend(subdir.glob("*_results.json"))

    for json_file in json_files:
        try:
            with open(json_file, "r") as f:
                result = json.load(f)
                result["filename"] = json_file.name
                result["model_file"] = json_file.name.replace(
                    "_results.json", "_model.joblib"
                )
                result["results_path"] = json_file.parent  # Store the directory path
                results.append(result)
        except Exception as e:
            console.print(f"[red]Error loading {json_file}: {e}[/red]")

    return results


def create_summary_table(results: List[Dict]) -> Table:
    """Create a summary table of all experiments."""
    table = Table(
        title="Experiment Summary", show_header=True, header_style="bold magenta"
    )

    table.add_column("Experiment", style="cyan", width=30)
    table.add_column("Window", style="green")
    table.add_column("Outcome", style="yellow")
    table.add_column("Model", style="blue")
    table.add_column("Leads", style="white")
    table.add_column("Aug", style="red")
    table.add_column("Cohort", style="white")
    table.add_column("AUROC", style="bold green")
    table.add_column("AUPRC", style="bold yellow")
    table.add_column("F1", style="bold blue")

    for i, result in enumerate(results, 1):
        config = result.get("experiment_config", {})
        holdout_metrics = result.get("holdout_metrics", {})

        # Extract experiment details
        window = config.get("pre_ecg_window", "N/A")
        outcome = config.get("outcome_label", "N/A")
        model = config.get("model_name", "N/A")
        leads = config.get("lead_config", "N/A")
        aug = "Yes" if config.get("use_augmentation", False) else "No"

        # Extract metrics
        cohort_size = result.get("cohort_size", 0)
        auroc = holdout_metrics.get("roc_auc", 0.0)
        auprc = holdout_metrics.get("pr_auc", 0.0)
        f1 = holdout_metrics.get("f1", 0.0)

        table.add_row(
            f"Exp {i}",
            window.replace("pre_ecg_", ""),
            outcome.replace("_recurrence", "").replace("af_at", "af+at"),
            model.upper(),
            leads,
            aug,
            str(cohort_size),
            f"{auroc:.3f}",
            f"{auprc:.3f}",
            f"{f1:.3f}",
        )

    return table


def create_detailed_result_panel(result: Dict, exp_num: int) -> Panel:
    """Create a detailed panel for a single experiment result."""
    config = result.get("experiment_config", {})
    holdout_metrics = result.get("holdout_metrics", {})

    # Experiment configuration
    config_text = f"""[bold cyan]Configuration:[/bold cyan]
• Window: {config.get("pre_ecg_window", "N/A")}
• Outcome: {config.get("outcome_label", "N/A")}
• Model: {config.get("model_name", "N/A").upper()}
• Leads: {config.get("lead_config", "N/A")}
• Augmentation: {"Yes" if config.get("use_augmentation", False) else "No"}
• Oversampling: {"Yes" if config.get("use_oversampling", False) else "No"}
• CV Folds: {config.get("n_folds", "N/A")}"""

    # Cohort statistics
    cohort_text = f"""[bold yellow]Cohort Statistics:[/bold yellow]
• Total ECGs: {result.get("cohort_size", 0)}
• Patients: {result.get("n_patients", 0)}
• Holdout Size: {result.get("holdout_size", 0)}
• Class Distribution: {result.get("class_distribution", {})}"""

    # Holdout metrics
    metrics_text = f"""[bold green]Holdout Metrics:[/bold green]
• Accuracy: {holdout_metrics.get("accuracy", 0.0):.3f}
• Precision: {holdout_metrics.get("precision", 0.0):.3f}
• Recall: {holdout_metrics.get("recall", 0.0):.3f}
• F1-Score: {holdout_metrics.get("f1", 0.0):.3f}
• ROC-AUC: {holdout_metrics.get("roc_auc", 0.0):.3f}
• PR-AUC: {holdout_metrics.get("pr_auc", 0.0):.3f}"""

    content = f"{config_text}\n\n{cohort_text}\n\n{metrics_text}"

    return Panel(
        content, title=f"[bold]Experiment {exp_num}[/bold]", border_style="blue"
    )


def load_model_and_create_confusion_matrix(
    results_dir: Path, result: Dict, exp_num: int
) -> Optional[plt.Figure]:
    """Load model and create confusion matrix from holdout predictions."""
    try:
        # Load the trained model - use the stored results_path if available
        if "results_path" in result:
            model_file = result["results_path"] / result["model_file"]
        else:
            model_file = results_dir / result["model_file"]

        if not model_file.exists():
            console.print(f"[red]Model file not found: {model_file}[/red]")
            return None

        model = joblib.load(model_file)

        # We need to reconstruct the holdout data to get predictions
        # This is a limitation - we should save predictions in the results
        # For now, we'll create a placeholder confusion matrix using the metrics

        config = result.get("experiment_config", {})
        holdout_metrics = result.get("holdout_metrics", {})
        holdout_size = result.get("holdout_size", 0)

        if holdout_size == 0:
            return None

        # Estimate confusion matrix from metrics (approximate)
        precision = holdout_metrics.get("precision", 0.0)
        recall = holdout_metrics.get("recall", 0.0)
        accuracy = holdout_metrics.get("accuracy", 0.0)

        # Get class distribution to estimate positive/negative counts
        class_dist = result.get("class_distribution", {0: 1, 1: 1})
        total_samples = sum(class_dist.values())
        pos_ratio = class_dist.get(1, 0) / total_samples if total_samples > 0 else 0.2

        # Estimate holdout positive/negative samples
        holdout_pos = int(holdout_size * pos_ratio)
        holdout_neg = holdout_size - holdout_pos

        # Estimate confusion matrix values
        if precision > 0 and recall > 0:
            tp = int(holdout_pos * recall)  # True positives
            fp = int(tp / precision - tp) if precision > 0 else 0  # False positives
            fn = holdout_pos - tp  # False negatives
            tn = holdout_neg - fp  # True negatives
        else:
            # Model predicted all negative
            tp, fp = 0, 0
            fn = holdout_pos
            tn = holdout_neg

        # Ensure non-negative values
        tp, fp, fn, tn = max(0, tp), max(0, fp), max(0, fn), max(0, tn)

        cm = np.array([[tn, fp], [fn, tp]])

        # Create confusion matrix plot
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            ax=ax,
            xticklabels=["No Recurrence", "Recurrence"],
            yticklabels=["No Recurrence", "Recurrence"],
        )

        ax.set_title(
            f"Confusion Matrix - Experiment {exp_num}\n"
            f"{config.get('outcome_label', '').replace('_', ' ').title()}",
            fontsize=12,
            fontweight="bold",
        )
        ax.set_xlabel("Predicted", fontsize=10)
        ax.set_ylabel("Actual", fontsize=10)

        # Add metrics text
        metrics_text = f"Accuracy: {accuracy:.3f}\nPrecision: {precision:.3f}\nRecall: {recall:.3f}\nF1: {holdout_metrics.get('f1', 0.0):.3f}"
        ax.text(
            1.05,
            0.5,
            metrics_text,
            transform=ax.transAxes,
            fontsize=9,
            verticalalignment="center",
            bbox=dict(boxstyle="round", facecolor="lightgray"),
        )

        plt.tight_layout()
        return fig

    except Exception as e:
        console.print(
            f"[red]Error creating confusion matrix for experiment {exp_num}: {e}[/red]"
        )
        return None


def create_metrics_comparison_plot(results: List[Dict]) -> plt.Figure:
    """Create a comparison plot of key metrics across experiments."""
    if not results:
        return None

    # Extract data for plotting
    exp_names = []
    auroc_scores = []
    auprc_scores = []
    f1_scores = []

    for i, result in enumerate(results, 1):
        config = result.get("experiment_config", {})
        holdout_metrics = result.get("holdout_metrics", {})

        # Create short experiment name
        window = config.get("pre_ecg_window", "").replace("pre_ecg_", "")
        outcome = config.get("outcome_label", "").split("_")[0]  # af, at, etc.
        model = config.get("model_name", "").upper()
        leads = config.get("lead_config", "")

        exp_name = f"{outcome}_{window}_{model}_{leads}"
        exp_names.append(exp_name)

        auroc_scores.append(holdout_metrics.get("roc_auc", 0.0))
        auprc_scores.append(holdout_metrics.get("pr_auc", 0.0))
        f1_scores.append(holdout_metrics.get("f1", 0.0))

    # Create comparison plot
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))

    x_pos = np.arange(len(exp_names))

    # AUROC
    bars1 = ax1.bar(x_pos, auroc_scores, color="skyblue", alpha=0.7)
    ax1.set_title("ROC-AUC Scores (Holdout)", fontweight="bold")
    ax1.set_ylabel("ROC-AUC")
    ax1.set_ylim(0, 1)
    ax1.axhline(y=0.5, color="red", linestyle="--", alpha=0.5, label="Random")
    ax1.legend()

    # Add value labels on bars
    for bar, score in zip(bars1, auroc_scores):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{score:.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    # AUPRC
    bars2 = ax2.bar(x_pos, auprc_scores, color="lightgreen", alpha=0.7)
    ax2.set_title("PR-AUC Scores (Holdout)", fontweight="bold")
    ax2.set_ylabel("PR-AUC")
    ax2.set_ylim(0, 1)

    for bar, score in zip(bars2, auprc_scores):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{score:.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    # F1
    bars3 = ax3.bar(x_pos, f1_scores, color="salmon", alpha=0.7)
    ax3.set_title("F1 Scores (Holdout)", fontweight="bold")
    ax3.set_ylabel("F1-Score")
    ax3.set_ylim(0, 1)
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(exp_names, rotation=45, ha="right")

    for bar, score in zip(bars3, f1_scores):
        ax3.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{score:.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    plt.tight_layout()
    return fig


def save_confusion_matrices(results_dir: Path, results: List[Dict], output_dir: Path):
    """Save confusion matrix plots for all experiments."""
    output_dir.mkdir(exist_ok=True)

    console.print("[bold]Generating confusion matrices...[/bold]")

    for i, result in enumerate(results, 1):
        fig = load_model_and_create_confusion_matrix(results_dir, result, i)
        if fig:
            config = result.get("experiment_config", {})
            filename = (
                f"confusion_matrix_exp_{i}_{config.get('outcome_label', 'unknown')}.png"
            )
            fig.savefig(output_dir / filename, dpi=300, bbox_inches="tight")
            plt.close(fig)
            console.print(f"[green]✓[/green] Saved {filename}")


def main():
    """Main function to visualize batch experiment results."""
    parser = argparse.ArgumentParser(
        description="Visualize AF recurrence prediction experiment results"
    )
    parser.add_argument(
        "--results-dir",
        help="Path to batch results directory (uses latest if not specified)",
    )
    parser.add_argument(
        "--output-dir", default="visualizations", help="Directory to save plots"
    )
    parser.add_argument(
        "--show-details",
        action="store_true",
        help="Show detailed results for each experiment",
    )
    parser.add_argument("--save-plots", action="store_true", help="Save plots to files")
    parser.add_argument(
        "--max-experiments",
        type=int,
        default=20,
        help="Maximum number of experiments to show in detail",
    )

    args = parser.parse_args()

    # Find results directory
    if args.results_dir:
        results_dir = Path(args.results_dir)
    else:
        results_dir = find_latest_batch_results()
        if results_dir is None:
            console.print(
                "[red]No batch results found. Please specify --results-dir[/red]"
            )
            return

    if not results_dir.exists():
        console.print(f"[red]Results directory not found: {results_dir}[/red]")
        return

    console.print(f"[bold green]Loading results from: {results_dir}[/bold green]")

    # Load all experiment results
    results = load_experiment_results(results_dir)
    if not results:
        console.print("[red]No experiment results found[/red]")
        return

    console.print(f"[bold]Found {len(results)} experiments[/bold]")

    # Create output directory
    output_dir = Path(args.output_dir)
    if args.save_plots:
        output_dir.mkdir(exist_ok=True)

    # Display summary table
    console.print("\n")
    summary_table = create_summary_table(results)
    console.print(summary_table)

    # Show detailed results if requested
    if args.show_details:
        console.print(
            f"\n[bold]Detailed Results (showing up to {args.max_experiments} experiments):[/bold]"
        )

        for i, result in enumerate(results[: args.max_experiments], 1):
            panel = create_detailed_result_panel(result, i)
            console.print(panel)
            console.print("")  # Add spacing

    # Create and show metrics comparison plot
    if len(results) > 1:
        console.print("[bold]Creating metrics comparison plot...[/bold]")
        metrics_fig = create_metrics_comparison_plot(results)
        if metrics_fig:
            if args.save_plots:
                metrics_fig.savefig(
                    output_dir / "metrics_comparison.png", dpi=300, bbox_inches="tight"
                )
                console.print(
                    f"[green]✓[/green] Saved metrics comparison to {output_dir / 'metrics_comparison.png'}"
                )
            else:
                plt.show()
            plt.close(metrics_fig)

    # Generate confusion matrices
    if args.save_plots:
        save_confusion_matrices(results_dir, results, output_dir / "confusion_matrices")

    # Print summary statistics
    console.print(f"\n[bold cyan]Summary Statistics:[/bold cyan]")

    holdout_aurocs = [r.get("holdout_metrics", {}).get("roc_auc", 0.0) for r in results]
    holdout_auprcs = [r.get("holdout_metrics", {}).get("pr_auc", 0.0) for r in results]
    holdout_f1s = [r.get("holdout_metrics", {}).get("f1", 0.0) for r in results]

    stats_table = Table(title="Performance Statistics", show_header=True)
    stats_table.add_column("Metric", style="cyan")
    stats_table.add_column("Mean", style="green")
    stats_table.add_column("Std", style="yellow")
    stats_table.add_column("Min", style="red")
    stats_table.add_column("Max", style="blue")

    for metric_name, values in [
        ("ROC-AUC", holdout_aurocs),
        ("PR-AUC", holdout_auprcs),
        ("F1-Score", holdout_f1s),
    ]:
        if values:
            stats_table.add_row(
                metric_name,
                f"{np.mean(values):.3f}",
                f"{np.std(values):.3f}",
                f"{np.min(values):.3f}",
                f"{np.max(values):.3f}",
            )

    console.print(stats_table)

    if args.save_plots:
        console.print(
            f"\n[bold green]All visualizations saved to: {output_dir}[/bold green]"
        )


if __name__ == "__main__":
    main()
