"""
Evaluation script for comparing multiple runs.
Independent script that fetches results from WandB and generates comparison plots.
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Any
import matplotlib

matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import wandb


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate and compare multiple runs")
    parser.add_argument("--results_dir", required=True, help="Results directory")
    parser.add_argument("--run_ids", required=True, help="JSON string list of run IDs")
    parser.add_argument(
        "--entity", default=None, help="WandB entity (default: from env WANDB_ENTITY)"
    )
    parser.add_argument(
        "--project",
        default=None,
        help="WandB project (default: from env WANDB_PROJECT)",
    )
    return parser.parse_args()


def fetch_run_data(entity: str, project: str, run_id: str) -> Dict[str, Any]:
    """
    Fetch run data from WandB API.

    Args:
        entity: WandB entity
        project: WandB project
        run_id: Run display name

    Returns:
        Dictionary with run config, summary, and history
    """
    api = wandb.Api()

    # Find run by display name (most recent)
    runs = api.runs(
        f"{entity}/{project}", filters={"display_name": run_id}, order="-created_at"
    )

    if not runs:
        raise ValueError(
            f"No run found with display_name={run_id} in {entity}/{project}"
        )

    run = runs[0]  # Most recent run with that name

    # Extract data
    run_data = {
        "run_id": run_id,
        "config": run.config,
        "summary": dict(run.summary),
        "history": run.history().to_dict(orient="list")
        if hasattr(run, "history")
        else {},
    }

    return run_data


def export_per_run_metrics(
    results_dir: Path, run_id: str, run_data: Dict[str, Any]
) -> None:
    """
    Export per-run metrics to JSON and create per-run figures.

    Args:
        results_dir: Results directory
        run_id: Run ID
        run_data: Run data from WandB
    """
    run_dir = results_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # Export metrics
    metrics_file = run_dir / "metrics.json"
    with open(metrics_file, "w") as f:
        json.dump(run_data["summary"], f, indent=2)
    print(f"Exported metrics: {metrics_file}")

    # Create per-run figure: accuracy over progress
    if "progress" in run_data["history"] and "accuracy" in run_data["history"]:
        fig, ax = plt.subplots(figsize=(8, 6))
        progress = run_data["history"]["progress"]
        accuracy = run_data["history"]["accuracy"]

        ax.plot(progress, accuracy, marker="o", linewidth=2)
        ax.set_xlabel("Progress (samples)", fontsize=12)
        ax.set_ylabel("Accuracy", fontsize=12)
        ax.set_title(f"{run_id} - Accuracy over Progress", fontsize=14)
        ax.grid(True, alpha=0.3)

        fig_file = run_dir / f"{run_id}_accuracy.pdf"
        plt.savefig(fig_file, bbox_inches="tight", dpi=300)
        plt.close(fig)
        print(f"Generated figure: {fig_file}")


def generate_comparison_plots(
    results_dir: Path, all_run_data: List[Dict[str, Any]]
) -> None:
    """
    Generate comparison plots overlaying all runs.

    Args:
        results_dir: Results directory
        all_run_data: List of run data dictionaries
    """
    comparison_dir = results_dir / "comparison"
    comparison_dir.mkdir(parents=True, exist_ok=True)

    # Comparison plot: Accuracy over progress
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    markers = ["o", "s", "^", "D", "v"]

    for idx, run_data in enumerate(all_run_data):
        run_id = run_data["run_id"]
        history = run_data["history"]

        if "progress" in history and "accuracy" in history:
            progress = history["progress"]
            accuracy = history["accuracy"]

            color = colors[idx % len(colors)]
            marker = markers[idx % len(markers)]

            ax.plot(
                progress,
                accuracy,
                marker=marker,
                color=color,
                label=run_id,
                linewidth=2,
                markersize=6,
                alpha=0.8,
            )

    ax.set_xlabel("Progress (samples)", fontsize=12)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title("Accuracy Comparison Across Methods", fontsize=14)
    ax.legend(loc="best", fontsize=10)
    ax.grid(True, alpha=0.3)

    fig_file = comparison_dir / "comparison_accuracy.pdf"
    plt.savefig(fig_file, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Generated comparison figure: {fig_file}")

    # Bar chart: Final accuracy comparison
    fig, ax = plt.subplots(figsize=(8, 6))

    run_ids = [rd["run_id"] for rd in all_run_data]
    accuracies = [rd["summary"].get("accuracy", 0.0) for rd in all_run_data]

    bars = ax.bar(
        range(len(run_ids)), accuracies, color=colors[: len(run_ids)], alpha=0.7
    )
    ax.set_xticks(range(len(run_ids)))
    ax.set_xticklabels(run_ids, rotation=45, ha="right")
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title("Final Accuracy Comparison", fontsize=14)
    ax.grid(True, axis="y", alpha=0.3)

    # Add value labels on bars
    for i, (bar, acc) in enumerate(zip(bars, accuracies)):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{acc:.3f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    fig_file = comparison_dir / "comparison_final_accuracy.pdf"
    plt.savefig(fig_file, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Generated comparison figure: {fig_file}")


def export_aggregated_metrics(
    results_dir: Path, all_run_data: List[Dict[str, Any]]
) -> None:
    """
    Export aggregated metrics across all runs.

    Args:
        results_dir: Results directory
        all_run_data: List of run data dictionaries
    """
    comparison_dir = results_dir / "comparison"
    comparison_dir.mkdir(parents=True, exist_ok=True)

    # Extract metrics by run_id
    metrics_by_run = {}
    for run_data in all_run_data:
        run_id = run_data["run_id"]
        metrics_by_run[run_id] = run_data["summary"]

    # Identify proposed vs baseline runs
    proposed_runs = [rd for rd in all_run_data if "proposed" in rd["run_id"]]
    baseline_runs = [rd for rd in all_run_data if "comparative" in rd["run_id"]]

    # Get best from each
    best_proposed = None
    best_baseline = None

    if proposed_runs:
        best_proposed = max(
            proposed_runs, key=lambda x: x["summary"].get("accuracy", 0.0)
        )
    if baseline_runs:
        best_baseline = max(
            baseline_runs, key=lambda x: x["summary"].get("accuracy", 0.0)
        )

    # Calculate gap
    gap = None
    if best_proposed and best_baseline:
        gap = best_proposed["summary"].get("accuracy", 0.0) - best_baseline[
            "summary"
        ].get("accuracy", 0.0)

    # Create aggregated metrics
    aggregated = {
        "primary_metric": "accuracy",
        "metrics_by_run": metrics_by_run,
        "best_proposed": {
            "run_id": best_proposed["run_id"] if best_proposed else None,
            "accuracy": best_proposed["summary"].get("accuracy", 0.0)
            if best_proposed
            else None,
        },
        "best_baseline": {
            "run_id": best_baseline["run_id"] if best_baseline else None,
            "accuracy": best_baseline["summary"].get("accuracy", 0.0)
            if best_baseline
            else None,
        },
        "gap": gap,
    }

    # Export
    metrics_file = comparison_dir / "aggregated_metrics.json"
    with open(metrics_file, "w") as f:
        json.dump(aggregated, f, indent=2)
    print(f"Exported aggregated metrics: {metrics_file}")


def main():
    """Main entry point."""
    args = parse_args()

    # Parse run_ids
    run_ids = json.loads(args.run_ids)
    if not isinstance(run_ids, list):
        raise ValueError("run_ids must be a JSON list")

    # Get WandB credentials
    entity = args.entity or os.environ.get("WANDB_ENTITY")
    project = args.project or os.environ.get("WANDB_PROJECT")

    if not entity or not project:
        raise ValueError(
            "WandB entity and project must be provided via args or env vars"
        )

    print(f"Fetching data from WandB: {entity}/{project}")
    print(f"Run IDs: {run_ids}")

    # Fetch data for all runs
    all_run_data = []
    for run_id in run_ids:
        print(f"\nFetching data for: {run_id}")
        try:
            run_data = fetch_run_data(entity, project, run_id)
            all_run_data.append(run_data)
            print(f"  Accuracy: {run_data['summary'].get('accuracy', 'N/A')}")
        except Exception as e:
            print(f"  Error: {e}")
            continue

    if not all_run_data:
        print("ERROR: No run data fetched")
        sys.exit(1)

    # Create results directory
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    # Export per-run metrics and figures
    print("\n=== Exporting Per-Run Metrics ===")
    for run_data in all_run_data:
        export_per_run_metrics(results_dir, run_data["run_id"], run_data)

    # Generate comparison plots
    print("\n=== Generating Comparison Plots ===")
    generate_comparison_plots(results_dir, all_run_data)

    # Export aggregated metrics
    print("\n=== Exporting Aggregated Metrics ===")
    export_aggregated_metrics(results_dir, all_run_data)

    print("\n=== Evaluation Complete ===")
    print(f"All results saved to: {results_dir}")


if __name__ == "__main__":
    main()
