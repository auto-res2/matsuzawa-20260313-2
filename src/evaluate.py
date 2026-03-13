"""Evaluation script for comparing experimental runs."""

import os
import json
import argparse
from pathlib import Path
from typing import Dict, List, Any
import wandb
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend


def parse_args():
    """Parse command line arguments."""
    # [VALIDATOR FIX - Attempt 1]
    # [PROBLEM]: evaluate.py called with arguments but argparse expects --flag format
    # [CAUSE]: Workflow passes args as "results_dir=X run_ids=Y" but argparse needs "--results_dir X --run_ids Y"
    # [FIX]: Support both Hydra-style (key=value) and argparse-style (--key value) arguments
    #
    # [OLD CODE]:
    # parser = argparse.ArgumentParser(
    #     description="Evaluate and compare experimental runs"
    # )
    # parser.add_argument(
    #     "--results_dir", type=str, required=True, help="Results directory"
    # )
    # parser.add_argument(
    #     "--run_ids", type=str, required=True, help="JSON list of run IDs"
    # )
    # return parser.parse_args()
    #
    # [NEW CODE]:
    import sys

    # Check if arguments are in Hydra-style format (key=value)
    hydra_style = any("=" in arg and not arg.startswith("--") for arg in sys.argv[1:])

    if hydra_style:
        # Parse Hydra-style arguments
        args_dict = {}
        for arg in sys.argv[1:]:
            if "=" in arg:
                key, value = arg.split("=", 1)
                args_dict[key] = value

        # Validate required arguments
        if "results_dir" not in args_dict:
            raise ValueError("Missing required argument: results_dir")
        if "run_ids" not in args_dict:
            raise ValueError("Missing required argument: run_ids")

        # Create namespace object
        class Args:
            def __init__(self, results_dir: str, run_ids: str):
                self.results_dir = results_dir
                self.run_ids = run_ids

        return Args(results_dir=args_dict["results_dir"], run_ids=args_dict["run_ids"])
    else:
        # Parse argparse-style arguments
        parser = argparse.ArgumentParser(
            description="Evaluate and compare experimental runs"
        )
        parser.add_argument(
            "--results_dir", type=str, required=True, help="Results directory"
        )
        parser.add_argument(
            "--run_ids", type=str, required=True, help="JSON list of run IDs"
        )
        return parser.parse_args()


def fetch_wandb_run(entity: str, project: str, run_id: str) -> Dict[str, Any]:
    """
    Fetch run data from WandB API.

    Args:
        entity: WandB entity
        project: WandB project
        run_id: Run display name

    Returns:
        Dictionary with run data
    """
    # [VALIDATOR FIX - Attempt 1]
    # [PROBLEM]: Script fails when W&B project doesn't exist yet
    # [CAUSE]: fetch_wandb_run doesn't handle non-existent projects gracefully
    # [FIX]: Check if project exists first, raise specific error if not
    #
    # [OLD CODE]:
    # api = wandb.Api()
    # runs = api.runs(
    #     f"{entity}/{project}", filters={"display_name": run_id}, order="-created_at"
    # )
    #
    # [NEW CODE]:
    api = wandb.Api()

    try:
        # Fetch runs by display name
        runs = api.runs(
            f"{entity}/{project}", filters={"display_name": run_id}, order="-created_at"
        )
    except wandb.errors.CommError as e:
        if "Could not find project" in str(e) or "404" in str(e):
            raise ValueError(
                f"W&B project {entity}/{project} does not exist yet. Please run the experiments first."
            )
        raise

    if not runs:
        raise ValueError(
            f"No run found with display_name={run_id} in {entity}/{project}"
        )

    # Get most recent run with this name
    run = runs[0]

    # Extract data
    history = []
    for row in run.scan_history():
        history.append(row)

    data = {
        "run_id": run_id,
        "wandb_id": run.id,
        "summary": dict(run.summary),
        "config": dict(run.config),
        "history": history,
    }

    return data


def export_per_run_metrics(results_dir: Path, run_id: str, run_data: Dict[str, Any]):
    """
    Export per-run metrics and figures.

    Args:
        results_dir: Base results directory
        run_id: Run identifier
        run_data: Run data from WandB
    """
    run_dir = results_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # Export metrics
    metrics = run_data["summary"]
    with open(run_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Exported metrics for {run_id} to {run_dir / 'metrics.json'}")

    # Generate per-run figures
    history = run_data["history"]

    if history:
        # Accuracy over time
        if any("accuracy" in row for row in history):
            steps = [
                row.get("step", i) for i, row in enumerate(history) if "accuracy" in row
            ]
            accuracy = [row["accuracy"] for row in history if "accuracy" in row]

            plt.figure(figsize=(8, 6))
            plt.plot(steps, accuracy, marker="o", linestyle="-", linewidth=2)
            plt.xlabel("Step")
            plt.ylabel("Accuracy")
            plt.title(f"Accuracy over Time - {run_id}")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(
                run_dir / "accuracy_over_time.pdf", format="pdf", bbox_inches="tight"
            )
            plt.close()
            print(f"  Generated {run_dir / 'accuracy_over_time.pdf'}")

        # Compactness metrics
        if any("num_words" in row for row in history):
            steps = [
                row.get("step", i)
                for i, row in enumerate(history)
                if "num_words" in row
            ]
            num_words = [row["num_words"] for row in history if "num_words" in row]

            plt.figure(figsize=(8, 6))
            plt.plot(
                steps, num_words, marker="o", linestyle="-", linewidth=2, color="green"
            )
            plt.xlabel("Step")
            plt.ylabel("Number of Words")
            plt.title(f"Response Compactness - {run_id}")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(run_dir / "compactness.pdf", format="pdf", bbox_inches="tight")
            plt.close()
            print(f"  Generated {run_dir / 'compactness.pdf'}")


def generate_comparison_figures(results_dir: Path, all_run_data: List[Dict[str, Any]]):
    """
    Generate comparison figures across all runs.

    Args:
        results_dir: Base results directory
        all_run_data: List of run data dictionaries
    """
    comparison_dir = results_dir / "comparison"
    comparison_dir.mkdir(parents=True, exist_ok=True)

    # Collect common metrics
    common_metrics = set()
    for run_data in all_run_data:
        if run_data["history"]:
            common_metrics.update(run_data["history"][0].keys())

    # Filter to numeric metrics we care about
    metrics_to_plot = ["accuracy", "num_words", "num_lines", "is_robust"]
    metrics_to_plot = [m for m in metrics_to_plot if m in common_metrics]

    colors = ["blue", "red", "green", "orange", "purple"]
    linestyles = ["-", "--", "-.", ":"]

    for metric in metrics_to_plot:
        plt.figure(figsize=(10, 6))

        for idx, run_data in enumerate(all_run_data):
            run_id = run_data["run_id"]
            history = run_data["history"]

            # Extract metric values
            steps = [
                row.get("step", i) for i, row in enumerate(history) if metric in row
            ]
            values = [row[metric] for row in history if metric in row]

            if values:
                color = colors[idx % len(colors)]
                linestyle = linestyles[idx % len(linestyles)]
                plt.plot(
                    steps,
                    values,
                    marker="o",
                    linestyle=linestyle,
                    linewidth=2,
                    color=color,
                    label=run_id,
                    markersize=4,
                )

        plt.xlabel("Step")
        plt.ylabel(metric.replace("_", " ").title())
        plt.title(f"Comparison: {metric.replace('_', ' ').title()}")
        plt.legend(loc="best")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        filename = f"comparison_{metric}.pdf"
        plt.savefig(comparison_dir / filename, format="pdf", bbox_inches="tight")
        plt.close()
        print(f"Generated {comparison_dir / filename}")

    # Final summary bar chart
    plt.figure(figsize=(10, 6))

    run_ids = [d["run_id"] for d in all_run_data]
    accuracies = [d["summary"].get("accuracy", 0.0) for d in all_run_data]

    x = np.arange(len(run_ids))
    plt.bar(
        x, accuracies, color=["blue" if "proposed" in rid else "red" for rid in run_ids]
    )
    plt.xlabel("Run ID")
    plt.ylabel("Accuracy")
    plt.title("Final Accuracy Comparison")
    plt.xticks(x, run_ids, rotation=45, ha="right")
    plt.ylim(0, 1.0)
    plt.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(
        comparison_dir / "final_accuracy_comparison.pdf",
        format="pdf",
        bbox_inches="tight",
    )
    plt.close()
    print(f"Generated {comparison_dir / 'final_accuracy_comparison.pdf'}")


def compute_aggregated_metrics(all_run_data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Compute aggregated metrics across runs.

    Args:
        all_run_data: List of run data dictionaries

    Returns:
        Aggregated metrics dictionary
    """
    # Extract final metrics
    metrics_by_run = {}
    for run_data in all_run_data:
        run_id = run_data["run_id"]
        summary = run_data["summary"]
        metrics_by_run[run_id] = {
            "accuracy": summary.get("accuracy", 0.0),
            "avg_num_words": summary.get("avg_num_words", 0.0),
            "avg_num_lines": summary.get("avg_num_lines", 0.0),
            "robustness_rate": summary.get("robustness_rate", 0.0),
        }

    # Identify proposed and baseline runs
    proposed_runs = [rid for rid in metrics_by_run.keys() if "proposed" in rid]
    baseline_runs = [
        rid
        for rid in metrics_by_run.keys()
        if "comparative" in rid or "baseline" in rid
    ]

    # Compute best proposed and baseline
    best_proposed = None
    best_baseline = None

    if proposed_runs:
        best_proposed = max(
            proposed_runs, key=lambda rid: metrics_by_run[rid]["accuracy"]
        )

    if baseline_runs:
        best_baseline = max(
            baseline_runs, key=lambda rid: metrics_by_run[rid]["accuracy"]
        )

    # Compute gap
    gap = None
    if best_proposed and best_baseline:
        gap = (
            metrics_by_run[best_proposed]["accuracy"]
            - metrics_by_run[best_baseline]["accuracy"]
        )

    aggregated = {
        "primary_metric": "accuracy",
        "metrics_by_run": metrics_by_run,
        "best_proposed": best_proposed,
        "best_baseline": best_baseline,
        "gap": gap,
    }

    return aggregated


def generate_placeholder_visualizations(
    results_dir: Path, run_ids: List[str], error_msg: str
):
    """
    Generate placeholder visualizations when no run data is available.

    Args:
        results_dir: Base results directory
        run_ids: List of expected run IDs
        error_msg: Error message to display
    """
    # [VALIDATOR FIX - Attempt 1]
    # [PROBLEM]: No visualizations generated when W&B project doesn't exist
    # [CAUSE]: Script exits early when no run data is available
    # [FIX]: Generate informative placeholder visualizations that indicate runs need to be executed
    #
    # [NEW CODE]:
    comparison_dir = results_dir / "comparison"
    comparison_dir.mkdir(parents=True, exist_ok=True)

    # Create a placeholder figure with clear message
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.text(
        0.5,
        0.5,
        f"No Run Data Available\n\n{error_msg}\n\nExpected runs: {', '.join(run_ids)}\n\nPlease execute the experiments first before running visualization.",
        ha="center",
        va="center",
        fontsize=14,
        wrap=True,
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(
        comparison_dir / "placeholder_no_data.pdf", format="pdf", bbox_inches="tight"
    )
    plt.close()
    print(
        f"Generated placeholder visualization: {comparison_dir / 'placeholder_no_data.pdf'}"
    )

    # Create empty metrics file
    with open(comparison_dir / "aggregated_metrics.json", "w") as f:
        json.dump(
            {"error": error_msg, "expected_runs": run_ids, "status": "no_data"},
            f,
            indent=2,
        )
    print(
        f"Generated placeholder metrics: {comparison_dir / 'aggregated_metrics.json'}"
    )


def main():
    """Main evaluation function."""
    args = parse_args()

    # Parse run IDs
    run_ids = json.loads(args.run_ids)
    print(f"Evaluating runs: {run_ids}")

    results_dir = Path(args.results_dir)

    # Get WandB credentials from environment or default
    entity = os.getenv("WANDB_ENTITY", "airas")
    project = os.getenv("WANDB_PROJECT", "2026-0313-matsuzawa-2")

    print(f"Fetching data from WandB: {entity}/{project}")

    # Fetch data for each run
    all_run_data = []
    project_not_found = False
    error_message = ""

    for run_id in run_ids:
        print(f"\nFetching run: {run_id}")
        try:
            run_data = fetch_wandb_run(entity, project, run_id)
            all_run_data.append(run_data)

            # Export per-run metrics
            export_per_run_metrics(results_dir, run_id, run_data)
        except ValueError as e:
            error_str = str(e)
            if (
                "does not exist yet" in error_str
                or "Could not find project" in error_str
            ):
                project_not_found = True
                error_message = error_str
            print(f"Error fetching run {run_id}: {e}")
            continue
        except Exception as e:
            print(f"Error fetching run {run_id}: {e}")
            continue

    if not all_run_data:
        if project_not_found:
            print(f"\nW&B project not found. Generating placeholder visualizations.")
            generate_placeholder_visualizations(results_dir, run_ids, error_message)
        else:
            print("No run data available. Exiting.")
        return

    # Generate comparison figures
    print("\nGenerating comparison figures...")
    generate_comparison_figures(results_dir, all_run_data)

    # Compute aggregated metrics
    print("\nComputing aggregated metrics...")
    aggregated = compute_aggregated_metrics(all_run_data)

    # Export aggregated metrics
    comparison_dir = results_dir / "comparison"
    comparison_dir.mkdir(parents=True, exist_ok=True)

    with open(comparison_dir / "aggregated_metrics.json", "w") as f:
        json.dump(aggregated, f, indent=2)

    print(
        f"\nExported aggregated metrics to {comparison_dir / 'aggregated_metrics.json'}"
    )

    # Print summary
    print("\n" + "=" * 80)
    print("EVALUATION SUMMARY")
    print("=" * 80)
    print(f"Primary Metric: {aggregated['primary_metric']}")
    print(f"Best Proposed: {aggregated.get('best_proposed', 'N/A')}")
    print(f"Best Baseline: {aggregated.get('best_baseline', 'N/A')}")
    if aggregated.get("gap") is not None:
        print(f"Gap: {aggregated['gap']:.4f}")
    print("=" * 80)


if __name__ == "__main__":
    main()
