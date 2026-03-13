"""Main orchestration script for experiments."""

import sys
import subprocess
from pathlib import Path
import hydra
from omegaconf import DictConfig, OmegaConf


@hydra.main(config_path="../config", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    """
    Orchestrate experiment execution.

    This script determines the task type and invokes the appropriate execution script.
    For inference-only tasks, it calls src.inference directly.
    """
    print("=" * 80)
    print(f"Running experiment: {cfg.run.run_id}")
    print(f"Mode: {cfg.mode}")
    print("=" * 80)

    # Apply mode-specific overrides
    apply_mode_overrides(cfg)

    # Print effective config
    print("\nEffective Configuration:")
    print(OmegaConf.to_yaml(cfg))
    print("=" * 80)

    # Determine task type and execute
    # This is an inference-only task (prompt tuning)
    print(f"\nTask type: Inference (Prompt Tuning)")
    print(f"Invoking inference module...")

    # Import and run inference
    from src.inference import run_inference

    try:
        result = run_inference(cfg)
        print("\n" + "=" * 80)
        print("Experiment completed successfully!")
        print("=" * 80)
    except Exception as e:
        print(f"\nError during execution: {e}", file=sys.stderr)
        raise


def apply_mode_overrides(cfg: DictConfig) -> None:
    """
    Apply mode-specific configuration overrides.

    Args:
        cfg: Hydra configuration (modified in-place)
    """
    mode = cfg.mode

    if mode == "sanity":
        # Sanity mode: minimal execution for validation
        print("Applying sanity mode overrides...")

        # Override dataset samples
        if "dataset" in cfg.run and "max_samples" in cfg.run.dataset:
            original_samples = cfg.run.dataset.max_samples
            cfg.run.dataset.max_samples = min(10, original_samples)
            print(
                f"  Dataset samples: {original_samples} -> {cfg.run.dataset.max_samples}"
            )

        # Override WandB project for sanity runs
        if cfg.wandb.mode == "online":
            original_project = cfg.wandb.project
            cfg.wandb.project = f"{original_project}-sanity"
            print(f"  WandB project: {original_project} -> {cfg.wandb.project}")

    elif mode == "pilot":
        # Pilot mode: reduced scale for preliminary results
        print("Applying pilot mode overrides...")

        # Override dataset samples (20% or at least 50)
        if "dataset" in cfg.run and "max_samples" in cfg.run.dataset:
            original_samples = cfg.run.dataset.max_samples
            cfg.run.dataset.max_samples = max(50, int(original_samples * 0.2))
            print(
                f"  Dataset samples: {original_samples} -> {cfg.run.dataset.max_samples}"
            )

        # Override WandB project for pilot runs
        if cfg.wandb.mode == "online":
            original_project = cfg.wandb.project
            cfg.wandb.project = f"{original_project}-pilot"
            print(f"  WandB project: {original_project} -> {cfg.wandb.project}")

    elif mode == "full":
        # Full mode: no overrides needed
        print("Running in full mode (no overrides)")

    else:
        print(f"Warning: Unknown mode '{mode}', proceeding without overrides")


if __name__ == "__main__":
    main()
