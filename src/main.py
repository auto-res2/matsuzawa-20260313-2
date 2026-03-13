"""
Main orchestrator for inference-only experiments.
Invokes src.inference as a subprocess with mode-specific overrides.
"""

import os
import sys
import subprocess
from pathlib import Path
import hydra
from omegaconf import DictConfig, OmegaConf


@hydra.main(config_path="../config", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    """
    Orchestrate a single run_id inference task.
    Apply mode-specific overrides before invoking inference.py.
    """
    # Resolve all variables
    OmegaConf.resolve(cfg)

    # Validate required fields
    if not cfg.get("run"):
        raise ValueError("run configuration is required. Use run=<run_id> in CLI")

    # Apply mode-specific overrides for sanity mode
    if cfg.mode == "sanity":
        # Sanity mode: minimal samples, online wandb
        cfg.inference.samples = cfg.run.inference.samples_sanity
        cfg.wandb.mode = "online"
        # Use separate wandb namespace for sanity
        cfg.wandb.project = f"{cfg.wandb.project}-sanity"

    elif cfg.mode == "pilot":
        # Pilot mode: 50-100 samples, online wandb
        cfg.inference.samples = cfg.run.inference.samples_pilot
        cfg.wandb.mode = "online"
        # Use separate wandb namespace for pilot
        cfg.wandb.project = f"{cfg.wandb.project}-pilot"

    elif cfg.mode == "full":
        # Full mode: all samples
        cfg.inference.samples = cfg.run.inference.samples_full
        cfg.wandb.mode = "online"
    else:
        raise ValueError(f"Unknown mode: {cfg.mode}. Must be sanity, pilot, or full")

    # Create results directory
    results_dir = Path(cfg.results_dir) / cfg.run.run_id
    results_dir.mkdir(parents=True, exist_ok=True)

    # Print configuration summary
    print(f"=== Experiment Configuration ===")
    print(f"Run ID: {cfg.run.run_id}")
    print(f"Method: {cfg.run.method.name} ({cfg.run.method.type})")
    print(f"Model: {cfg.run.model.name}")
    print(f"Dataset: {cfg.run.dataset.name}")
    print(f"Mode: {cfg.mode}")
    print(f"Samples: {cfg.inference.samples}")
    print(f"WandB: {cfg.wandb.entity}/{cfg.wandb.project}")
    print(f"Results: {results_dir}")
    print(f"================================\n")

    # Serialize config to YAML for subprocess
    config_yaml = OmegaConf.to_yaml(cfg)
    config_file = results_dir / "config.yaml"
    config_file.write_text(config_yaml)

    # Invoke inference.py as subprocess
    print(f"Launching inference subprocess...")
    cmd = [sys.executable, "-u", "-m", "src.inference", f"--config={config_file}"]

    result = subprocess.run(cmd, cwd=Path.cwd())

    if result.returncode != 0:
        print(f"ERROR: Inference subprocess failed with code {result.returncode}")
        sys.exit(result.returncode)

    print(f"\nExperiment completed successfully!")
    print(f"Results saved to: {results_dir}")


if __name__ == "__main__":
    main()
