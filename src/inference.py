"""Inference script for prompt tuning experiments."""

import os
import json
import time
from pathlib import Path
from typing import Dict, List, Any
import wandb
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from src.preprocess import (
    load_gsm8k,
    format_prompt,
    extract_model_answer,
    normalize_answer,
    compute_compactness,
    check_robustness,
)


def get_llm_client(cfg: DictConfig):
    """Get LLM client based on provider."""
    provider = cfg.model.provider.lower()

    if provider == "openai":
        import openai

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        return openai.OpenAI(api_key=api_key)
    elif provider == "anthropic":
        import anthropic

        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable not set")
        return anthropic.Anthropic(api_key=api_key)
    else:
        raise ValueError(f"Unsupported provider: {provider}")


def generate_response(client, cfg: DictConfig, prompt: str) -> str:
    """Generate response from LLM."""
    provider = cfg.model.provider.lower()

    if provider == "openai":
        response = client.chat.completions.create(
            model=cfg.model.name,
            messages=[{"role": "user", "content": prompt}],
            temperature=cfg.model.temperature,
            max_tokens=cfg.model.max_tokens,
        )
        return response.choices[0].message.content
    elif provider == "anthropic":
        response = client.messages.create(
            model=cfg.model.name,
            messages=[{"role": "user", "content": prompt}],
            temperature=cfg.model.temperature,
            max_tokens=cfg.model.max_tokens,
        )
        return response.content[0].text
    else:
        raise ValueError(f"Unsupported provider: {provider}")


def run_inference(cfg: DictConfig) -> Dict[str, Any]:
    """
    Run inference experiment.

    Args:
        cfg: Hydra configuration

    Returns:
        Dictionary with results and metrics
    """
    # Initialize WandB if enabled
    wandb_enabled = cfg.wandb.mode == "online"
    if wandb_enabled:
        wandb.init(
            entity=cfg.wandb.entity,
            project=cfg.wandb.project,
            name=cfg.run.run_id,
            config=OmegaConf.to_container(cfg, resolve=True),
        )
        print(f"WandB run URL: {wandb.run.url}")

    # Load dataset
    max_samples = cfg.run.dataset.max_samples
    if cfg.mode == "sanity":
        max_samples = min(10, max_samples) if max_samples else 10
    elif cfg.mode == "pilot":
        max_samples = max(50, int(max_samples * 0.2)) if max_samples else 50

    print(
        f"Loading dataset: {cfg.run.dataset.name} (split={cfg.run.dataset.split}, max_samples={max_samples})"
    )
    examples = load_gsm8k(
        split=cfg.run.dataset.split,
        max_samples=max_samples,
        cache_dir=cfg.run.inference.cache_dir,
    )
    print(f"Loaded {len(examples)} examples")

    # Initialize LLM client
    client = get_llm_client(cfg)

    # Run inference
    results = []
    correct = 0
    total = 0

    all_compactness = []
    all_robustness = []

    prompt_template = cfg.run.method.prompt_template
    answer_pattern = (
        cfg.run.evaluation.answer_extraction.pattern
        if "answer_extraction" in cfg.run.evaluation
        else None
    )

    print(f"Running inference with method: {cfg.run.method.name}")
    for example in tqdm(examples, desc="Inference"):
        # Format prompt
        prompt = format_prompt(example["question"], prompt_template)

        # Generate response
        try:
            response = generate_response(client, cfg, prompt)

            # Extract answer
            predicted_answer = extract_model_answer(response, answer_pattern)
            gold_answer = normalize_answer(example["answer"])

            # Check correctness
            is_correct = predicted_answer == gold_answer
            if is_correct:
                correct += 1
            total += 1

            # Compute metrics
            compactness = compute_compactness(response)
            robustness = check_robustness(response)

            all_compactness.append(compactness)
            all_robustness.append(robustness)

            # Store result
            result = {
                "index": example["index"],
                "question": example["question"],
                "gold_answer": gold_answer,
                "predicted_answer": predicted_answer,
                "response": response,
                "correct": is_correct,
                "compactness": compactness,
                "robustness": robustness,
            }
            results.append(result)

            # Log to WandB
            if wandb_enabled:
                wandb.log(
                    {
                        "step": total,
                        "accuracy": correct / total,
                        "num_words": compactness["num_words"],
                        "num_lines": compactness["num_lines"],
                        "is_robust": robustness["is_robust"],
                    }
                )

        except Exception as e:
            print(f"Error on example {example['index']}: {e}")
            continue

    # Compute final metrics
    accuracy = correct / total if total > 0 else 0.0

    # Aggregate compactness metrics
    avg_compactness = {
        "avg_num_words": sum(c["num_words"] for c in all_compactness)
        / len(all_compactness)
        if all_compactness
        else 0,
        "avg_num_lines": sum(c["num_lines"] for c in all_compactness)
        / len(all_compactness)
        if all_compactness
        else 0,
    }

    # Aggregate robustness metrics
    robustness_rate = (
        sum(r["is_robust"] for r in all_robustness) / len(all_robustness)
        if all_robustness
        else 0
    )

    metrics = {
        "accuracy": accuracy,
        "total_samples": total,
        "correct_samples": correct,
        **avg_compactness,
        "robustness_rate": robustness_rate,
    }

    print(f"\nFinal Metrics:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Robustness Rate: {robustness_rate:.4f}")
    print(f"  Avg Words: {avg_compactness['avg_num_words']:.1f}")
    print(f"  Avg Lines: {avg_compactness['avg_num_lines']:.1f}")

    # Log final metrics to WandB
    if wandb_enabled:
        for key, value in metrics.items():
            wandb.summary[key] = value

    # Save results
    results_dir = Path(cfg.results_dir) / cfg.run.run_id
    results_dir.mkdir(parents=True, exist_ok=True)

    with open(results_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    with open(results_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nResults saved to {results_dir}")

    # Validation for sanity mode
    if cfg.mode == "sanity":
        validate_sanity(total, metrics)
    elif cfg.mode == "pilot":
        validate_pilot(total, metrics)

    if wandb_enabled:
        wandb.finish()

    return {"results": results, "metrics": metrics}


def validate_sanity(total_samples: int, metrics: Dict[str, Any]):
    """Validate sanity mode run."""
    summary = {
        "samples": total_samples,
        "accuracy": metrics.get("accuracy", 0.0),
        "avg_words": metrics.get("avg_num_words", 0),
        "robustness_rate": metrics.get("robustness_rate", 0.0),
    }

    print(f"SANITY_VALIDATION_SUMMARY: {json.dumps(summary)}")

    # Check conditions
    if total_samples < 5:
        print("SANITY_VALIDATION: FAIL reason=insufficient_samples")
        return

    if not all(
        isinstance(v, (int, float))
        and not (isinstance(v, float) and (v != v or abs(v) == float("inf")))
        for v in metrics.values()
    ):
        print("SANITY_VALIDATION: FAIL reason=invalid_metrics")
        return

    print("SANITY_VALIDATION: PASS")


def validate_pilot(total_samples: int, metrics: Dict[str, Any]):
    """Validate pilot mode run."""
    summary = {
        "samples": total_samples,
        "primary_metric": "accuracy",
        "primary_metric_value": metrics.get("accuracy", 0.0),
        "robustness_rate": metrics.get("robustness_rate", 0.0),
    }

    print(f"PILOT_VALIDATION_SUMMARY: {json.dumps(summary)}")

    # Check conditions
    if total_samples < 50:
        print("PILOT_VALIDATION: FAIL reason=insufficient_samples")
        return

    if metrics.get("accuracy", 0.0) == 0.0:
        print("PILOT_VALIDATION: FAIL reason=zero_accuracy")
        return

    if not all(
        isinstance(v, (int, float))
        and not (isinstance(v, float) and (v != v or abs(v) == float("inf")))
        for v in metrics.values()
    ):
        print("PILOT_VALIDATION: FAIL reason=invalid_metrics")
        return

    print("PILOT_VALIDATION: PASS")


if __name__ == "__main__":
    import hydra
    from hydra import compose, initialize

    @hydra.main(config_path="../config", config_name="config", version_base=None)
    def main(cfg: DictConfig):
        run_inference(cfg)

    main()
