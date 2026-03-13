"""
Inference script for AC-CoT vs Zero-shot-CoT comparison on GSM8K.
Single run executor invoked by main.py.
"""

import os
import sys
import json
import re
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional
import yaml
from datasets import load_dataset
import wandb


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def format_prompt(question: str, method_template: str, method_name: str) -> str:
    """
    Format the prompt based on the method.

    Args:
        question: The GSM8K question
        method_template: The prompt template from config
        method_name: Name of the method (ac-cot or zero-shot-cot)

    Returns:
        Formatted prompt string
    """
    if method_name == "ac-cot":
        # AC-CoT: Abstract-then-Compact Chain-of-Thought
        prompt = f"""Question: {question}

{method_template}

Answer:"""
    elif method_name == "zero-shot-cot":
        # Zero-shot-CoT: Standard baseline
        prompt = f"""Question: {question}

{method_template}

Answer:"""
    else:
        raise ValueError(f"Unknown method: {method_name}")

    return prompt


def call_model_api(prompt: str, model_config: Dict[str, Any]) -> str:
    """
    Call the model API with the given prompt.

    Args:
        prompt: The formatted prompt
        model_config: Model configuration from config

    Returns:
        Raw model response text
    """
    provider = model_config.get("provider", "together")
    model_name = model_config["name"]
    temperature = model_config.get("temperature", 0.0)
    max_tokens = model_config.get("max_tokens", 512)
    top_p = model_config.get("top_p", 1.0)

    if provider == "together":
        try:
            # [VALIDATOR FIX - Attempt 1]
            # [PROBLEM]: module 'together' has no attribute 'Complete'
            # [CAUSE]: Together API changed from together.Complete.create() to client-based API
            # [FIX]: Use Together client class with completions.create() method
            #
            # [OLD CODE]:
            # import together
            # together.api_key = os.environ.get("TOGETHER_API_KEY")
            # response = together.Complete.create(
            #     model=model_name,
            #     prompt=prompt,
            #     temperature=temperature,
            #     max_tokens=max_tokens,
            #     top_p=top_p,
            #     stop=["Question:", "\n\n\n"],
            # )
            # return response["output"]["choices"][0]["text"].strip()
            #
            # [NEW CODE]:
            from together import Together

            client = Together(api_key=os.environ.get("TOGETHER_API_KEY"))

            response = client.completions.create(
                model=model_name,
                prompt=prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                stop=["Question:", "\n\n\n"],
            )
            return response.choices[0].text.strip()
        except ImportError:
            raise ImportError(
                "together package required. Install with: pip install together"
            )
    else:
        raise ValueError(f"Unsupported provider: {provider}")


def extract_answer(response: str, pattern: str, method_name: str) -> Optional[str]:
    """
    Extract the numerical answer from model response.

    Args:
        response: Raw model output
        pattern: Regex pattern for answer extraction
        method_name: Method name for fallback logic

    Returns:
        Extracted answer string or None
    """
    # Try primary pattern
    match = re.search(pattern, response, re.IGNORECASE)
    if match:
        answer = match.group(1).strip()
        # Extract just the number
        number_match = re.search(r"[\d,]+\.?\d*", answer)
        if number_match:
            return number_match.group(0).replace(",", "")

    # Fallback: look for numbers at the end of response
    lines = response.strip().split("\n")
    for line in reversed(lines):
        number_match = re.search(r"[\d,]+\.?\d*", line)
        if number_match:
            return number_match.group(0).replace(",", "")

    return None


def normalize_answer(answer: str) -> str:
    """Normalize answer for comparison."""
    if answer is None:
        return ""
    # Remove commas, spaces, leading zeros
    answer = str(answer).replace(",", "").replace(" ", "").strip()
    # Convert to float and back to handle decimals consistently
    try:
        num = float(answer)
        if num.is_integer():
            return str(int(num))
        return str(num)
    except (ValueError, AttributeError):
        return answer


def check_correctness(pred: str, gold: str) -> bool:
    """Check if predicted answer matches gold answer."""
    return normalize_answer(pred) == normalize_answer(gold)


def run_inference(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run inference on GSM8K dataset.

    Args:
        cfg: Configuration dictionary

    Returns:
        Metrics dictionary
    """
    # Extract config sections
    run_cfg = cfg["run"]
    method_cfg = run_cfg["method"]
    model_cfg = run_cfg["model"]
    dataset_cfg = run_cfg["dataset"]
    inference_cfg = cfg["inference"]

    # Initialize WandB
    wandb_cfg = cfg["wandb"]
    if wandb_cfg["mode"] != "disabled":
        wandb.init(
            entity=wandb_cfg["entity"],
            project=wandb_cfg["project"],
            name=run_cfg["run_id"],
            config=cfg,
        )

    # Load dataset
    print(f"Loading dataset: {dataset_cfg['name']}...")
    cache_dir = dataset_cfg.get("cache_dir", ".cache")
    dataset = load_dataset(
        "gsm8k", "main", split=dataset_cfg["split"], cache_dir=cache_dir
    )

    # Limit samples based on mode
    num_samples = inference_cfg["samples"]
    dataset = dataset.select(range(min(num_samples, len(dataset))))
    print(f"Processing {len(dataset)} samples in {cfg['mode']} mode")

    # Run inference
    results = []
    correct_count = 0
    total_count = 0

    for idx, example in enumerate(dataset):
        question = example["question"]
        gold_answer = example["answer"].split("####")[-1].strip()

        # Format prompt
        prompt = format_prompt(
            question, method_cfg["prompt_template"], method_cfg["name"]
        )

        # Call model
        try:
            response = call_model_api(prompt, model_cfg)
        except Exception as e:
            print(f"Error calling model for example {idx}: {e}")
            response = ""

        # Extract answer
        pred_answer = extract_answer(
            response,
            run_cfg["inference"].get("answer_extraction_pattern", r"(.+)"),
            method_cfg["name"],
        )

        # Check correctness
        is_correct = check_correctness(pred_answer if pred_answer else "", gold_answer)
        if is_correct:
            correct_count += 1
        total_count += 1

        # Record result
        result = {
            "example_id": idx,
            "question": question,
            "gold_answer": gold_answer,
            "prompt": prompt,
            "response": response,
            "pred_answer": pred_answer,
            "correct": is_correct,
        }
        results.append(result)

        # Log progress
        if (idx + 1) % 10 == 0 or idx == 0:
            current_acc = correct_count / total_count
            print(f"Progress: {idx + 1}/{len(dataset)} | Accuracy: {current_acc:.3f}")
            if wandb_cfg["mode"] != "disabled":
                wandb.log({"progress": idx + 1, "accuracy": current_acc})

    # Calculate final metrics
    accuracy = correct_count / total_count if total_count > 0 else 0.0
    metrics = {
        "accuracy": accuracy,
        "correct": correct_count,
        "total": total_count,
        "method": method_cfg["name"],
        "model": model_cfg["name"],
        "dataset": dataset_cfg["name"],
        "samples": num_samples,
    }

    print(f"\n=== Final Results ===")
    print(f"Accuracy: {accuracy:.4f} ({correct_count}/{total_count})")
    print(f"Method: {method_cfg['name']}")

    # Save results
    results_dir = Path(cfg["results_dir"]) / run_cfg["run_id"]
    results_dir.mkdir(parents=True, exist_ok=True)

    # Save detailed results to JSONL
    results_file = results_dir / "results.jsonl"
    with open(results_file, "w") as f:
        for result in results:
            f.write(json.dumps(result) + "\n")
    print(f"Saved detailed results to: {results_file}")

    # Save metrics
    metrics_file = results_dir / "metrics.json"
    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics to: {metrics_file}")

    # Log to WandB
    if wandb_cfg["mode"] != "disabled":
        wandb.summary.update(metrics)
        print(f"WandB run: {wandb.run.url}")
        wandb.finish()

    # Perform validation based on mode
    perform_validation(cfg, metrics, results)

    return metrics


def perform_validation(
    cfg: Dict[str, Any], metrics: Dict[str, Any], results: List[Dict]
) -> None:
    """
    Perform mode-specific validation and emit verdict.

    Args:
        cfg: Configuration dictionary
        metrics: Computed metrics
        results: Individual example results
    """
    mode = cfg["mode"]

    if mode == "sanity":
        # Sanity validation
        samples = metrics["total"]
        accuracy = metrics["accuracy"]

        # Check if at least 5 samples processed
        if samples < 5:
            print(
                f"SANITY_VALIDATION: FAIL reason=insufficient_samples (got {samples}, need >= 5)"
            )
            print(
                f"SANITY_VALIDATION_SUMMARY: {json.dumps({'samples': samples, 'accuracy': accuracy})}"
            )
            sys.exit(1)

        # Check if all outputs are valid
        invalid_count = sum(1 for r in results if r["pred_answer"] is None)
        if invalid_count == len(results):
            print(f"SANITY_VALIDATION: FAIL reason=all_outputs_invalid")
            print(
                f"SANITY_VALIDATION_SUMMARY: {json.dumps({'samples': samples, 'accuracy': accuracy, 'invalid_count': invalid_count})}"
            )
            sys.exit(1)

        # Check if accuracy is not always 0 (at least one correct)
        if accuracy == 0.0 and samples >= 5:
            print(f"SANITY_VALIDATION: FAIL reason=zero_accuracy")
            print(
                f"SANITY_VALIDATION_SUMMARY: {json.dumps({'samples': samples, 'accuracy': accuracy})}"
            )
            sys.exit(1)

        # Check for finite metrics
        if not all(
            isinstance(v, (int, float))
            and not (
                isinstance(v, float)
                and (v != v or v == float("inf") or v == float("-inf"))
            )
            for k, v in metrics.items()
            if isinstance(v, (int, float))
        ):
            print(f"SANITY_VALIDATION: FAIL reason=non_finite_metrics")
            print(
                f"SANITY_VALIDATION_SUMMARY: {json.dumps({'samples': samples, 'accuracy': accuracy})}"
            )
            sys.exit(1)

        print(f"SANITY_VALIDATION: PASS")
        print(
            f"SANITY_VALIDATION_SUMMARY: {json.dumps({'samples': samples, 'accuracy': accuracy, 'correct': metrics['correct']})}"
        )

    elif mode == "pilot":
        # Pilot validation
        samples = metrics["total"]
        accuracy = metrics["accuracy"]

        # Check if at least 50 samples processed
        if samples < 50:
            print(
                f"PILOT_VALIDATION: FAIL reason=insufficient_samples (got {samples}, need >= 50)"
            )
            print(
                f"PILOT_VALIDATION_SUMMARY: {json.dumps({'samples': samples, 'primary_metric': 'accuracy', 'primary_metric_value': accuracy})}"
            )
            sys.exit(1)

        # Check if primary metric is computed and finite
        if (
            accuracy != accuracy
            or accuracy == float("inf")
            or accuracy == float("-inf")
        ):
            print(f"PILOT_VALIDATION: FAIL reason=non_finite_metric")
            print(
                f"PILOT_VALIDATION_SUMMARY: {json.dumps({'samples': samples, 'primary_metric': 'accuracy', 'primary_metric_value': accuracy})}"
            )
            sys.exit(1)

        # Check if outputs are non-trivial (not all identical)
        unique_answers = len(
            set(r["pred_answer"] for r in results if r["pred_answer"] is not None)
        )
        if unique_answers <= 1:
            print(f"PILOT_VALIDATION: FAIL reason=trivial_outputs")
            print(
                f"PILOT_VALIDATION_SUMMARY: {json.dumps({'samples': samples, 'primary_metric': 'accuracy', 'primary_metric_value': accuracy, 'unique_answers': unique_answers})}"
            )
            sys.exit(1)

        print(f"PILOT_VALIDATION: PASS")
        print(
            f"PILOT_VALIDATION_SUMMARY: {json.dumps({'samples': samples, 'primary_metric': 'accuracy', 'primary_metric_value': accuracy, 'correct': metrics['correct'], 'unique_answers': unique_answers})}"
        )


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to config YAML file")
    args = parser.parse_args()

    # Load config
    cfg = load_config(args.config)

    # Run inference
    run_inference(cfg)


if __name__ == "__main__":
    main()
