"""Dataset preprocessing for GSM8K."""

import re
from datasets import load_dataset
from typing import Dict, List, Any


def load_gsm8k(
    split: str = "test", max_samples: int = None, cache_dir: str = ".cache"
) -> List[Dict[str, Any]]:
    """
    Load GSM8K dataset.

    Args:
        split: Dataset split (train or test)
        max_samples: Maximum number of samples to load
        cache_dir: Directory to cache downloaded datasets

    Returns:
        List of examples with 'question' and 'answer' fields
    """
    dataset = load_dataset("gsm8k", "main", split=split, cache_dir=cache_dir)

    examples = []
    for i, item in enumerate(dataset):
        if max_samples is not None and i >= max_samples:
            break

        # GSM8K format: question and answer (with #### separator)
        question = item["question"]
        answer_text = item["answer"]

        # Extract numeric answer after #### marker
        gold_answer = extract_gold_answer(answer_text)

        examples.append(
            {
                "question": question,
                "answer": gold_answer,
                "full_answer": answer_text,
                "index": i,
            }
        )

    return examples


def extract_gold_answer(answer_text: str) -> str:
    """
    Extract numeric answer from GSM8K answer field.
    GSM8K format: solution text followed by #### and the numeric answer.

    Args:
        answer_text: Full answer text with #### separator

    Returns:
        Normalized numeric answer as string
    """
    # Split by #### and take the last part
    parts = answer_text.split("####")
    if len(parts) > 1:
        numeric_answer = parts[-1].strip()
        # Remove commas and normalize
        numeric_answer = numeric_answer.replace(",", "")
        return numeric_answer

    # Fallback: try to extract any number
    numbers = re.findall(r"[+-]?\d+(?:\.\d+)?", answer_text)
    if numbers:
        return numbers[-1]

    return answer_text.strip()


def extract_model_answer(response: str, pattern: str = None) -> str:
    """
    Extract final answer from model response.

    Args:
        response: Model's full response text
        pattern: Regex pattern to extract answer (optional)

    Returns:
        Extracted answer string
    """
    # [VALIDATOR FIX - Attempt 1]
    # [PROBLEM]: Model answers are full sentences like "Janet makes $18 every day" but gold answers are just "18"
    # [CAUSE]: When custom pattern captures full sentence, normalize_answer doesn't extract numbers from text
    # [FIX]: After capturing with custom pattern, extract numeric value from the captured text using priority:
    #        1. Number after "=" (for calculations like "20 + 80 = 100")
    #        2. Number after answer verbs (is/are/makes/needs/etc.)
    #        3. First number in the sentence (most likely the answer)
    #
    # [OLD CODE]:
    # if pattern:
    #     match = re.search(pattern, response, re.IGNORECASE | re.DOTALL)
    #     if match:
    #         for group in match.groups():
    #             if group is not None:
    #                 return normalize_answer(group.strip())
    #
    # [NEW CODE]:
    if pattern:
        # Try custom pattern first
        match = re.search(pattern, response, re.IGNORECASE | re.DOTALL)
        if match:
            # Return first non-None group
            for group in match.groups():
                if group is not None:
                    captured_text = group.strip()

                    # Priority 1: Check for number after an equals sign (e.g., "20 + 80 = 100")
                    equals_match = re.search(
                        r"=\s*\$?([+-]?\d+(?:,\d{3})*(?:\.\d+)?)", captured_text
                    )
                    if equals_match:
                        return normalize_answer(equals_match.group(1))

                    # Priority 2: Check for number after common answer verbs
                    verb_match = re.search(
                        r"\b(?:is|are|makes?|needs?|have|earns?|costs?|takes?|totals?)\s+\$?([+-]?\d+(?:,\d{3})*(?:\.\d+)?)",
                        captured_text,
                        re.IGNORECASE,
                    )
                    if verb_match:
                        return normalize_answer(verb_match.group(1))

                    # Priority 3: Extract first number from the text
                    numbers = re.findall(
                        r"\$?([+-]?\d+(?:,\d{3})*(?:\.\d+)?)", captured_text
                    )
                    if numbers:
                        return normalize_answer(numbers[0])

                    # Fallback to original captured text if no number found
                    return normalize_answer(captured_text)

    # Default patterns for numeric answers
    patterns = [
        r"Final answer:\s*([+-]?\d+(?:\.\d+)?)",
        r"(?:answer is|equals?)\s*([+-]?\d+(?:\.\d+)?)",
        r"####\s*([+-]?\d+(?:\.\d+)?)",
        r"\$?\s*([+-]?\d+(?:\.\d+)?)\s*$",  # Last number in response
    ]

    for p in patterns:
        match = re.search(p, response, re.IGNORECASE | re.MULTILINE)
        if match:
            return normalize_answer(match.group(1).strip())

    # Last resort: extract last number
    numbers = re.findall(r"[+-]?\d+(?:\.\d+)?", response)
    if numbers:
        return normalize_answer(numbers[-1])

    return ""


def normalize_answer(answer: str) -> str:
    """
    Normalize numeric answer for comparison.

    Args:
        answer: Answer string

    Returns:
        Normalized answer
    """
    # Remove commas, dollar signs, and extra whitespace
    answer = answer.replace(",", "").replace("$", "").strip()

    # Try to convert to float and back to remove trailing zeros
    try:
        num = float(answer)
        # If it's an integer, return as int
        if num.is_integer():
            return str(int(num))
        return str(num)
    except (ValueError, AttributeError):
        return answer


def format_prompt(question: str, prompt_template: str) -> str:
    """
    Format question with prompt template.

    Args:
        question: Question text
        prompt_template: Template with {question} placeholder

    Returns:
        Formatted prompt
    """
    return prompt_template.format(question=question)


def compute_compactness(response: str) -> Dict[str, float]:
    """
    Compute compactness metrics for a response.

    Args:
        response: Model response text

    Returns:
        Dictionary with compactness metrics
    """
    lines = [l.strip() for l in response.split("\n") if l.strip()]
    words = response.split()

    return {
        "num_lines": len(lines),
        "num_words": len(words),
        "avg_words_per_line": len(words) / max(len(lines), 1),
    }


def check_robustness(response: str) -> Dict[str, bool]:
    """
    Check robustness indicators in response.

    Args:
        response: Model response text

    Returns:
        Dictionary with robustness flags
    """
    # Check for signs of revision or uncertainty
    revision_keywords = [
        "wait",
        "actually",
        "correction",
        "revised",
        "change",
        "instead",
    ]
    uncertainty_keywords = ["maybe", "perhaps", "might be", "could be", "not sure"]

    has_revision = any(keyword in response.lower() for keyword in revision_keywords)
    has_uncertainty = any(
        keyword in response.lower() for keyword in uncertainty_keywords
    )

    # Check for multiple different numeric answers (potential answer switching)
    numbers = re.findall(r"[+-]?\d+(?:\.\d+)?", response)
    unique_numbers = set(numbers)
    has_answer_switching = len(unique_numbers) > 3  # Heuristic threshold

    return {
        "has_revision": has_revision,
        "has_uncertainty": has_uncertainty,
        "has_answer_switching": has_answer_switching,
        "is_robust": not (has_revision or has_uncertainty or has_answer_switching),
    }
