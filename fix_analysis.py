#!/usr/bin/env python3
"""
Quick fix for existing analysis results without re-running inference.
Re-classifies responses and regenerates analysis.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any
import glob

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def improved_classify_response(response: str, expected: str, question_type: str) -> Dict[str, bool]:
    """Improved classification logic matching the fixed evaluator."""
    response_lower = response.lower()

    # Uncertainty markers for refusal detection
    uncertainty_markers = [
        "i don't know", "i do not know", "i'm not sure", "i am not sure", "uncertain",
        "unclear", "cannot determine", "can't determine", "insufficient information",
        "not enough information", "unable to answer", "cannot answer", "can't answer",
        "no information", "unknown", "i cannot", "i can't", "not possible to",
        "impossible to", "no way to"
    ]

    # Check for refusal/uncertainty markers
    is_refusal = any(marker in response_lower for marker in uncertainty_markers)

    # Check correctness for "known" questions
    is_correct = False
    if question_type == "known":
        expected_lower = expected.lower()

        # Improved semantic matching for key concepts
        expected_words = set(expected_lower.split())
        response_words = set(response_lower.split())

        # For physics/science answers, check if core concepts are present
        if "fall" in expected_lower and "rate" in expected_lower:
            # Check for concepts like "fall at same rate", "same rate", etc.
            has_fall = any(word in response_lower for word in ["fall", "falls", "falling"])
            has_same_rate = "same rate" in response_lower or "equal rate" in response_lower
            is_correct = has_fall and has_same_rate
        else:
            # Fallback to word overlap for other types
            # Require at least 50% of expected words to be present
            overlap = len(expected_words.intersection(response_words))
            is_correct = overlap >= len(expected_words) * 0.5

    # Classify as incorrect guess if not refusal and not correct
    is_incorrect_guess = not is_refusal and not is_correct

    return {
        "is_refusal": is_refusal,
        "is_correct": is_correct,
        "is_incorrect_guess": is_incorrect_guess
    }


def fix_result_file(file_path: str) -> None:
    """Fix classifications in a single result file."""
    logger.info(f"Fixing classifications in {file_path}")

    with open(file_path, 'r') as f:
        data = json.load(f)

    fixed_count = 0
    for result in data["results"]:
        # Re-classify with improved logic
        old_classification = {
            "is_refusal": result["is_refusal"],
            "is_correct": result["is_correct"],
            "is_incorrect_guess": result["is_incorrect_guess"]
        }

        new_classification = improved_classify_response(
            result["response"],
            result["expected"],
            result["type"]
        )

        # Update the result
        result.update(new_classification)

        # Check if classification changed
        if old_classification != new_classification:
            fixed_count += 1
            logger.debug(f"Fixed {result['id']}: {old_classification} -> {new_classification}")

    # Save fixed file
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=2)

    logger.info(f"Fixed {fixed_count} classifications in {file_path}")


def main():
    """Fix all result files and regenerate analysis."""

    # Find all result files
    result_files = glob.glob(
        "/Users/lyndonswan/PycharmProjects/TheQuantisationTrap/quantization-hallucination-results/hyperparam_results/hyperparam_*.json"
    )

    # Exclude analysis files
    result_files = [f for f in result_files if not f.endswith('_analysis.json')]

    logger.info(f"Found {len(result_files)} result files to fix")

    # Fix each file
    for file_path in result_files:
        if "analysis" not in file_path:  # Skip analysis files
            fix_result_file(file_path)

    logger.info("All result files fixed. Now regenerating analysis...")

    # Import and run the analysis report generator
    import sys
    sys.path.append('/Users/lyndonswan/PycharmProjects/TheQuantisationTrap')

    from hyperparam_analysis_report import HyperparamAnalyzer

    # Generate new analysis
    analyzer = HyperparamAnalyzer()
    analyzer.load_results(result_files)
    analyzer.generate_report(
        "/Users/lyndonswan/PycharmProjects/TheQuantisationTrap/quantization-hallucination-results/hyperparam_results/hyperparam_analysis_fixed.json"
    )

    logger.info("✅ Analysis regenerated with fixed classifications!")


if __name__ == "__main__":
    main()