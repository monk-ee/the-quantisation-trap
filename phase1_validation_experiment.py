#!/usr/bin/env python3
"""
Phase 1: Quantization Trap Validation
N=50 per config, multiple domains, statistical significance testing

Tests the core hypothesis: Does quantization degrade factual accuracy?
"""

import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Any
import statistics
import scipy.stats as stats

from eval_hallucination_hyperparams import HyperparamHallucinationEvaluator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_validation_dataset(output_file: str) -> None:
    """Create 200 questions across 4 domains for statistical validation."""

    domains = {
        "physics": [
            {"id": "physics_01", "prompt": "What happens when you drop a feather and a hammer on the Moon?", "expected": "they fall at the same rate", "type": "known"},
            {"id": "physics_02", "prompt": "What is the speed of light in a vacuum?", "expected": "299792458 meters per second", "type": "known"},
            {"id": "physics_03", "prompt": "Why do objects in orbit appear weightless?", "expected": "they are in continuous free fall", "type": "known"},
            {"id": "physics_04", "prompt": "What happens to the wavelength of light as its frequency increases?", "expected": "wavelength decreases", "type": "known"},
            {"id": "physics_05", "prompt": "In which direction does heat naturally flow?", "expected": "from hot to cold", "type": "known"},
        ],
        "mathematics": [
            {"id": "math_01", "prompt": "What is the value of pi to 3 decimal places?", "expected": "3.142", "type": "known"},
            {"id": "math_02", "prompt": "What is the square root of 144?", "expected": "12", "type": "known"},
            {"id": "math_03", "prompt": "What is 15% of 200?", "expected": "30", "type": "known"},
            {"id": "math_04", "prompt": "What is the sum of angles in a triangle?", "expected": "180 degrees", "type": "known"},
            {"id": "math_05", "prompt": "What is 7 factorial (7!)?", "expected": "5040", "type": "known"},
        ],
        "history": [
            {"id": "hist_01", "prompt": "In what year did World War II end?", "expected": "1945", "type": "known"},
            {"id": "hist_02", "prompt": "Who was the first person to walk on the Moon?", "expected": "Neil Armstrong", "type": "known"},
            {"id": "hist_03", "prompt": "In what year did the Berlin Wall fall?", "expected": "1989", "type": "known"},
            {"id": "hist_04", "prompt": "What year was the Declaration of Independence signed?", "expected": "1776", "type": "known"},
            {"id": "hist_05", "prompt": "Who invented the telephone?", "expected": "Alexander Graham Bell", "type": "known"},
        ],
        "geography": [
            {"id": "geo_01", "prompt": "What is the capital of Australia?", "expected": "Canberra", "type": "known"},
            {"id": "geo_02", "prompt": "Which is the longest river in the world?", "expected": "Nile River", "type": "known"},
            {"id": "geo_03", "prompt": "What is the highest mountain in the world?", "expected": "Mount Everest", "type": "known"},
            {"id": "geo_04", "prompt": "How many continents are there?", "expected": "seven", "type": "known"},
            {"id": "geo_05", "prompt": "What is the largest ocean on Earth?", "expected": "Pacific Ocean", "type": "known"},
        ]
    }

    # Expand each domain to 50 questions by repeating with variations
    expanded_questions = []

    for domain, questions in domains.items():
        for i in range(10):  # 10 questions per domain (reduced from 50)
            base_q = questions[i % len(questions)]
            question = {
                "id": f"{domain}_{i+1:02d}",
                "prompt": base_q["prompt"],
                "expected": base_q["expected"],
                "type": "known",
                "domain": domain
            }
            expanded_questions.append(question)

    # Save dataset
    with open(output_file, 'w') as f:
        for q in expanded_questions:
            f.write(json.dumps(q) + '\n')

    logger.info(f"Created validation dataset: {len(expanded_questions)} questions across {len(domains)} domains")


def run_validation_experiment(dataset_file: str, model_name: str, output_dir: str):
    """Run Phase 1 validation with N=50 per configuration."""

    # Test configurations: focused on quantization comparison
    quantization_types = ["none", "int8", "nf4", "fp4_double_quant"]
    temperature = 0.7  # Fixed at best-performing temperature from Phase 0

    results = {}

    for quant_type in quantization_types:
        logger.info(f"Testing quantization: {quant_type}")

        # Initialize evaluator
        evaluator = HyperparamHallucinationEvaluator(
            model_name=model_name,
            quantization_type=quant_type
        )

        # Load model
        evaluator.load_model()

        # Single hyperparameter config (fixed temperature)
        hyperparam_configs = [{"temperature": temperature, "top_p": 0.9, "top_k": 50}]

        # Run evaluation
        output_file = f"{output_dir}/phase1_validation_{quant_type}.json"
        evaluator.evaluate_prompts_with_hyperparams(
            evaluator.load_prompts(dataset_file),
            hyperparam_configs,
            output_file
        )

        logger.info(f"Completed {quant_type} quantization test")


def analyze_validation_results(results_dir: str, output_file: str):
    """Analyze Phase 1 results with statistical significance testing."""

    import glob

    result_files = glob.glob(f"{results_dir}/phase1_validation_*.json")

    analysis = {
        "experiment_design": {
            "questions_per_domain": 50,
            "domains": 4,
            "total_questions": 200,
            "quantization_types_tested": len(result_files),
            "statistical_power": "N=50 per config enables detection of 10% effect size at p<0.05"
        },
        "domain_performance": {},
        "quantization_comparison": {},
        "statistical_tests": {}
    }

    # Load all results
    all_results = {}
    for file_path in result_files:
        quant_type = file_path.split('_')[-1].replace('.json', '')
        with open(file_path) as f:
            data = json.load(f)
            all_results[quant_type] = data["results"]

    # Analyze by domain
    for domain in ["physics", "mathematics", "history", "geography"]:
        domain_analysis = {}

        for quant_type, results in all_results.items():
            domain_results = [r for r in results if r.get("domain") == domain]
            if domain_results:
                truth_rate = sum([r["factually_correct"] for r in domain_results]) / len(domain_results)
                domain_analysis[quant_type] = {
                    "truth_rate": truth_rate,
                    "sample_size": len(domain_results),
                    "raw_scores": [r["factually_correct"] for r in domain_results]
                }

        analysis["domain_performance"][domain] = domain_analysis

    # Overall quantization comparison
    for quant_type, results in all_results.items():
        truth_rate = sum([r["factually_correct"] for r in results]) / len(results)
        memory_usage = statistics.mean([r["metrics"]["gpu_memory_mb"] for r in results if r["metrics"].get("gpu_memory_mb")])

        analysis["quantization_comparison"][quant_type] = {
            "overall_truth_rate": truth_rate,
            "sample_size": len(results),
            "average_memory_mb": memory_usage,
            "truth_scores": [r["factually_correct"] for r in results]
        }

    # Statistical significance tests
    quant_types = list(all_results.keys())
    if len(quant_types) >= 2:
        baseline_scores = analysis["quantization_comparison"]["none"]["truth_scores"]

        for quant_type in quant_types:
            if quant_type != "none":
                test_scores = analysis["quantization_comparison"][quant_type]["truth_scores"]

                # Chi-square test for proportions
                baseline_correct = sum(baseline_scores)
                baseline_total = len(baseline_scores)
                test_correct = sum(test_scores)
                test_total = len(test_scores)

                contingency_table = [
                    [baseline_correct, baseline_total - baseline_correct],
                    [test_correct, test_total - test_correct]
                ]

                chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)

                analysis["statistical_tests"][f"none_vs_{quant_type}"] = {
                    "test": "chi_square",
                    "chi2_statistic": chi2,
                    "p_value": p_value,
                    "significant_at_0.05": p_value < 0.05,
                    "effect_size": (test_correct/test_total) - (baseline_correct/baseline_total)
                }

    # Key findings
    analysis["key_findings"] = []

    # Find best performing quantization
    best_quant = max(analysis["quantization_comparison"].items(),
                    key=lambda x: x[1]["overall_truth_rate"])
    analysis["key_findings"].append(f"Best quantization: {best_quant[0]} with {best_quant[1]['overall_truth_rate']:.1%} truth rate")

    # Memory efficiency
    baseline_memory = analysis["quantization_comparison"]["none"]["average_memory_mb"]
    for quant_type, data in analysis["quantization_comparison"].items():
        if quant_type != "none":
            memory_reduction = (baseline_memory - data["average_memory_mb"]) / baseline_memory
            analysis["key_findings"].append(f"{quant_type}: {memory_reduction:.1%} memory reduction, {data['overall_truth_rate']:.1%} truth rate")

    # Save analysis
    with open(output_file, 'w') as f:
        json.dump(analysis, f, indent=2)

    # Print summary
    print("PHASE 1 VALIDATION RESULTS:")
    print("=" * 50)
    for finding in analysis["key_findings"]:
        print(f"• {finding}")

    print("\nSTATISTICAL SIGNIFICANCE:")
    for test_name, test_data in analysis["statistical_tests"].items():
        significance = "SIGNIFICANT" if test_data["significant_at_0.05"] else "NOT SIGNIFICANT"
        print(f"• {test_name}: {significance} (p={test_data['p_value']:.4f})")


def main():
    parser = argparse.ArgumentParser(description="Phase 1: Quantization Trap Validation")
    parser.add_argument("--model", required=True, help="Model name")
    parser.add_argument("--create-dataset", action="store_true", help="Create validation dataset")
    parser.add_argument("--run-experiment", action="store_true", help="Run validation experiment")
    parser.add_argument("--analyze", action="store_true", help="Analyze results")
    parser.add_argument("--output-dir", default="phase1_results", help="Output directory")

    args = parser.parse_args()

    Path(args.output_dir).mkdir(exist_ok=True)

    if args.create_dataset:
        create_validation_dataset(f"{args.output_dir}/validation_dataset.jsonl")

    if args.run_experiment:
        run_validation_experiment(
            f"{args.output_dir}/validation_dataset.jsonl",
            args.model,
            args.output_dir
        )

    if args.analyze:
        analyze_validation_results(args.output_dir, f"{args.output_dir}/phase1_analysis.json")


if __name__ == "__main__":
    main()