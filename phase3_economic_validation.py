#!/usr/bin/env python3
"""
Phase 3: Economic Validation
Cost-per-correct-answer metrics, deployment ROI analysis

Measures: What is the actual economic trade-off between cost reduction and reliability?
"""

import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Any, Tuple
import time
import statistics

from eval_hallucination_hyperparams import HyperparamHallucinationEvaluator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_economic_test_dataset(output_file: str) -> None:
    """Create dataset for economic analysis - production-like workload."""

    # Simulate real-world question types and frequencies
    production_workload = {
        "customer_support": {
            "frequency_pct": 40,
            "questions": [
                {"prompt": "What are your business hours?", "expected": "9 AM to 5 PM weekdays", "type": "known"},
                {"prompt": "How do I reset my password?", "expected": "click forgot password link", "type": "known"},
                {"prompt": "What is your return policy?", "expected": "30 day return window", "type": "known"},
                {"prompt": "How do I cancel my subscription?", "expected": "contact customer service", "type": "known"},
                {"prompt": "What payment methods do you accept?", "expected": "credit cards and PayPal", "type": "known"}
            ]
        },
        "factual_lookup": {
            "frequency_pct": 30,
            "questions": [
                {"prompt": "What is the capital of France?", "expected": "Paris", "type": "known"},
                {"prompt": "How many days are in February during a leap year?", "expected": "29", "type": "known"},
                {"prompt": "What is the chemical symbol for gold?", "expected": "Au", "type": "known"},
                {"prompt": "Who wrote Romeo and Juliet?", "expected": "William Shakespeare", "type": "known"},
                {"prompt": "What is the largest planet in our solar system?", "expected": "Jupiter", "type": "known"}
            ]
        },
        "calculation": {
            "frequency_pct": 20,
            "questions": [
                {"prompt": "What is 15% of 100?", "expected": "15", "type": "known"},
                {"prompt": "How many seconds are in an hour?", "expected": "3600", "type": "known"},
                {"prompt": "What is 12 times 8?", "expected": "96", "type": "known"},
                {"prompt": "Convert 100 degrees Fahrenheit to Celsius", "expected": "37.8 degrees", "type": "known"},
                {"prompt": "What is the square root of 64?", "expected": "8", "type": "known"}
            ]
        },
        "complex_reasoning": {
            "frequency_pct": 10,
            "questions": [
                {"prompt": "If a train leaves Chicago at 2 PM traveling 60 mph and arrives in Milwaukee at 4 PM, how far is Milwaukee from Chicago?", "expected": "120 miles", "type": "known"},
                {"prompt": "A shirt costs $25 and is on sale for 20% off. What is the sale price?", "expected": "$20", "type": "known"},
                {"prompt": "If you invest $1000 at 5% annual interest, how much will you have after 2 years?", "expected": "$1102.50", "type": "known"},
                {"prompt": "A rectangle has a length of 10 meters and width of 6 meters. What is its area?", "expected": "60 square meters", "type": "known"},
                {"prompt": "If it takes 3 workers 6 hours to build a wall, how long would it take 6 workers?", "expected": "3 hours", "type": "known"}
            ]
        }
    }

    # Generate weighted dataset based on production frequencies
    economic_questions = []
    question_id = 1

    for category, data in production_workload.items():
        frequency = data["frequency_pct"]
        questions = data["questions"]

        # Generate questions proportional to frequency (out of 100 total)
        num_questions = frequency
        questions_per_type = num_questions // len(questions)

        for q in questions:
            for i in range(questions_per_type):
                question = {
                    "id": f"econ_{question_id:03d}",
                    "prompt": q["prompt"],
                    "expected": q["expected"],
                    "type": q["type"],
                    "category": category,
                    "business_value": get_business_value(category)
                }
                economic_questions.append(question)
                question_id += 1

    with open(output_file, 'w') as f:
        for q in economic_questions:
            f.write(json.dumps(q) + '\n')

    logger.info(f"Created economic validation dataset: {len(economic_questions)} questions across {len(production_workload)} categories")


def get_business_value(category: str) -> float:
    """Assign business value weights to different question categories."""
    values = {
        "customer_support": 0.8,     # High value - customer satisfaction
        "factual_lookup": 0.6,       # Medium value - information accuracy
        "calculation": 0.9,          # Very high value - financial/numerical correctness
        "complex_reasoning": 1.0     # Maximum value - critical thinking tasks
    }
    return values.get(category, 0.5)


def run_economic_benchmark(dataset_file: str, model_name: str, output_dir: str):
    """Run economic validation across production-optimized configurations."""

    # Test economically relevant configurations
    test_configs = [
        # Production-safe configurations
        {"name": "production_safe", "quant": "none", "temp": 0.3, "description": "Conservative baseline"},
        {"name": "cost_optimized", "quant": "nf4", "temp": 0.3, "description": "60% memory reduction, conservative"},
        {"name": "aggressive_cost", "quant": "fp4_double_quant", "temp": 0.3, "description": "65% memory reduction"},

        # Higher performance configurations
        {"name": "balanced", "quant": "int8", "temp": 0.7, "description": "Moderate compression, better creativity"},
        {"name": "high_performance", "quant": "none", "temp": 0.7, "description": "No compression, optimal temperature"},

        # Edge case configurations
        {"name": "maximum_cost_reduction", "quant": "fp4_double_quant", "temp": 0.1, "description": "Extreme cost cutting"},
        {"name": "risky_creative", "quant": "nf4", "temp": 1.2, "description": "Cost efficient but creative"}
    ]

    results = {}

    for config in test_configs:
        logger.info(f"Testing economic configuration: {config['name']}")

        # Initialize evaluator
        evaluator = HyperparamHallucinationEvaluator(
            model_name=model_name,
            quantization_type=config["quant"]
        )

        # Load model and measure loading time
        load_start = time.time()
        evaluator.load_model()
        load_time = time.time() - load_start

        # Single hyperparameter config
        hyperparam_configs = [{"temperature": config["temp"], "top_p": 0.9, "top_k": 50}]

        # Run evaluation with detailed timing
        output_file = f"{output_dir}/economic_{config['name']}.json"

        # Load prompts
        prompts = []
        with open(dataset_file, 'r') as f:
            for line in f:
                if line.strip():
                    prompts.append(json.loads(line))

        # Custom evaluation with economic metrics
        start_time = time.time()
        evaluator.evaluate_prompts_with_hyperparams(prompts, hyperparam_configs, output_file)
        total_time = time.time() - start_time

        # Load results and calculate economic metrics
        with open(output_file) as f:
            result_data = json.load(f)

        results[config["name"]] = calculate_economic_metrics(
            result_data["results"],
            config,
            load_time,
            total_time
        )

        logger.info(f"Completed economic test: {config['name']}")

    # Save compiled economic analysis
    economic_analysis = {
        "experiment_design": {
            "configurations_tested": len(test_configs),
            "total_questions": len(prompts),
            "business_categories": ["customer_support", "factual_lookup", "calculation", "complex_reasoning"]
        },
        "configuration_performance": results,
        "economic_rankings": rank_configurations(results)
    }

    with open(f"{output_dir}/economic_analysis.json", 'w') as f:
        json.dump(economic_analysis, f, indent=2)

    return economic_analysis


def calculate_economic_metrics(results: List[Dict], config: Dict, load_time: float, total_time: float) -> Dict[str, Any]:
    """Calculate comprehensive economic metrics for a configuration."""

    # Basic performance metrics
    total_questions = len(results)
    correct_answers = sum([r["factually_correct"] for r in results])
    wrong_answers = sum([not r["factually_correct"] and not r["is_refusal"] for r in results])
    refusals = sum([r["is_refusal"] for r in results])

    accuracy = correct_answers / total_questions
    error_rate = wrong_answers / total_questions
    refusal_rate = refusals / total_questions

    # Business value weighted metrics
    business_value_delivered = 0
    total_business_value = 0

    for result in results:
        bv = result.get("business_value", 0.5)
        total_business_value += bv
        if result["factually_correct"]:
            business_value_delivered += bv

    value_efficiency = business_value_delivered / total_business_value if total_business_value > 0 else 0

    # Cost metrics
    avg_memory = statistics.mean([r["metrics"]["gpu_memory_mb"] for r in results if r["metrics"].get("gpu_memory_mb")])
    avg_latency = statistics.mean([r["metrics"]["generation_time_ms"] for r in results])
    throughput = total_questions / total_time  # questions per second

    # Memory cost (assuming $0.50/GB/hour for GPU memory)
    memory_cost_per_hour = (avg_memory / 1024) * 0.50
    cost_per_question = memory_cost_per_hour / (throughput * 3600)  # Cost per question in dollars

    # Economic efficiency metrics
    cost_per_correct_answer = cost_per_question / accuracy if accuracy > 0 else float('inf')
    value_per_dollar = business_value_delivered / (cost_per_question * total_questions) if cost_per_question > 0 else 0

    # Risk metrics
    high_value_questions = [r for r in results if r.get("business_value", 0.5) >= 0.8]
    high_value_accuracy = sum([r["factually_correct"] for r in high_value_questions]) / len(high_value_questions) if high_value_questions else 0

    # Category-specific performance
    category_performance = {}
    for category in ["customer_support", "factual_lookup", "calculation", "complex_reasoning"]:
        cat_results = [r for r in results if r.get("category") == category]
        if cat_results:
            cat_accuracy = sum([r["factually_correct"] for r in cat_results]) / len(cat_results)
            category_performance[category] = {
                "accuracy": cat_accuracy,
                "sample_size": len(cat_results)
            }

    return {
        "configuration": config,
        "performance_metrics": {
            "accuracy": accuracy,
            "error_rate": error_rate,
            "refusal_rate": refusal_rate,
            "high_value_accuracy": high_value_accuracy,
            "value_efficiency": value_efficiency
        },
        "cost_metrics": {
            "avg_memory_mb": avg_memory,
            "avg_latency_ms": avg_latency,
            "throughput_qps": throughput,
            "memory_cost_per_hour": memory_cost_per_hour,
            "cost_per_question": cost_per_question,
            "cost_per_correct_answer": cost_per_correct_answer
        },
        "economic_efficiency": {
            "value_per_dollar": value_per_dollar,
            "business_value_delivered": business_value_delivered,
            "total_business_value": total_business_value
        },
        "category_performance": category_performance,
        "sample_size": total_questions
    }


def rank_configurations(results: Dict[str, Dict]) -> Dict[str, List]:
    """Rank configurations by different economic criteria."""

    configs = list(results.keys())

    rankings = {
        "best_accuracy": sorted(configs, key=lambda x: results[x]["performance_metrics"]["accuracy"], reverse=True),
        "lowest_cost": sorted(configs, key=lambda x: results[x]["cost_metrics"]["cost_per_question"]),
        "best_value_efficiency": sorted(configs, key=lambda x: results[x]["economic_efficiency"]["value_per_dollar"], reverse=True),
        "best_cost_per_correct": sorted(configs, key=lambda x: results[x]["cost_metrics"]["cost_per_correct_answer"]),
        "safest_high_value": sorted(configs, key=lambda x: results[x]["performance_metrics"]["high_value_accuracy"], reverse=True)
    }

    return rankings


def print_economic_summary(analysis: Dict[str, Any]) -> None:
    """Print economic analysis summary."""

    print("PHASE 3 ECONOMIC VALIDATION RESULTS:")
    print("=" * 60)

    results = analysis["configuration_performance"]
    rankings = analysis["economic_rankings"]

    print("\nCOST EFFICIENCY RANKING:")
    print("-" * 30)
    for i, config_name in enumerate(rankings["best_cost_per_correct"][:3], 1):
        config = results[config_name]
        cost = config["cost_metrics"]["cost_per_correct_answer"]
        accuracy = config["performance_metrics"]["accuracy"]
        print(f"{i}. {config_name}: ${cost:.6f} per correct answer ({accuracy:.1%} accuracy)")

    print("\nMEMORY COST REDUCTION:")
    print("-" * 30)
    baseline_memory = None
    for config_name, data in results.items():
        if "production_safe" in config_name:
            baseline_memory = data["cost_metrics"]["avg_memory_mb"]
            break

    if baseline_memory:
        for config_name, data in results.items():
            memory = data["cost_metrics"]["avg_memory_mb"]
            reduction = (baseline_memory - memory) / baseline_memory * 100
            accuracy = data["performance_metrics"]["accuracy"]
            print(f"{config_name}: {reduction:+.1f}% memory reduction, {accuracy:.1%} accuracy")

    print("\nBUSINESS VALUE EFFICIENCY:")
    print("-" * 30)
    for i, config_name in enumerate(rankings["best_value_efficiency"][:3], 1):
        config = results[config_name]
        value_eff = config["economic_efficiency"]["value_per_dollar"]
        accuracy = config["performance_metrics"]["accuracy"]
        print(f"{i}. {config_name}: {value_eff:.2f} value per dollar ({accuracy:.1%} accuracy)")

    print("\nHIGH-VALUE QUESTION SAFETY:")
    print("-" * 30)
    for config_name, data in results.items():
        hv_accuracy = data["performance_metrics"]["high_value_accuracy"]
        overall_accuracy = data["performance_metrics"]["accuracy"]
        print(f"{config_name}: {hv_accuracy:.1%} high-value accuracy (vs {overall_accuracy:.1%} overall)")


def main():
    parser = argparse.ArgumentParser(description="Phase 3: Economic Validation")
    parser.add_argument("--model", required=True, help="Model name")
    parser.add_argument("--create-dataset", action="store_true", help="Create economic test dataset")
    parser.add_argument("--run-benchmark", action="store_true", help="Run economic benchmark")
    parser.add_argument("--output-dir", default="phase3_results", help="Output directory")

    args = parser.parse_args()

    Path(args.output_dir).mkdir(exist_ok=True)

    if args.create_dataset:
        create_economic_test_dataset(f"{args.output_dir}/economic_dataset.jsonl")

    if args.run_benchmark:
        analysis = run_economic_benchmark(
            f"{args.output_dir}/economic_dataset.jsonl",
            args.model,
            args.output_dir
        )
        print_economic_summary(analysis)


if __name__ == "__main__":
    main()