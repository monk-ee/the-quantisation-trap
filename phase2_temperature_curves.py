#!/usr/bin/env python3
"""
Phase 2: Temperature Degradation Curve Mapping
Fine-grained temperature sweep to find exact breakdown points per quantization type

Maps: At what temperature does each quantization type start failing?
"""

import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Any, Tuple
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
import statistics

from eval_hallucination_hyperparams import HyperparamHallucinationEvaluator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_temperature_sweep_dataset(output_file: str) -> None:
    """Create focused dataset for temperature curve mapping."""

    # Use high-quality physics questions for precise measurement
    questions = [
        {"id": "curve_01", "prompt": "What happens when you drop a feather and a hammer in a vacuum?", "expected": "they fall at the same rate", "type": "known"},
        {"id": "curve_02", "prompt": "What is the acceleration due to gravity on Earth?", "expected": "9.8 meters per second squared", "type": "known"},
        {"id": "curve_03", "prompt": "In a vacuum, do heavy and light objects fall at different speeds?", "expected": "no they fall at the same speed", "type": "known"},
        {"id": "curve_04", "prompt": "What determines how fast an object falls in a vacuum?", "expected": "gravity alone", "type": "known"},
        {"id": "curve_05", "prompt": "Why do objects of different masses fall at the same rate in a vacuum?", "expected": "no air resistance", "type": "known"},
        {"id": "curve_06", "prompt": "What is the relationship between mass and falling speed in vacuum?", "expected": "no relationship", "type": "known"},
        {"id": "curve_07", "prompt": "Will a bowling ball and tennis ball hit the ground simultaneously if dropped in vacuum?", "expected": "yes", "type": "known"},
        {"id": "curve_08", "prompt": "What did Galileo discover about falling objects?", "expected": "they fall at the same rate", "type": "known"},
        {"id": "curve_09", "prompt": "In the absence of air resistance, what affects falling speed?", "expected": "only gravity", "type": "known"},
        {"id": "curve_10", "prompt": "Do heavier objects fall faster than lighter ones in a vacuum?", "expected": "no", "type": "known"}
    ]

    # Replicate 5 times for speed (N=50 per temperature)
    expanded_questions = []
    for i in range(5):
        for q in questions:
            expanded_q = q.copy()
            expanded_q["id"] = f"{q['id']}_rep{i+1:02d}"
            expanded_questions.append(expanded_q)

    with open(output_file, 'w') as f:
        for q in expanded_questions:
            f.write(json.dumps(q) + '\n')

    logger.info(f"Created temperature curve dataset: {len(expanded_questions)} questions")


def run_temperature_sweep(dataset_file: str, model_name: str, quantization_type: str, output_dir: str):
    """Run fine-grained temperature sweep for one quantization type."""

    # Focused temperature range: key points only
    temperatures = [0.3, 0.7, 1.0, 1.5]  # 4 strategic points instead of 20

    logger.info(f"Testing {quantization_type} across {len(temperatures)} temperature points")

    # Initialize evaluator
    evaluator = HyperparamHallucinationEvaluator(
        model_name=model_name,
        quantization_type=quantization_type
    )

    # Load model once
    evaluator.load_model()

    # Load dataset once
    prompts = []
    with open(dataset_file, 'r') as f:
        for line in f:
            if line.strip():
                prompts.append(json.loads(line))

    results_by_temperature = {}

    for temp in temperatures:
        logger.info(f"Testing temperature {temp} for {quantization_type}")

        hyperparam_configs = [{"temperature": temp, "top_p": 0.9, "top_k": 50}]

        # Run evaluation for this temperature
        temp_output = f"{output_dir}/temp_curve_{quantization_type}_temp{temp}.json"
        evaluator.evaluate_prompts_with_hyperparams(prompts, hyperparam_configs, temp_output)

        # Load and store results
        with open(temp_output) as f:
            temp_data = json.load(f)
            truth_rate = sum([r["factually_correct"] for r in temp_data["results"]]) / len(temp_data["results"])
            results_by_temperature[temp] = {
                "truth_rate": truth_rate,
                "sample_size": len(temp_data["results"]),
                "results": temp_data["results"]
            }

    # Save compiled results
    curve_data = {
        "quantization_type": quantization_type,
        "model_name": model_name,
        "temperature_range": [min(temperatures), max(temperatures)],
        "temperature_points": len(temperatures),
        "results_by_temperature": results_by_temperature
    }

    with open(f"{output_dir}/temperature_curve_{quantization_type}.json", 'w') as f:
        json.dump(curve_data, f, indent=2)

    logger.info(f"Completed temperature curve for {quantization_type}")


def analyze_temperature_curves(results_dir: str, output_file: str):
    """Analyze temperature curves to find degradation thresholds."""

    import glob

    curve_files = glob.glob(f"{results_dir}/temperature_curve_*.json")

    analysis = {
        "experiment_design": {
            "temperature_range": "0.1 to 2.0",
            "temperature_increment": 0.1,
            "questions_per_temperature": 200,
            "quantization_types": len(curve_files)
        },
        "degradation_thresholds": {},
        "curve_characteristics": {},
        "comparative_analysis": {}
    }

    all_curves = {}

    # Load all curve data
    for file_path in curve_files:
        quant_type = file_path.split('_')[-1].replace('.json', '')
        with open(file_path) as f:
            curve_data = json.load(f)
            all_curves[quant_type] = curve_data

    # Analyze each curve
    for quant_type, data in all_curves.items():
        temperatures = []
        truth_rates = []

        for temp_str, temp_data in data["results_by_temperature"].items():
            temperatures.append(float(temp_str))
            truth_rates.append(temp_data["truth_rate"])

        # Sort by temperature
        sorted_data = sorted(zip(temperatures, truth_rates))
        temps, rates = zip(*sorted_data)

        # Find degradation threshold (where truth rate drops below 80%)
        degradation_temp = None
        for temp, rate in sorted_data:
            if rate < 0.8:
                degradation_temp = temp
                break

        # Find peak performance temperature
        peak_temp = temps[rates.index(max(rates))]
        peak_rate = max(rates)

        # Calculate curve stability (standard deviation)
        rate_std = statistics.stdev(rates)

        analysis["degradation_thresholds"][quant_type] = {
            "degradation_temperature": degradation_temp,
            "peak_temperature": peak_temp,
            "peak_truth_rate": peak_rate,
            "curve_stability": rate_std,
            "temperature_truth_pairs": list(zip(temps, rates))
        }

        # Fit polynomial curve for smooth visualization
        if len(temps) >= 3:
            try:
                poly_coeffs = np.polyfit(temps, rates, 3)
                poly_func = np.poly1d(poly_coeffs)

                # Generate smooth curve
                smooth_temps = np.linspace(min(temps), max(temps), 100)
                smooth_rates = poly_func(smooth_temps)

                analysis["curve_characteristics"][quant_type] = {
                    "polynomial_coefficients": poly_coeffs.tolist(),
                    "smooth_curve": list(zip(smooth_temps.tolist(), smooth_rates.tolist()))
                }
            except:
                logger.warning(f"Could not fit polynomial for {quant_type}")

    # Comparative analysis
    if len(all_curves) >= 2:
        # Find most stable quantization type
        stability_ranking = sorted(
            analysis["degradation_thresholds"].items(),
            key=lambda x: x[1]["curve_stability"]
        )

        # Find most robust (highest degradation threshold)
        robustness_ranking = sorted(
            [(k, v) for k, v in analysis["degradation_thresholds"].items() if v["degradation_temperature"] is not None],
            key=lambda x: x[1]["degradation_temperature"],
            reverse=True
        )

        analysis["comparative_analysis"] = {
            "most_stable": stability_ranking[0][0] if stability_ranking else None,
            "most_robust": robustness_ranking[0][0] if robustness_ranking else None,
            "stability_ranking": [(name, data["curve_stability"]) for name, data in stability_ranking],
            "robustness_ranking": [(name, data["degradation_temperature"]) for name, data in robustness_ranking]
        }

    # Generate visualization
    if len(all_curves) >= 1:
        plt.figure(figsize=(12, 8))

        colors = ['blue', 'red', 'green', 'orange', 'purple']
        for i, (quant_type, data) in enumerate(analysis["degradation_thresholds"].items()):
            temps, rates = zip(*data["temperature_truth_pairs"])
            plt.plot(temps, rates, marker='o', label=quant_type, color=colors[i % len(colors)])

            # Mark degradation threshold
            if data["degradation_temperature"]:
                plt.axvline(x=data["degradation_temperature"], color=colors[i % len(colors)],
                           linestyle='--', alpha=0.5)

        plt.xlabel('Temperature')
        plt.ylabel('Truth Rate')
        plt.title('Temperature vs Truth Rate by Quantization Type')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.axhline(y=0.8, color='black', linestyle='-', alpha=0.3, label='80% Threshold')

        plt.savefig(f"{results_dir}/temperature_curves.png", dpi=300, bbox_inches='tight')
        plt.close()

        analysis["visualization_file"] = f"{results_dir}/temperature_curves.png"

    # Save analysis
    with open(output_file, 'w') as f:
        json.dump(analysis, f, indent=2)

    # Print summary
    print("PHASE 2 TEMPERATURE CURVE ANALYSIS:")
    print("=" * 50)

    for quant_type, data in analysis["degradation_thresholds"].items():
        degradation = data["degradation_temperature"] or "No degradation"
        peak = f"{data['peak_temperature']} ({data['peak_truth_rate']:.1%})"
        print(f"{quant_type}: Degrades at {degradation}, Peak at {peak}")

    if analysis["comparative_analysis"]:
        comp = analysis["comparative_analysis"]
        print(f"\nMost stable: {comp['most_stable']}")
        print(f"Most robust: {comp['most_robust']}")


def main():
    parser = argparse.ArgumentParser(description="Phase 2: Temperature Degradation Curves")
    parser.add_argument("--model", required=True, help="Model name")
    parser.add_argument("--quantization", required=True, help="Quantization type")
    parser.add_argument("--create-dataset", action="store_true", help="Create temperature sweep dataset")
    parser.add_argument("--run-sweep", action="store_true", help="Run temperature sweep")
    parser.add_argument("--analyze", action="store_true", help="Analyze temperature curves")
    parser.add_argument("--output-dir", default="phase2_results", help="Output directory")

    args = parser.parse_args()

    Path(args.output_dir).mkdir(exist_ok=True)

    if args.create_dataset:
        create_temperature_sweep_dataset(f"{args.output_dir}/temperature_sweep_dataset.jsonl")

    if args.run_sweep:
        run_temperature_sweep(
            f"{args.output_dir}/temperature_sweep_dataset.jsonl",
            args.model,
            args.quantization,
            args.output_dir
        )

    if args.analyze:
        analyze_temperature_curves(args.output_dir, f"{args.output_dir}/phase2_analysis.json")


if __name__ == "__main__":
    main()