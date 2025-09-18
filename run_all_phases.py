#!/usr/bin/env python3
"""
Master runner for all three phases of the Quantization Trap experiment.
"""

import argparse
import subprocess
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_phase_1(model_name: str, output_base: str):
    """Run Phase 1: Statistical validation."""

    phase1_dir = f"{output_base}/phase1_results"
    Path(phase1_dir).mkdir(parents=True, exist_ok=True)

    logger.info("Starting Phase 1: Statistical Validation")

    # Create dataset
    subprocess.run([
        "python", "phase1_validation_experiment.py",
        "--model", model_name,
        "--create-dataset",
        "--output-dir", phase1_dir
    ], check=True)

    # Run experiments for each quantization type
    for quant_type in ["none", "int8", "nf4", "fp4_double_quant"]:
        logger.info(f"Running Phase 1 for {quant_type}")
        subprocess.run([
            "python", "phase1_validation_experiment.py",
            "--model", model_name,
            "--run-experiment",
            "--output-dir", phase1_dir
        ], check=True)

    # Analyze results
    subprocess.run([
        "python", "phase1_validation_experiment.py",
        "--analyze",
        "--output-dir", phase1_dir
    ], check=True)

    logger.info("Phase 1 completed")


def run_phase_2(model_name: str, output_base: str):
    """Run Phase 2: Temperature curve mapping."""

    phase2_dir = f"{output_base}/phase2_results"
    Path(phase2_dir).mkdir(parents=True, exist_ok=True)

    logger.info("Starting Phase 2: Temperature Curve Mapping")

    # Create dataset
    subprocess.run([
        "python", "phase2_temperature_curves.py",
        "--model", model_name,
        "--quantization", "none",  # Dummy value for dataset creation
        "--create-dataset",
        "--output-dir", phase2_dir
    ], check=True)

    # Run temperature sweeps for each quantization type
    for quant_type in ["none", "int8", "nf4", "fp4_double_quant"]:
        logger.info(f"Running Phase 2 temperature sweep for {quant_type}")
        subprocess.run([
            "python", "phase2_temperature_curves.py",
            "--model", model_name,
            "--quantization", quant_type,
            "--run-sweep",
            "--output-dir", phase2_dir
        ], check=True)

    # Analyze temperature curves
    subprocess.run([
        "python", "phase2_temperature_curves.py",
        "--model", model_name,
        "--quantization", "none",  # Dummy value for analysis
        "--analyze",
        "--output-dir", phase2_dir
    ], check=True)

    logger.info("Phase 2 completed")


def run_phase_3(model_name: str, output_base: str):
    """Run Phase 3: Economic validation."""

    phase3_dir = f"{output_base}/phase3_results"
    Path(phase3_dir).mkdir(parents=True, exist_ok=True)

    logger.info("Starting Phase 3: Economic Validation")

    # Create dataset
    subprocess.run([
        "python", "phase3_economic_validation.py",
        "--model", model_name,
        "--create-dataset",
        "--output-dir", phase3_dir
    ], check=True)

    # Run economic benchmark
    subprocess.run([
        "python", "phase3_economic_validation.py",
        "--model", model_name,
        "--run-benchmark",
        "--output-dir", phase3_dir
    ], check=True)

    logger.info("Phase 3 completed")


def generate_final_report(output_base: str):
    """Generate comprehensive final report."""

    import json

    final_report = {
        "experiment_summary": "The Quantization Trap: Three-Phase Analysis",
        "phases": {}
    }

    # Load Phase 1 results
    try:
        with open(f"{output_base}/phase1_results/phase1_analysis.json") as f:
            final_report["phases"]["phase1_statistical_validation"] = json.load(f)
    except:
        logger.warning("Phase 1 results not found")

    # Load Phase 2 results
    try:
        with open(f"{output_base}/phase2_results/phase2_analysis.json") as f:
            final_report["phases"]["phase2_temperature_curves"] = json.load(f)
    except:
        logger.warning("Phase 2 results not found")

    # Load Phase 3 results
    try:
        with open(f"{output_base}/phase3_results/economic_analysis.json") as f:
            final_report["phases"]["phase3_economic_validation"] = json.load(f)
    except:
        logger.warning("Phase 3 results not found")

    # Save final report
    with open(f"{output_base}/quantization_trap_final_report.json", 'w') as f:
        json.dump(final_report, f, indent=2)

    logger.info(f"Final report saved to {output_base}/quantization_trap_final_report.json")


def main():
    parser = argparse.ArgumentParser(description="Run all phases of the Quantization Trap experiment")
    parser.add_argument("--model", required=True, help="Model name")
    parser.add_argument("--phases", nargs="+", choices=["1", "2", "3", "all"], default=["all"], help="Which phases to run")
    parser.add_argument("--output-dir", default="quantization_trap_results", help="Base output directory")

    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    phases_to_run = args.phases
    if "all" in phases_to_run:
        phases_to_run = ["1", "2", "3"]

    if "1" in phases_to_run:
        run_phase_1(args.model, args.output_dir)

    if "2" in phases_to_run:
        run_phase_2(args.model, args.output_dir)

    if "3" in phases_to_run:
        run_phase_3(args.model, args.output_dir)

    # Generate final comprehensive report
    generate_final_report(args.output_dir)

    logger.info("All requested phases completed")


if __name__ == "__main__":
    main()