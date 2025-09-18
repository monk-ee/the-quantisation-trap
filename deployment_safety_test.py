#!/usr/bin/env python3
"""
Deployment Safety Test: Quantify truth/cost/reliability trade-offs with hard metrics.

Tests specific deployment scenarios with measurable outcomes:
1. Memory usage vs factual accuracy
2. Temperature thresholds for truth degradation
3. Cost reduction vs reliability maintenance
4. Production-ready configuration boundaries
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Tuple
from collections import defaultdict
import statistics

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DeploymentSafetyAnalyzer:
    """Analyzes deployment safety metrics from experimental results."""

    def __init__(self):
        self.results = []
        self.baseline_metrics = {}

    def load_results(self, results_files: List[str]) -> None:
        """Load experimental results."""
        for file_path in results_files:
            with open(file_path, 'r') as f:
                data = json.load(f)

            for result in data["results"]:
                # Extract deployment-relevant metrics
                self.results.append({
                    'quantization': result['quantization'],
                    'temperature': result['hyperparams']['temperature'],
                    'top_p': result['hyperparams']['top_p'],
                    'factually_correct': result.get('factually_correct', result['is_correct']),
                    'is_refusal': result['is_refusal'],
                    'safety_classification': result.get('safety_classification', 'unknown'),
                    'response_length': len(result['response']),
                    'question_type': result['type'],
                    'gpu_memory_mb': result['metrics'].get('gpu_memory_mb', 0),
                    'generation_time_ms': result['metrics']['generation_time_ms'],
                    'prompt_type': 'prompted' if 'prompted' in file_path else 'baseline'
                })

    def measure_truth_degradation_thresholds(self) -> Dict[str, Any]:
        """Find exact temperature thresholds where truth breaks down."""

        # Group by quantization type and temperature
        truth_by_config = defaultdict(list)

        for result in self.results:
            if result['question_type'] == 'known':  # Focus on factual questions
                key = (result['quantization'], result['temperature'], result['prompt_type'])
                truth_by_config[key].append(result['factually_correct'])

        # Calculate truth rates for each configuration
        truth_rates = {}
        for (quant, temp, prompt_type), truth_values in truth_by_config.items():
            truth_rate = sum(truth_values) / len(truth_values)
            truth_rates[(quant, temp, prompt_type)] = {
                'truth_rate': truth_rate,
                'sample_size': len(truth_values),
                'config': f"{quant}_{prompt_type}_temp{temp}"
            }

        # Find degradation thresholds
        thresholds = {}
        for quant in ['none', 'int8', 'nf4', 'fp4_double_quant']:
            for prompt_type in ['baseline', 'prompted']:
                temps = [0.3, 0.7, 1.2]
                quant_thresholds = []

                for temp in temps:
                    key = (quant, temp, prompt_type)
                    if key in truth_rates:
                        rate = truth_rates[key]['truth_rate']
                        quant_thresholds.append((temp, rate))

                # Find where truth rate drops below 80%
                degradation_temp = None
                for temp, rate in quant_thresholds:
                    if rate < 0.8:
                        degradation_temp = temp
                        break

                thresholds[f"{quant}_{prompt_type}"] = {
                    'truth_breakdown_temperature': degradation_temp,
                    'temperature_truth_curve': quant_thresholds
                }

        return {
            'truth_rates_by_config': truth_rates,
            'degradation_thresholds': thresholds
        }

    def measure_memory_cost_efficiency(self) -> Dict[str, Any]:
        """Calculate memory reduction vs truth preservation efficiency."""

        # Baseline: none quantization, conservative settings
        baseline_memory = None
        baseline_truth_rate = None

        memory_efficiency = {}

        for result in self.results:
            if (result['quantization'] == 'none' and
                result['temperature'] == 0.3 and
                result['prompt_type'] == 'baseline' and
                result['question_type'] == 'known'):

                if baseline_memory is None:
                    baseline_memory = result['gpu_memory_mb']

        # Calculate efficiency for each quantization type
        for quant in ['none', 'int8', 'nf4', 'fp4_double_quant']:
            quant_results = [r for r in self.results
                           if (r['quantization'] == quant and
                               r['temperature'] == 0.3 and  # Conservative temp
                               r['question_type'] == 'known')]

            if quant_results:
                avg_memory = statistics.mean([r['gpu_memory_mb'] for r in quant_results if r['gpu_memory_mb'] > 0])
                truth_rate = sum([r['factually_correct'] for r in quant_results]) / len(quant_results)

                memory_reduction = (baseline_memory - avg_memory) / baseline_memory if baseline_memory else 0

                memory_efficiency[quant] = {
                    'memory_mb': avg_memory,
                    'memory_reduction_pct': memory_reduction * 100,
                    'truth_rate': truth_rate,
                    'efficiency_score': truth_rate / (avg_memory / baseline_memory) if baseline_memory else 0,
                    'sample_size': len(quant_results)
                }

        return memory_efficiency

    def measure_prompt_engineering_effectiveness(self) -> Dict[str, Any]:
        """Quantify prompt engineering impact on safety metrics."""

        effectiveness = {}

        for quant in ['none', 'int8', 'nf4', 'fp4_double_quant']:
            baseline_results = [r for r in self.results
                              if (r['quantization'] == quant and
                                  r['prompt_type'] == 'baseline' and
                                  r['question_type'] == 'known')]

            prompted_results = [r for r in self.results
                              if (r['quantization'] == quant and
                                  r['prompt_type'] == 'prompted' and
                                  r['question_type'] == 'known')]

            if baseline_results and prompted_results:
                baseline_truth = sum([r['factually_correct'] for r in baseline_results]) / len(baseline_results)
                prompted_truth = sum([r['factually_correct'] for r in prompted_results]) / len(prompted_results)

                baseline_refusal = sum([r['is_refusal'] for r in baseline_results]) / len(baseline_results)
                prompted_refusal = sum([r['is_refusal'] for r in prompted_results]) / len(prompted_results)

                effectiveness[quant] = {
                    'baseline_truth_rate': baseline_truth,
                    'prompted_truth_rate': prompted_truth,
                    'truth_improvement': prompted_truth - baseline_truth,
                    'baseline_refusal_rate': baseline_refusal,
                    'prompted_refusal_rate': prompted_refusal,
                    'refusal_improvement': prompted_refusal - baseline_refusal,
                    'safety_improvement': (prompted_refusal - baseline_refusal) + (prompted_truth - baseline_truth)
                }

        return effectiveness

    def find_production_safe_configs(self) -> Dict[str, Any]:
        """Identify configurations safe for production deployment."""

        # Define production safety criteria
        MIN_TRUTH_RATE = 0.95  # 95% factual accuracy
        MAX_WRONG_ANSWERS = 0.05  # 5% or less factually wrong responses

        safe_configs = []
        risky_configs = []

        # Group results by configuration
        config_performance = defaultdict(list)

        for result in self.results:
            if result['question_type'] == 'known':
                config_key = (
                    result['quantization'],
                    result['temperature'],
                    result['prompt_type']
                )
                config_performance[config_key].append(result)

        # Evaluate each configuration
        for config_key, config_results in config_performance.items():
            quant, temp, prompt_type = config_key

            truth_rate = sum([r['factually_correct'] for r in config_results]) / len(config_results)
            wrong_rate = sum([not r['factually_correct'] and not r['is_refusal'] for r in config_results]) / len(config_results)
            refusal_rate = sum([r['is_refusal'] for r in config_results]) / len(config_results)

            avg_memory = statistics.mean([r['gpu_memory_mb'] for r in config_results if r['gpu_memory_mb'] > 0])
            avg_latency = statistics.mean([r['generation_time_ms'] for r in config_results])

            config_analysis = {
                'quantization': quant,
                'temperature': temp,
                'prompt_type': prompt_type,
                'truth_rate': truth_rate,
                'wrong_answer_rate': wrong_rate,
                'refusal_rate': refusal_rate,
                'memory_mb': avg_memory,
                'latency_ms': avg_latency,
                'sample_size': len(config_results),
                'production_safe': truth_rate >= MIN_TRUTH_RATE and wrong_rate <= MAX_WRONG_ANSWERS
            }

            if config_analysis['production_safe']:
                safe_configs.append(config_analysis)
            else:
                risky_configs.append(config_analysis)

        # Sort by efficiency (truth rate / memory usage)
        safe_configs.sort(key=lambda x: x['truth_rate'] / (x['memory_mb'] / 1000), reverse=True)
        risky_configs.sort(key=lambda x: x['wrong_answer_rate'], reverse=True)

        return {
            'production_safe_configs': safe_configs,
            'risky_configs': risky_configs,
            'safety_criteria': {
                'min_truth_rate': MIN_TRUTH_RATE,
                'max_wrong_rate': MAX_WRONG_ANSWERS
            }
        }

    def generate_deployment_report(self, output_file: str) -> None:
        """Generate comprehensive deployment safety analysis."""

        logger.info("Analyzing deployment safety metrics...")

        # Run all analyses
        truth_analysis = self.measure_truth_degradation_thresholds()
        memory_analysis = self.measure_memory_cost_efficiency()
        prompt_analysis = self.measure_prompt_engineering_effectiveness()
        safety_analysis = self.find_production_safe_configs()

        # Compile report
        report = {
            'deployment_safety_analysis': {
                'total_configurations_tested': len(set([(r['quantization'], r['temperature'], r['prompt_type']) for r in self.results])),
                'total_responses_analyzed': len(self.results),
                'factual_questions_analyzed': len([r for r in self.results if r['question_type'] == 'known'])
            },
            'truth_degradation_thresholds': truth_analysis,
            'memory_cost_efficiency': memory_analysis,
            'prompt_engineering_effectiveness': prompt_analysis,
            'production_deployment_guidance': safety_analysis
        }

        # Save report
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)

        # Print key findings
        self.print_deployment_insights(report)

        logger.info(f"Deployment safety analysis saved to {output_file}")

    def print_deployment_insights(self, report: Dict[str, Any]) -> None:
        """Print actionable deployment insights."""

        print("\n" + "="*80)
        print("🏭 DEPLOYMENT SAFETY ANALYSIS")
        print("="*80)

        # Truth degradation insights
        print("\n📊 TRUTH DEGRADATION THRESHOLDS:")
        thresholds = report['truth_degradation_thresholds']['degradation_thresholds']
        for config, data in thresholds.items():
            breakdown_temp = data['truth_breakdown_temperature']
            if breakdown_temp:
                print(f"   {config}: Truth breaks at temperature {breakdown_temp}")
            else:
                print(f"   {config}: Maintains truth across all tested temperatures")

        # Memory efficiency insights
        print("\n💰 MEMORY COST EFFICIENCY:")
        memory_data = report['memory_cost_efficiency']
        for quant, data in memory_data.items():
            print(f"   {quant}: {data['memory_reduction_pct']:.1f}% memory reduction, {data['truth_rate']:.1%} truth rate")

        # Production safe configurations
        print("\n✅ PRODUCTION-SAFE CONFIGURATIONS:")
        safe_configs = report['production_deployment_guidance']['production_safe_configs']
        for config in safe_configs[:3]:  # Top 3
            print(f"   {config['quantization']} + temp {config['temperature']} + {config['prompt_type']}: "
                  f"{config['truth_rate']:.1%} truth, {config['memory_mb']:.0f}MB memory")

        # Risky configurations
        print("\n⚠️  RISKY CONFIGURATIONS (avoid in production):")
        risky_configs = report['production_deployment_guidance']['risky_configs']
        for config in risky_configs[:3]:  # Top 3 most dangerous
            print(f"   {config['quantization']} + temp {config['temperature']} + {config['prompt_type']}: "
                  f"{config['wrong_answer_rate']:.1%} wrong answers")


def main():
    import glob

    # Load experimental results
    result_files = glob.glob("/Users/lyndonswan/PycharmProjects/TheQuantisationTrap/quantization-hallucination-results/hyperparam_results/hyperparam_*.json")
    result_files = [f for f in result_files if "analysis" not in f]

    if not result_files:
        logger.error("No result files found")
        return

    # Run deployment safety analysis
    analyzer = DeploymentSafetyAnalyzer()
    analyzer.load_results(result_files)
    analyzer.generate_deployment_report("deployment_safety_analysis.json")


if __name__ == "__main__":
    main()