#!/usr/bin/env python3
"""
Specialized analysis comparing prompt engineering vs quantization effects.
Generates direct comparisons between baseline and prompted experiments.
"""

import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Any, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PromptVsQuantizationAnalyzer:
    """Analyzes the effectiveness of prompt engineering vs quantization for reducing hallucination."""
    
    def __init__(self):
        self.baseline_results = {}  # quantization_type -> results
        self.prompted_results = {}  # quantization_type -> results
        
    def load_results(self, results_files: List[str]) -> None:
        """Load and categorize baseline vs prompted results."""
        for file_path in results_files:
            logger.info(f"Loading results from {file_path}")
            
            with open(file_path, 'r') as f:
                data = json.load(f)
                
            quantization_type = data["quantization_type"]
            file_basename = Path(file_path).stem
            
            if file_basename.endswith('_baseline'):
                self.baseline_results[quantization_type] = data
            elif file_basename.endswith('_prompted'):
                self.prompted_results[quantization_type] = data
            else:
                # Assume it's baseline if no suffix
                self.baseline_results[quantization_type] = data
                
        logger.info(f"Loaded baseline results for: {list(self.baseline_results.keys())}")
        logger.info(f"Loaded prompted results for: {list(self.prompted_results.keys())}")
    
    def calculate_hallucination_rates(self, results: List[Dict]) -> Dict[str, float]:
        """Calculate hallucination metrics for a set of results."""
        if not results:
            return {'refusal_rate': 0, 'hallucination_rate': 0, 'correct_rate': 0}
            
        total = len(results)
        refusal_count = sum(1 for r in results if r['is_refusal'])
        correct_count = sum(1 for r in results if r['is_correct'])
        hallucination_count = sum(1 for r in results if r['is_incorrect_guess'])
        
        return {
            'refusal_rate': refusal_count / total,
            'hallucination_rate': hallucination_count / total,
            'correct_rate': correct_count / total
        }
    
    def analyze_prompt_effectiveness(self) -> Dict[str, Any]:
        """Analyze how effective prompt engineering is compared to quantization."""
        analysis = {
            'quantization_comparison': {},
            'prompt_engineering_effectiveness': {},
            'key_findings': []
        }
        
        # Find quantization types that have both baseline and prompted results
        common_quantization_types = set(self.baseline_results.keys()) & set(self.prompted_results.keys())
        
        if not common_quantization_types:
            analysis['key_findings'].append("No overlapping quantization types found for comparison")
            return analysis
            
        for quant_type in sorted(common_quantization_types):
            baseline_data = self.baseline_results[quant_type]
            prompted_data = self.prompted_results[quant_type]
            
            # Calculate rates for baseline
            baseline_rates = self.calculate_hallucination_rates(baseline_data['results'])
            
            # Calculate rates for prompted
            prompted_rates = self.calculate_hallucination_rates(prompted_data['results'])
            
            # Calculate improvements
            refusal_improvement = prompted_rates['refusal_rate'] - baseline_rates['refusal_rate']
            hallucination_reduction = baseline_rates['hallucination_rate'] - prompted_rates['hallucination_rate']
            
            analysis['quantization_comparison'][quant_type] = {
                'baseline': baseline_rates,
                'prompted': prompted_rates,
                'improvements': {
                    'refusal_improvement': refusal_improvement,
                    'hallucination_reduction': hallucination_reduction,
                    'refusal_improvement_percent': (refusal_improvement / baseline_rates['refusal_rate'] * 100) if baseline_rates['refusal_rate'] > 0 else float('inf'),
                    'hallucination_reduction_percent': (hallucination_reduction / baseline_rates['hallucination_rate'] * 100) if baseline_rates['hallucination_rate'] > 0 else 0
                }
            }
        
        # Overall effectiveness analysis
        total_baseline_refusal = sum(analysis['quantization_comparison'][qt]['baseline']['refusal_rate'] for qt in common_quantization_types)
        total_prompted_refusal = sum(analysis['quantization_comparison'][qt]['prompted']['refusal_rate'] for qt in common_quantization_types)
        total_baseline_hallucination = sum(analysis['quantization_comparison'][qt]['baseline']['hallucination_rate'] for qt in common_quantization_types)
        total_prompted_hallucination = sum(analysis['quantization_comparison'][qt]['prompted']['hallucination_rate'] for qt in common_quantization_types)
        
        analysis['prompt_engineering_effectiveness'] = {
            'avg_baseline_refusal_rate': total_baseline_refusal / len(common_quantization_types),
            'avg_prompted_refusal_rate': total_prompted_refusal / len(common_quantization_types),
            'avg_baseline_hallucination_rate': total_baseline_hallucination / len(common_quantization_types),
            'avg_prompted_hallucination_rate': total_prompted_hallucination / len(common_quantization_types),
            'overall_refusal_improvement': (total_prompted_refusal - total_baseline_refusal) / len(common_quantization_types),
            'overall_hallucination_reduction': (total_baseline_hallucination - total_prompted_hallucination) / len(common_quantization_types)
        }
        
        # Generate key findings
        effectiveness = analysis['prompt_engineering_effectiveness']
        
        if effectiveness['overall_refusal_improvement'] > 0.1:  # 10% improvement
            analysis['key_findings'].append("Prompt engineering significantly increases refusal rates across all quantization levels")
            
        if effectiveness['overall_hallucination_reduction'] > 0.1:  # 10% reduction
            analysis['key_findings'].append("Prompt engineering significantly reduces hallucination across all quantization levels")
            
        # Check if prompting is more effective than quantization
        baseline_none_rates = analysis['quantization_comparison'].get('none', {}).get('baseline', {})
        baseline_extreme_rates = analysis['quantization_comparison'].get('fp4_double_quant', {}).get('baseline', {})
        prompted_none_rates = analysis['quantization_comparison'].get('none', {}).get('prompted', {})
        
        if baseline_none_rates and baseline_extreme_rates and prompted_none_rates:
            # Compare: extreme quantization vs prompt engineering (both at full precision)
            quantization_effect = baseline_extreme_rates['refusal_rate'] - baseline_none_rates['refusal_rate']
            prompt_effect = prompted_none_rates['refusal_rate'] - baseline_none_rates['refusal_rate']
            
            if prompt_effect > quantization_effect:
                analysis['key_findings'].append("Prompt engineering is more effective than extreme quantization for increasing uncertainty admission")
            elif quantization_effect > prompt_effect:
                analysis['key_findings'].append("Extreme quantization is more effective than prompt engineering for increasing uncertainty admission")
            else:
                analysis['key_findings'].append("Prompt engineering and quantization have similar effects on uncertainty admission")
                
        return analysis
    
    def generate_report(self, output_file: str) -> None:
        """Generate comprehensive prompt vs quantization analysis."""
        logger.info("Generating prompt engineering vs quantization analysis...")
        
        analysis = self.analyze_prompt_effectiveness()
        
        # Save detailed report
        report = {
            'experiment_type': 'prompt_engineering_vs_quantization',
            'quantization_types_tested': list(set(self.baseline_results.keys()) | set(self.prompted_results.keys())),
            'baseline_experiments': list(self.baseline_results.keys()),
            'prompted_experiments': list(self.prompted_results.keys()),
            'analysis': analysis,
            'summary': {
                'total_comparisons': len(analysis['quantization_comparison']),
                'prompt_engineering_winner': sum(1 for comp in analysis['quantization_comparison'].values() 
                                                if comp['improvements']['refusal_improvement'] > 0),
                'quantization_winner': sum(1 for comp in analysis['quantization_comparison'].values() 
                                          if comp['improvements']['refusal_improvement'] <= 0)
            }
        }
        
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
            
        logger.info(f"Prompt vs quantization analysis saved to {output_file}")
        
        # Print summary
        self.print_summary(report)
    
    def print_summary(self, report: Dict[str, Any]) -> None:
        """Print summary to console."""
        print("\n" + "="*70)
        print("🧠 PROMPT ENGINEERING vs QUANTIZATION ANALYSIS")
        print("="*70)
        
        analysis = report['analysis']
        summary = report['summary']
        
        print(f"Quantization Types Compared: {summary['total_comparisons']}")
        print(f"Prompt Engineering Wins: {summary['prompt_engineering_winner']}")
        print(f"Quantization Wins: {summary['quantization_winner']}")
        print()
        
        if 'prompt_engineering_effectiveness' in analysis:
            effectiveness = analysis['prompt_engineering_effectiveness']
            print("📊 OVERALL EFFECTIVENESS:")
            print(f"Average Baseline Refusal Rate: {effectiveness['avg_baseline_refusal_rate']:.1%}")
            print(f"Average Prompted Refusal Rate: {effectiveness['avg_prompted_refusal_rate']:.1%}")
            print(f"Average Baseline Hallucination Rate: {effectiveness['avg_baseline_hallucination_rate']:.1%}")
            print(f"Average Prompted Hallucination Rate: {effectiveness['avg_prompted_hallucination_rate']:.1%}")
            print()
            
            refusal_improvement = effectiveness['overall_refusal_improvement']
            hallucination_reduction = effectiveness['overall_hallucination_reduction']
            
            print(f"🎯 Refusal Rate Improvement: {refusal_improvement:+.1%}")
            print(f"🎯 Hallucination Rate Reduction: {hallucination_reduction:+.1%}")
            print()
        
        if analysis['key_findings']:
            print("🔍 KEY FINDINGS:")
            for i, finding in enumerate(analysis['key_findings'], 1):
                print(f"{i}. {finding}")
        
        print("\n" + "="*70)


def main():
    parser = argparse.ArgumentParser(description="Analyze prompt engineering vs quantization effectiveness")
    
    parser.add_argument("--results", nargs="+", required=True,
                       help="JSON result files from baseline and prompted experiments")
    parser.add_argument("--output", required=True,
                       help="Output analysis JSON file")
    parser.add_argument("--verbose", action="store_true",
                       help="Enable verbose logging")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Generate analysis
    analyzer = PromptVsQuantizationAnalyzer()
    analyzer.load_results(args.results)
    analyzer.generate_report(args.output)
    
    logger.info("Prompt vs quantization analysis completed successfully")
    return 0


if __name__ == "__main__":
    exit(main())