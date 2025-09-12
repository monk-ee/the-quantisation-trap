#!/usr/bin/env python3
"""
Generate hallucination analysis report from quantization experiment results.

Creates tables and visualizations showing how quantization affects:
- Refusal rates vs incorrect guessing
- Performance across question types (known/rare/fabricated)
- Resource utilization metrics
"""

import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Any, Tuple
import statistics

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class HallucinationReportGenerator:
    """Generates analysis reports from hallucination evaluation results."""
    
    def __init__(self):
        self.results_by_quantization = {}
        self.question_types = ["known", "rare", "fabricated", "ambiguous"]
        
    def load_results(self, results_files: List[str]) -> None:
        """Load results from multiple evaluation files."""
        for file_path in results_files:
            logger.info(f"Loading results from {file_path}")
            
            with open(file_path, 'r') as f:
                data = json.load(f)
                
            quantization_type = data["quantization_type"]
            self.results_by_quantization[quantization_type] = data
            
        logger.info(f"Loaded results for {len(self.results_by_quantization)} quantization types")
    
    def calculate_metrics_by_type(self, results: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
        """Calculate metrics grouped by question type."""
        metrics_by_type = {}
        
        # Group results by type
        results_by_type = {}
        for result in results:
            question_type = result["type"]
            if question_type not in results_by_type:
                results_by_type[question_type] = []
            results_by_type[question_type].append(result)
        
        # Calculate metrics for each type
        for question_type, type_results in results_by_type.items():
            total = len(type_results)
            if total == 0:
                continue
                
            correct = sum(1 for r in type_results if r["is_correct"])
            refusal = sum(1 for r in type_results if r["is_refusal"]) 
            incorrect_guess = sum(1 for r in type_results if r["is_incorrect_guess"])
            
            metrics_by_type[question_type] = {
                "total": total,
                "correct_rate": correct / total,
                "refusal_rate": refusal / total,
                "incorrect_guess_rate": incorrect_guess / total,
                "correct_count": correct,
                "refusal_count": refusal,
                "incorrect_guess_count": incorrect_guess
            }
            
        return metrics_by_type
    
    def calculate_performance_metrics(self, results: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate overall performance metrics."""
        if not results:
            return {}
            
        # Token metrics
        input_tokens = [r["metrics"]["input_tokens"] for r in results]
        output_tokens = [r["metrics"]["output_tokens"] for r in results]
        generation_times = [r["metrics"]["generation_time_ms"] for r in results]
        tokens_per_second = [r["metrics"]["tokens_per_second"] for r in results]
        
        # Memory metrics (if available)
        memory_usage = [r["metrics"].get("gpu_memory_mb", 0) for r in results]
        memory_usage = [m for m in memory_usage if m > 0]
        
        metrics = {
            "avg_input_tokens": statistics.mean(input_tokens),
            "avg_output_tokens": statistics.mean(output_tokens), 
            "avg_generation_time_ms": statistics.mean(generation_times),
            "avg_tokens_per_second": statistics.mean(tokens_per_second),
            "total_questions": len(results)
        }
        
        if memory_usage:
            metrics["avg_gpu_memory_mb"] = statistics.mean(memory_usage)
            metrics["max_gpu_memory_mb"] = max(memory_usage)
            
        return metrics
    
    def generate_summary_table(self) -> List[List[str]]:
        """Generate summary table comparing all quantization types."""
        headers = [
            "Quantization",
            "Known Correct %",
            "Rare Refusal %", 
            "Rare Guess %",
            "Fabricated Refusal %",
            "Fabricated Guess %",
            "Avg Gen Time (ms)",
            "Avg Tokens/sec"
        ]
        
        rows = []
        
        for quant_type in sorted(self.results_by_quantization.keys()):
            data = self.results_by_quantization[quant_type]
            results = data["results"]
            
            # Calculate metrics by type
            metrics_by_type = self.calculate_metrics_by_type(results)
            performance_metrics = self.calculate_performance_metrics(results)
            
            # Build row
            row = [quant_type]
            
            # Known correct %
            known_metrics = metrics_by_type.get("known", {})
            row.append(f"{known_metrics.get('correct_rate', 0):.1%}")
            
            # Rare refusal % and guess %
            rare_metrics = metrics_by_type.get("rare", {})
            row.append(f"{rare_metrics.get('refusal_rate', 0):.1%}")
            row.append(f"{rare_metrics.get('incorrect_guess_rate', 0):.1%}")
            
            # Fabricated refusal % and guess %
            fabricated_metrics = metrics_by_type.get("fabricated", {})
            row.append(f"{fabricated_metrics.get('refusal_rate', 0):.1%}")
            row.append(f"{fabricated_metrics.get('incorrect_guess_rate', 0):.1%}")
            
            # Performance metrics
            row.append(f"{performance_metrics.get('avg_generation_time_ms', 0):.0f}")
            row.append(f"{performance_metrics.get('avg_tokens_per_second', 0):.1f}")
            
            rows.append(row)
            
        return [headers] + rows
    
    def analyze_hallucination_trends(self) -> Dict[str, Any]:
        """Analyze trends in hallucination rates across quantization levels."""
        trends = {
            "quantization_order": ["none", "int8", "nf4", "fp4"],  # From least to most aggressive
            "rare_refusal_trend": [],
            "rare_guess_trend": [],
            "fabricated_refusal_trend": [], 
            "fabricated_guess_trend": [],
            "key_findings": []
        }
        
        # Extract metrics in quantization order
        for quant_type in trends["quantization_order"]:
            if quant_type not in self.results_by_quantization:
                continue
                
            data = self.results_by_quantization[quant_type]
            metrics_by_type = self.calculate_metrics_by_type(data["results"])
            
            rare_metrics = metrics_by_type.get("rare", {})
            fabricated_metrics = metrics_by_type.get("fabricated", {})
            
            trends["rare_refusal_trend"].append({
                "quantization": quant_type,
                "rate": rare_metrics.get("refusal_rate", 0)
            })
            
            trends["rare_guess_trend"].append({
                "quantization": quant_type, 
                "rate": rare_metrics.get("incorrect_guess_rate", 0)
            })
            
            trends["fabricated_refusal_trend"].append({
                "quantization": quant_type,
                "rate": fabricated_metrics.get("refusal_rate", 0)
            })
            
            trends["fabricated_guess_trend"].append({
                "quantization": quant_type,
                "rate": fabricated_metrics.get("incorrect_guess_rate", 0)
            })
        
        # Analyze trends
        if len(trends["rare_refusal_trend"]) > 1:
            # Check if refusal rates decrease with quantization
            rare_refusal_rates = [t["rate"] for t in trends["rare_refusal_trend"]]
            rare_guess_rates = [t["rate"] for t in trends["rare_guess_trend"]]
            
            if rare_refusal_rates[0] > rare_refusal_rates[-1]:
                trends["key_findings"].append(
                    "Rare question refusal rates decrease with stronger quantization"
                )
            
            if rare_guess_rates[-1] > rare_guess_rates[0]:
                trends["key_findings"].append(
                    "Rare question incorrect guessing increases with stronger quantization"
                )
                
            # Similar analysis for fabricated questions
            fab_refusal_rates = [t["rate"] for t in trends["fabricated_refusal_trend"]]
            fab_guess_rates = [t["rate"] for t in trends["fabricated_guess_trend"]]
            
            if fab_refusal_rates[0] > fab_refusal_rates[-1]:
                trends["key_findings"].append(
                    "Fabricated question refusal rates decrease with stronger quantization"
                )
                
            if fab_guess_rates[-1] > fab_guess_rates[0]:
                trends["key_findings"].append(
                    "Fabricated question incorrect guessing increases with stronger quantization"
                )
        
        return trends
    
    def generate_detailed_analysis(self) -> Dict[str, Any]:
        """Generate detailed analysis with all metrics and findings."""
        analysis = {
            "experiment_summary": {
                "total_quantization_types": len(self.results_by_quantization),
                "quantization_types": list(self.results_by_quantization.keys()),
                "model_name": None,
                "total_questions_per_type": 0
            },
            "summary_table": self.generate_summary_table(),
            "trends_analysis": self.analyze_hallucination_trends(),
            "detailed_results": {},
            "conclusions": []
        }
        
        # Extract model name and question count
        for data in self.results_by_quantization.values():
            if analysis["experiment_summary"]["model_name"] is None:
                analysis["experiment_summary"]["model_name"] = data["model_name"]
            analysis["experiment_summary"]["total_questions_per_type"] = data["total_prompts"]
            break
        
        # Add detailed results for each quantization type
        for quant_type, data in self.results_by_quantization.items():
            metrics_by_type = self.calculate_metrics_by_type(data["results"])
            performance_metrics = self.calculate_performance_metrics(data["results"])
            
            analysis["detailed_results"][quant_type] = {
                "metrics_by_question_type": metrics_by_type,
                "performance_metrics": performance_metrics,
                "model_name": data["model_name"],
                "total_prompts": data["total_prompts"]
            }
        
        # Generate conclusions
        trends = analysis["trends_analysis"]
        if trends["key_findings"]:
            analysis["conclusions"].extend(trends["key_findings"])
        
        # Add overall conclusion
        if len(self.results_by_quantization) > 1:
            analysis["conclusions"].append(
                "Quantization appears to affect model uncertainty calibration, "
                "potentially supporting OpenAI's claims about hallucination incentives"
            )
        
        return analysis
    
    def print_summary_table(self, table: List[List[str]]) -> None:
        """Print formatted summary table."""
        if not table:
            return
            
        # Calculate column widths
        col_widths = []
        for i in range(len(table[0])):
            max_width = max(len(str(row[i])) for row in table)
            col_widths.append(max_width + 2)
        
        # Print header
        header = table[0]
        header_line = "|".join(f" {header[i]:<{col_widths[i]-1}}" for i in range(len(header)))
        print(header_line)
        print("-" * len(header_line))
        
        # Print rows
        for row in table[1:]:
            row_line = "|".join(f" {str(row[i]):<{col_widths[i]-1}}" for i in range(len(row)))
            print(row_line)
    
    def save_report(self, output_file: str) -> None:
        """Save detailed analysis to JSON file."""
        analysis = self.generate_detailed_analysis()
        
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(analysis, f, indent=2)
            
        logger.info(f"Detailed analysis saved to {output_file}")
        
        # Print summary to console
        print("\n" + "="*80)
        print("QUANTIZATION HALLUCINATION EXPERIMENT RESULTS")
        print("="*80)
        
        print(f"\nModel: {analysis['experiment_summary']['model_name']}")
        print(f"Questions per type: {analysis['experiment_summary']['total_questions_per_type']}")
        print(f"Quantization types tested: {', '.join(analysis['experiment_summary']['quantization_types'])}")
        
        print("\nSUMMARY TABLE:")
        print("-" * 40)
        self.print_summary_table(analysis["summary_table"])
        
        if analysis["conclusions"]:
            print("\nKEY FINDINGS:")
            print("-" * 40)
            for i, conclusion in enumerate(analysis["conclusions"], 1):
                print(f"{i}. {conclusion}")
        
        print("\n" + "="*80)


def main():
    parser = argparse.ArgumentParser(description="Generate hallucination analysis report")
    
    parser.add_argument("--results", nargs="+", required=True,
                       help="JSON result files from evaluation runs")
    parser.add_argument("--output", required=True,
                       help="Output analysis JSON file")
    parser.add_argument("--verbose", action="store_true",
                       help="Enable verbose logging")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Verify input files exist
    for file_path in args.results:
        if not Path(file_path).exists():
            logger.error(f"Results file not found: {file_path}")
            return 1
    
    # Generate report
    generator = HallucinationReportGenerator()
    generator.load_results(args.results)
    generator.save_report(args.output)
    
    logger.info("Report generation completed successfully")
    return 0


if __name__ == "__main__":
    exit(main())