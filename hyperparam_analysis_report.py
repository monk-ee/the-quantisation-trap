#!/usr/bin/env python3
"""
Hyperparameter × Quantization Analysis with Visualizations
Generates comprehensive graphs and analysis from sweep results.
"""

import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Any, Tuple
import statistics
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

# Set style
plt.style.use('default')
sns.set_palette("husl")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class HyperparamAnalyzer:
    """Analyzes hyperparameter sweep results and generates visualizations."""
    
    def __init__(self):
        self.results_by_quantization = {}
        self.all_results = []
        self.question_types = ["known", "rare", "fabricated", "trap", "plausible"]
        
    def load_results(self, results_files: List[str]) -> None:
        """Load results from multiple hyperparameter sweep files."""
        for file_path in results_files:
            logger.info(f"Loading results from {file_path}")
            
            with open(file_path, 'r') as f:
                data = json.load(f)
                
            quantization_type = data["quantization_type"]
            self.results_by_quantization[quantization_type] = data
            
            # Add quantization type to each result for easier processing
            for result in data["results"]:
                result["quantization_type"] = quantization_type
                self.all_results.append(result)
                
        logger.info(f"Loaded {len(self.all_results)} total results across {len(self.results_by_quantization)} quantization types")
    
    def extract_structured_data(self) -> Dict[str, Any]:
        """Extract structured data for analysis."""
        data = {
            'quantization_types': list(self.results_by_quantization.keys()),
            'temperatures': [],
            'top_p_values': [],
            'hyperparameter_configs': [],
            'results_matrix': defaultdict(lambda: defaultdict(list)),
            'question_performance': defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        }
        
        # Extract unique hyperparameter values
        temps = set()
        top_ps = set()
        configs = set()
        
        for result in self.all_results:
            temp = result['hyperparams']['temperature']
            top_p = result['hyperparams']['top_p']
            config_name = result['config_name']
            
            temps.add(temp)
            top_ps.add(top_p)
            configs.add(config_name)
            
            # Store results by quantization and config
            quant = result['quantization_type']
            data['results_matrix'][quant][config_name].append(result)
            
            # Store results by question type
            q_type = result['type']
            data['question_performance'][quant][q_type][config_name].append(result)
        
        data['temperatures'] = sorted(temps)
        data['top_p_values'] = sorted(top_ps)
        data['hyperparameter_configs'] = sorted(configs)
        
        return data
    
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
    
    def calculate_temperature_sensitivity(self, data: Dict[str, Any]) -> Dict[str, float]:
        """Calculate how sensitive each quantization type is to temperature changes."""
        sensitivity_scores = {}
        
        for quant_type in data['quantization_types']:
            temp_hallucination_rates = []
            
            # Get hallucination rates for each temperature
            for config in data['hyperparameter_configs']:
                if config in data['results_matrix'][quant_type]:
                    results = data['results_matrix'][quant_type][config]
                    rates = self.calculate_hallucination_rates(results)
                    temp_hallucination_rates.append(rates['hallucination_rate'])
            
            # Calculate variance as sensitivity measure
            if len(temp_hallucination_rates) > 1:
                sensitivity_scores[quant_type] = np.var(temp_hallucination_rates)
            else:
                sensitivity_scores[quant_type] = 0.0
                
        return sensitivity_scores
    
    def create_hallucination_heatmap(self, data: Dict[str, Any], output_dir: str) -> str:
        """Create heatmap of hallucination rates across quantization and temperature."""
        try:
            # Prepare data for heatmap
            quantization_types = data['quantization_types']
            configs = data['hyperparameter_configs']
            
            if not quantization_types or not configs:
                logger.warning("No quantization types or configs found for heatmap")
                return ""
            
            heatmap_data = np.zeros((len(quantization_types), len(configs)))
            
            for i, quant_type in enumerate(quantization_types):
                for j, config in enumerate(configs):
                    if config in data['results_matrix'][quant_type]:
                        results = data['results_matrix'][quant_type][config]
                        if results:  # Check if results exist
                            rates = self.calculate_hallucination_rates(results)
                            heatmap_data[i, j] = rates['hallucination_rate'] * 100
            
            # Create heatmap
            plt.figure(figsize=(14, 8))
            sns.heatmap(
                heatmap_data,
                xticklabels=[c.replace('temp_', 'T').replace('_topp_', ' P').replace('_topk_', ' K') for c in configs],
                yticklabels=quantization_types,
                annot=True,
                fmt='.1f',
                cmap='Reds',
                cbar_kws={'label': 'Hallucination Rate (%)'}
            )
            plt.title('Hallucination Rates: Quantization × Hyperparameters', fontsize=16, fontweight='bold')
            plt.xlabel('Hyperparameter Configuration', fontsize=12)
            plt.ylabel('Quantization Type', fontsize=12)
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            
            output_path = f"{output_dir}/hallucination_heatmap.png"
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            return output_path
            
        except Exception as e:
            logger.error(f"Failed to create hallucination heatmap: {e}")
            plt.close()  # Ensure plot is closed even on error
            return ""
    
    def create_temperature_sensitivity_plot(self, data: Dict[str, Any], output_dir: str) -> str:
        """Create line plot showing temperature sensitivity."""
        try:
            plt.figure(figsize=(12, 8))
            
            if not data['quantization_types']:
                logger.warning("No quantization types found for temperature sensitivity plot")
                plt.close()
                return ""
            
            colors = sns.color_palette("husl", len(data['quantization_types']))
            
            for i, quant_type in enumerate(data['quantization_types']):
                temperatures = []
                hallucination_rates = []
                
                # Extract temperature and hallucination rate pairs
                for config in data['hyperparameter_configs']:
                    if config in data['results_matrix'][quant_type]:
                        try:
                            # Parse temperature from config name
                            temp_str = config.split('temp_')[1].split('_')[0]
                            temp = float(temp_str)
                            
                            results = data['results_matrix'][quant_type][config]
                            if results:  # Check if results exist
                                rates = self.calculate_hallucination_rates(results)
                                temperatures.append(temp)
                                hallucination_rates.append(rates['hallucination_rate'] * 100)
                        except (IndexError, ValueError) as e:
                            logger.warning(f"Failed to parse temperature from config {config}: {e}")
                            continue
                
                # Sort by temperature
                sorted_pairs = sorted(zip(temperatures, hallucination_rates))
                temps, rates = zip(*sorted_pairs) if sorted_pairs else ([], [])
                
                if temps:  # Only plot if we have data
                    plt.plot(temps, rates, marker='o', linewidth=2, markersize=8, 
                            label=f'{quant_type}', color=colors[i])
            
            plt.title('Temperature Sensitivity: Hallucination Rate vs Temperature', fontsize=16, fontweight='bold')
            plt.xlabel('Temperature', fontsize=12)
            plt.ylabel('Hallucination Rate (%)', fontsize=12)
            plt.legend(title='Quantization Type', bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            output_path = f"{output_dir}/temperature_sensitivity.png"
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            return output_path
            
        except Exception as e:
            logger.error(f"Failed to create temperature sensitivity plot: {e}")
            plt.close()
            return ""
    
    def create_question_type_comparison(self, data: Dict[str, Any], output_dir: str) -> str:
        """Create comparison of performance across question types."""
        try:
            fig, axes = plt.subplots(2, 3, figsize=(18, 10))
            axes = axes.flatten()
            
            for i, q_type in enumerate(self.question_types):
                if i >= len(axes):
                    break
                    
                ax = axes[i]
                
                quant_types = []
                refusal_rates = []
                hallucination_rates = []
                
                for quant_type in data['quantization_types']:
                    # Aggregate all results for this question type and quantization
                    all_results = []
                    for config in data['hyperparameter_configs']:
                        if config in data['question_performance'][quant_type][q_type]:
                            all_results.extend(data['question_performance'][quant_type][q_type][config])
                    
                    if all_results:
                        rates = self.calculate_hallucination_rates(all_results)
                        quant_types.append(quant_type)
                        refusal_rates.append(rates['refusal_rate'] * 100)
                        hallucination_rates.append(rates['hallucination_rate'] * 100)
                
                if quant_types:  # Only create plot if we have data
                    x = np.arange(len(quant_types))
                    width = 0.35
                    
                    ax.bar(x - width/2, refusal_rates, width, label='Refusal %', alpha=0.8)
                    ax.bar(x + width/2, hallucination_rates, width, label='Hallucination %', alpha=0.8)
                    
                    ax.set_title(f'{q_type.capitalize()} Questions', fontweight='bold')
                    ax.set_xlabel('Quantization Type')
                    ax.set_ylabel('Percentage (%)')
                    ax.set_xticks(x)
                    ax.set_xticklabels(quant_types, rotation=45, ha='right')
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                else:
                    ax.text(0.5, 0.5, f'No data for {q_type}', ha='center', va='center', transform=ax.transAxes)
                    ax.set_title(f'{q_type.capitalize()} Questions', fontweight='bold')
            
            # Remove empty subplots
            for j in range(len(self.question_types), len(axes)):
                axes[j].remove()
            
            plt.suptitle('Question Type Performance Across Quantization Levels', fontsize=16, fontweight='bold')
            plt.tight_layout()
            
            output_path = f"{output_dir}/question_type_comparison.png"
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            return output_path
            
        except Exception as e:
            logger.error(f"Failed to create question type comparison: {e}")
            plt.close()
            return ""
    
    def create_response_length_analysis(self, data: Dict[str, Any], output_dir: str) -> str:
        """Analyze response length patterns across configurations."""
        try:
            plt.figure(figsize=(12, 8))
            
            # Create box plot of response lengths by quantization type
            quant_data = []
            labels = []
            
            for quant_type in data['quantization_types']:
                lengths = []
                for config in data['hyperparameter_configs']:
                    if config in data['results_matrix'][quant_type]:
                        for result in data['results_matrix'][quant_type][config]:
                            if 'metrics' in result and 'output_tokens' in result['metrics']:
                                lengths.append(result['metrics']['output_tokens'])
                
                if lengths:
                    quant_data.append(lengths)
                    labels.append(quant_type)
            
            if quant_data:  # Only create plot if we have data
                plt.boxplot(quant_data, tick_labels=labels)
                plt.title('Response Length Distribution by Quantization Type', fontsize=16, fontweight='bold')
                plt.xlabel('Quantization Type', fontsize=12)
                plt.ylabel('Output Tokens', fontsize=12)
                plt.xticks(rotation=45, ha='right')
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                
                output_path = f"{output_dir}/response_length_analysis.png"
                plt.savefig(output_path, dpi=300, bbox_inches='tight')
                plt.close()
                
                return output_path
            else:
                logger.warning("No response length data found")
                plt.close()
                return ""
                
        except Exception as e:
            logger.error(f"Failed to create response length analysis: {e}")
            plt.close()
            return ""
    
    def find_dramatic_examples(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Find the most dramatic response differences across hyperparameters."""
        dramatic_examples = []
        
        # Group results by question ID
        questions = defaultdict(lambda: defaultdict(list))
        for result in self.all_results:
            questions[result['id']][result['quantization_type']].append(result)
        
        for question_id, quant_results in questions.items():
            for quant_type, results in quant_results.items():
                if len(results) > 1:
                    # Find most conservative and most creative responses
                    conservative = min(results, key=lambda x: x['hyperparams']['temperature'])
                    creative = max(results, key=lambda x: x['hyperparams']['temperature'])
                    
                    if conservative['hyperparams']['temperature'] != creative['hyperparams']['temperature']:
                        dramatic_examples.append({
                            'question_id': question_id,
                            'question': conservative['prompt'],
                            'quantization': quant_type,
                            'conservative_temp': conservative['hyperparams']['temperature'],
                            'conservative_response': conservative['response'],
                            'conservative_is_refusal': conservative['is_refusal'],
                            'creative_temp': creative['hyperparams']['temperature'],
                            'creative_response': creative['response'],
                            'creative_is_refusal': creative['is_refusal'],
                            'behavior_change': conservative['is_refusal'] != creative['is_refusal']
                        })
        
        # Sort by most dramatic changes
        dramatic_examples.sort(key=lambda x: abs(x['creative_temp'] - x['conservative_temp']), reverse=True)
        
        return dramatic_examples[:10]  # Top 10 most dramatic
    
    def generate_report(self, output_file: str) -> None:
        """Generate comprehensive analysis report with visualizations."""
        logger.info("Generating hyperparameter analysis report...")
        
        output_dir = Path(output_file).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Extract structured data
        data = self.extract_structured_data()
        
        # Calculate sensitivity scores
        temperature_sensitivity = self.calculate_temperature_sensitivity(data)
        
        # Find dramatic examples
        dramatic_examples = self.find_dramatic_examples(data)
        
        # Generate visualizations (filter out failed ones)
        viz_paths = []
        
        heatmap_path = self.create_hallucination_heatmap(data, str(output_dir))
        if heatmap_path:
            viz_paths.append(heatmap_path)
            
        temp_path = self.create_temperature_sensitivity_plot(data, str(output_dir))
        if temp_path:
            viz_paths.append(temp_path)
            
        question_path = self.create_question_type_comparison(data, str(output_dir))
        if question_path:
            viz_paths.append(question_path)
            
        length_path = self.create_response_length_analysis(data, str(output_dir))
        if length_path:
            viz_paths.append(length_path)
        
        # Generate analysis report
        analysis = {
            "experiment_summary": {
                "model_name": list(self.results_by_quantization.values())[0]["model_name"],
                "quantization_types": data['quantization_types'],
                "total_hyperparam_configs": len(data['hyperparameter_configs']),
                "temperature_range": f"{min(data['temperatures']):.1f} - {max(data['temperatures']):.1f}",
                "question_categories": self.question_types,
                "total_responses": len(self.all_results)
            },
            "sensitivity_analysis": {
                "temperature_sensitivity_scores": temperature_sensitivity,
                "temperature_sensitivity_ranking": [
                    {"rank": i+1, "quantization": quant, "sensitivity_score": score}
                    for i, (quant, score) in enumerate(sorted(temperature_sensitivity.items(), 
                                                            key=lambda x: x[1], reverse=True))
                ]
            },
            "dramatic_examples": dramatic_examples,
            "visualization_files": viz_paths,
            "key_findings": self.generate_key_findings(data, temperature_sensitivity, dramatic_examples)
        }
        
        # Save report
        with open(output_file, 'w') as f:
            json.dump(analysis, f, indent=2)
            
        logger.info(f"Analysis report saved to {output_file}")
        logger.info(f"Generated {len(viz_paths)} visualization files")
        
        # Print summary to console
        self.print_summary(analysis)
    
    def generate_key_findings(self, data: Dict[str, Any], sensitivity: Dict[str, float], 
                            examples: List[Dict[str, Any]]) -> List[str]:
        """Generate key findings from the analysis."""
        findings = []
        
        # Sensitivity findings
        most_sensitive = max(sensitivity, key=sensitivity.get)
        least_sensitive = min(sensitivity, key=sensitivity.get)
        
        if sensitivity[most_sensitive] > sensitivity[least_sensitive] * 2:
            findings.append(f"{most_sensitive} quantization is most sensitive to temperature changes")
            findings.append(f"{least_sensitive} quantization is least affected by hyperparameter changes")
        else:
            findings.append("All quantization types show similar sensitivity to hyperparameter changes")
        
        # Behavior change findings
        behavior_changes = sum(1 for ex in examples if ex['behavior_change'])
        if behavior_changes > 0:
            findings.append(f"{behavior_changes} questions showed dramatic refusal behavior changes with temperature")
        
        # Overall patterns
        total_responses = len(self.all_results)
        findings.append(f"Tested {total_responses} total model responses across hyperparameter configurations")
        
        return findings
    
    def print_summary(self, analysis: Dict[str, Any]) -> None:
        """Print summary to console."""
        print("\n" + "="*60)
        print("🎯 HYPERPARAMETER × QUANTIZATION ANALYSIS COMPLETE")
        print("="*60)
        
        summary = analysis['experiment_summary']
        print(f"Model: {summary['model_name']}")
        print(f"Quantization Types: {', '.join(summary['quantization_types'])}")
        print(f"Temperature Range: {summary['temperature_range']}")
        print(f"Total Responses: {summary['total_responses']}")
        
        print("\n📊 TEMPERATURE SENSITIVITY RANKING:")
        print("-" * 40)
        for rank_data in analysis['sensitivity_analysis']['temperature_sensitivity_ranking']:
            print(f"{rank_data['rank']}. {rank_data['quantization']:<15} - Score: {rank_data['sensitivity_score']:.4f}")
        
        print(f"\n🎭 DRAMATIC EXAMPLES FOUND: {len(analysis['dramatic_examples'])}")
        
        if analysis['key_findings']:
            print("\n🔍 KEY FINDINGS:")
            print("-" * 20)
            for i, finding in enumerate(analysis['key_findings'], 1):
                print(f"{i}. {finding}")
        
        print(f"\n📈 VISUALIZATIONS GENERATED: {len(analysis['visualization_files'])}")
        for viz_path in analysis['visualization_files']:
            print(f"  • {Path(viz_path).name}")
        
        print("\n" + "="*60)


def main():
    parser = argparse.ArgumentParser(description="Analyze hyperparameter sweep results with visualizations")
    
    parser.add_argument("--results", nargs="+", required=True,
                       help="JSON result files from hyperparameter evaluation runs")
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
    
    # Generate analysis with visualizations
    analyzer = HyperparamAnalyzer()
    analyzer.load_results(args.results)
    analyzer.generate_report(args.output)
    
    logger.info("Hyperparameter analysis with visualizations completed successfully")
    return 0


if __name__ == "__main__":
    exit(main())