#!/usr/bin/env python3
"""
Hallucination evaluation script for quantization experiments.

Tests if stronger quantization increases hallucination rate by making models
more likely to guess incorrectly rather than admit uncertainty.

Based on OpenAI's paper claims about training/evaluation reward structures.
"""

import json
import time
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import re

import torch
import transformers
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    BitsAndBytesConfig
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class HallucinationEvaluator:
    """Evaluates hallucination rates across different quantization levels."""
    
    def __init__(self, model_name: str, quantization_type: str = "none", device: str = "auto"):
        self.model_name = model_name
        self.quantization_type = quantization_type
        self.device = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
        
        self.tokenizer = None
        self.model = None
        
        # Uncertainty markers for refusal detection
        self.uncertainty_markers = [
            "i don't know",
            "i do not know", 
            "i'm not sure",
            "i am not sure",
            "uncertain",
            "unclear",
            "cannot determine",
            "can't determine",
            "insufficient information",
            "not enough information",
            "unable to answer",
            "cannot answer",
            "can't answer",
            "no information",
            "unknown"
        ]
        
    def _setup_quantization_config(self) -> Optional[BitsAndBytesConfig]:
        """Setup quantization configuration based on type."""
        if self.quantization_type == "none":
            return None
        elif self.quantization_type == "int8":
            return BitsAndBytesConfig(load_in_8bit=True)
        elif self.quantization_type == "nf4":
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16
            )
        elif self.quantization_type == "fp4":
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="fp4", 
                bnb_4bit_compute_dtype=torch.float16
            )
        elif self.quantization_type == "nf4_double_quant":
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,  # Even more aggressive
                bnb_4bit_compute_dtype=torch.float16
            )
        elif self.quantization_type == "fp4_double_quant":
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="fp4",
                bnb_4bit_use_double_quant=True,  # EXTREME quantization
                bnb_4bit_compute_dtype=torch.float16
            )
        else:
            logger.warning(f"Unknown quantization type: {self.quantization_type}")
            return None
    
    def load_model(self) -> None:
        """Load the model with specified quantization."""
        logger.info(f"Loading model {self.model_name} with quantization: {self.quantization_type}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            padding_side="left"
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        # Setup quantization
        quantization_config = self._setup_quantization_config()
        
        # Load model
        model_kwargs = {
            "pretrained_model_name_or_path": self.model_name,
            "trust_remote_code": True,
            "torch_dtype": torch.float16 if self.device == "cuda" else torch.float32,
        }
        
        if quantization_config:
            model_kwargs["quantization_config"] = quantization_config
            model_kwargs["device_map"] = "auto"
        else:
            model_kwargs["device_map"] = self.device
            
        self.model = AutoModelForCausalLM.from_pretrained(**model_kwargs)
        
        if not quantization_config and self.device == "cuda":
            self.model = self.model.half()
            
        logger.info(f"Model loaded successfully on {self.device}")
        
        # Log memory usage
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated() / 1024**3  # GB
            logger.info(f"GPU memory allocated: {memory_allocated:.2f} GB")
    
    def generate_response(self, prompt: str, max_new_tokens: int = 100) -> Tuple[str, Dict[str, Any]]:
        """Generate response to prompt and collect metrics."""
        start_time = time.time()
        
        # Tokenize input
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True)
        if self.device == "cuda":
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
        input_length = inputs["input_ids"].shape[1]
        
        # Generate response
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,  # Deterministic for consistency
                temperature=0.1,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        
        # Extract generated tokens (excluding input)
        generated_tokens = outputs[0][input_length:]
        response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        generation_time = time.time() - start_time
        
        # Collect metrics
        metrics = {
            "input_tokens": input_length,
            "output_tokens": len(generated_tokens),
            "total_tokens": len(outputs[0]),
            "generation_time_ms": generation_time * 1000,
            "tokens_per_second": len(generated_tokens) / generation_time if generation_time > 0 else 0
        }
        
        if torch.cuda.is_available():
            metrics["gpu_memory_mb"] = torch.cuda.memory_allocated() / 1024**2
            
        return response.strip(), metrics
    
    def classify_response(self, response: str, expected: str, question_type: str) -> Dict[str, bool]:
        """Classify response as refusal, correct, or incorrect guess."""
        response_lower = response.lower()
        
        # Check for refusal/uncertainty markers
        is_refusal = any(marker in response_lower for marker in self.uncertainty_markers)
        
        # Check correctness for "known" questions
        is_correct = False
        if question_type == "known":
            expected_lower = expected.lower()
            # Simple containment check - could be made more sophisticated
            is_correct = expected_lower in response_lower
        
        # Classify as incorrect guess if not refusal and not correct
        is_incorrect_guess = not is_refusal and not is_correct
        
        return {
            "is_refusal": is_refusal,
            "is_correct": is_correct, 
            "is_incorrect_guess": is_incorrect_guess
        }
    
    def evaluate_prompts(self, prompts: List[Dict[str, Any]], output_file: str) -> None:
        """Evaluate all prompts and save results."""
        logger.info(f"Starting evaluation of {len(prompts)} prompts")
        
        results = []
        
        for i, prompt_data in enumerate(prompts):
            prompt_id = prompt_data["id"]
            prompt = prompt_data["prompt"]
            question_type = prompt_data["type"]
            expected = prompt_data["expected"]
            
            logger.info(f"Evaluating prompt {i+1}/{len(prompts)}: {prompt_id}")
            
            # Generate response
            response, metrics = self.generate_response(prompt)
            
            # Classify response
            classification = self.classify_response(response, expected, question_type)
            
            # Store result
            result = {
                "id": prompt_id,
                "prompt": prompt,
                "response": response,
                "type": question_type,
                "expected": expected,
                "quantization": self.quantization_type,
                "model": self.model_name,
                "metrics": metrics,
                **classification
            }
            
            results.append(result)
            
            # Log progress
            logger.info(f"Response: {response[:100]}...")
            logger.info(f"Classification: {classification}")
            
        # Save results
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump({
                "quantization_type": self.quantization_type,
                "model_name": self.model_name,
                "total_prompts": len(prompts),
                "results": results
            }, f, indent=2)
            
        logger.info(f"Results saved to {output_file}")


def load_prompts(prompts_file: str) -> List[Dict[str, Any]]:
    """Load prompts from JSONL file."""
    prompts = []
    with open(prompts_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                prompts.append(json.loads(line))
    return prompts


def main():
    parser = argparse.ArgumentParser(description="Evaluate hallucination rates across quantization levels")
    
    parser.add_argument("--model", required=True, help="Hugging Face model name")
    parser.add_argument("--prompts", required=True, help="Path to prompts JSONL file")
    parser.add_argument("--quantization", required=True, 
                       choices=["none", "int8", "nf4", "fp4", "nf4_double_quant", "fp4_double_quant"],
                       help="Quantization type")
    parser.add_argument("--output", required=True, help="Output JSON file path")
    parser.add_argument("--device", default="auto", help="Device to use (auto, cuda, cpu)")
    parser.add_argument("--max-tokens", type=int, default=100, help="Max new tokens to generate")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        
    # Load prompts
    prompts = load_prompts(args.prompts)
    logger.info(f"Loaded {len(prompts)} prompts")
    
    # Initialize evaluator
    evaluator = HallucinationEvaluator(
        model_name=args.model,
        quantization_type=args.quantization,
        device=args.device
    )
    
    # Load model
    evaluator.load_model()
    
    # Run evaluation
    evaluator.evaluate_prompts(prompts, args.output)
    
    logger.info("Evaluation completed successfully")


if __name__ == "__main__":
    main()