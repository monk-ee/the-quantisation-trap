#!/usr/bin/env python3
"""
Hallucination evaluation with hyperparameter sweep.

Tests if quantization affects model sensitivity to sampling parameters:
- Temperature (creativity/randomness)
- Top-p (nucleus sampling)
- Top-k (vocabulary filtering)
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


class HyperparamHallucinationEvaluator:
    """Evaluates hallucination rates across quantization + hyperparameter combinations."""
    
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
            "unknown",
            "i cannot",
            "i can't",
            "not possible to",
            "impossible to",
            "no way to"
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
        
        # Load model with retries for GPU errors
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
        
        # Try loading with retries for hardware errors    
        max_retries = 5  # Increased retries for ECC errors
        for attempt in range(max_retries):
            try:
                logger.info(f"Model loading attempt {attempt + 1}/{max_retries}")
                
                # Aggressive GPU memory clearing before each attempt
                if torch.cuda.is_available() and attempt > 0:
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                    logger.info(f"Cleared GPU cache before attempt {attempt + 1}")
                
                self.model = AutoModelForCausalLM.from_pretrained(**model_kwargs)
                logger.info(f"Model loaded successfully on attempt {attempt + 1}")
                break
                
            except torch.cuda.OutOfMemoryError as e:
                logger.error(f"GPU OOM error on attempt {attempt + 1}: {e}")
                if attempt == max_retries - 1:
                    logger.error("OOM on final attempt, falling back to CPU")
                    # Fallback to CPU for OOM
                    model_kwargs["device_map"] = "cpu"
                    model_kwargs["torch_dtype"] = torch.float32
                    if "quantization_config" in model_kwargs:
                        del model_kwargs["quantization_config"]
                        logger.warning("Removing quantization for CPU fallback")
                    self.model = AutoModelForCausalLM.from_pretrained(**model_kwargs)
                    self.device = "cpu"
                    break
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                time.sleep(5)
                
            except Exception as e:
                error_msg = str(e).lower()
                
                # Check for various GPU hardware errors
                is_gpu_error = any(keyword in error_msg for keyword in [
                    "cuda error", "ecc error", "device-side", "uncorrectable", 
                    "acceleratorerror", "hardware error", "gpu error"
                ])
                
                if is_gpu_error:
                    logger.error(f"GPU hardware error on attempt {attempt + 1}: {e}")
                    
                    if attempt == max_retries - 1:
                        logger.error("All GPU attempts failed due to hardware errors, falling back to CPU")
                        # Final CPU fallback
                        model_kwargs["device_map"] = "cpu"
                        model_kwargs["torch_dtype"] = torch.float32
                        if "quantization_config" in model_kwargs:
                            del model_kwargs["quantization_config"]
                            logger.warning("Removing quantization for CPU fallback")
                        try:
                            self.model = AutoModelForCausalLM.from_pretrained(**model_kwargs)
                            self.device = "cpu"
                            logger.info("Successfully loaded model on CPU after GPU hardware failure")
                            break
                        except Exception as cpu_error:
                            logger.error(f"CPU fallback also failed: {cpu_error}")
                            raise RuntimeError(f"Both GPU and CPU loading failed. GPU error: {e}, CPU error: {cpu_error}")
                    
                    # Aggressive GPU reset between attempts
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                        # Try to reset GPU device
                        try:
                            torch.cuda.reset_peak_memory_stats()
                        except:
                            pass
                    
                    # Exponential backoff for hardware errors
                    sleep_time = min(30, 10 * (2 ** attempt))
                    logger.info(f"Waiting {sleep_time}s before retry due to hardware error")
                    time.sleep(sleep_time)
                    
                else:
                    logger.error(f"Non-recoverable error on attempt {attempt + 1}: {e}")
                    raise
        
        if not quantization_config and self.device == "cuda":
            self.model = self.model.half()
            
        logger.info(f"Model loaded successfully on {self.device}")
        
        # Log memory usage
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated() / 1024**3  # GB
            logger.info(f"GPU memory allocated: {memory_allocated:.2f} GB")
    
    def generate_response(self, prompt: str, temperature: float = 0.7, 
                         top_p: float = 0.9, top_k: int = 50, 
                         max_new_tokens: int = 100) -> Tuple[str, Dict[str, Any]]:
        """Generate response with specified hyperparameters and collect metrics."""
        start_time = time.time()
        
        # Tokenize input
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True)
        if self.device == "cuda":
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
        input_length = inputs["input_ids"].shape[1]
        
        # Generate response with hyperparameters - with GPU error handling
        try:
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,  # Enable sampling
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
        except Exception as e:
            error_msg = str(e).lower()
            if any(keyword in error_msg for keyword in ["cuda error", "ecc error", "device-side", "uncorrectable", "acceleratorerror"]):
                logger.error(f"GPU hardware error during generation: {e}")
                # Clear GPU cache and try once more
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                try:
                    with torch.no_grad():
                        outputs = self.model.generate(
                            **inputs,
                            max_new_tokens=max_new_tokens,
                            do_sample=True,
                            temperature=temperature,
                            top_p=top_p,
                            top_k=top_k,
                            pad_token_id=self.tokenizer.pad_token_id,
                            eos_token_id=self.tokenizer.eos_token_id,
                        )
                    logger.info("Generation succeeded after GPU cache clear")
                except Exception as retry_error:
                    logger.error(f"Generation failed after retry: {retry_error}")
                    raise RuntimeError(f"GPU generation failed: {retry_error}")
            else:
                logger.error(f"Non-GPU error during generation: {e}")
                raise
        
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
            "tokens_per_second": len(generated_tokens) / generation_time if generation_time > 0 else 0,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k
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

            # Improved semantic matching for key concepts
            # Extract key concepts from expected answer
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
    
    def evaluate_prompts_with_hyperparams(self, prompts: List[Dict[str, Any]], 
                                        hyperparam_configs: List[Dict[str, Any]], 
                                        output_file: str) -> None:
        """Evaluate all prompts across different hyperparameter configurations."""
        logger.info(f"Starting evaluation of {len(prompts)} prompts across {len(hyperparam_configs)} hyperparameter configs")
        
        all_results = []
        
        for config_idx, config in enumerate(hyperparam_configs):
            config_name = f"temp_{config['temperature']}_topp_{config['top_p']}_topk_{config['top_k']}"
            logger.info(f"Testing config {config_idx+1}/{len(hyperparam_configs)}: {config_name}")
            
            config_results = []
            
            for i, prompt_data in enumerate(prompts):
                prompt_id = prompt_data["id"]
                prompt = prompt_data["prompt"]
                question_type = prompt_data["type"]
                expected = prompt_data["expected"]
                
                logger.info(f"  Evaluating prompt {i+1}/{len(prompts)}: {prompt_id}")
                
                # Generate response with current hyperparameters
                try:
                    response, metrics = self.generate_response(
                        prompt,
                        temperature=config["temperature"],
                        top_p=config["top_p"],
                        top_k=config["top_k"]
                    )
                except Exception as e:
                    logger.error(f"Failed to generate response for {prompt_id}: {e}")
                    response = "[GENERATION_ERROR]"
                    metrics = {
                        "input_tokens": 0,
                        "output_tokens": 0,
                        "total_tokens": 0,
                        "generation_time_ms": 0,
                        "tokens_per_second": 0,
                        "temperature": config["temperature"],
                        "top_p": config["top_p"],
                        "top_k": config["top_k"]
                    }
                
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
                    "config_name": config_name,
                    "hyperparams": config,
                    "metrics": metrics,
                    **classification
                }
                
                config_results.append(result)
                
                # Log progress
                logger.info(f"    Response: {response[:80]}...")
                logger.info(f"    Classification: {classification}")
            
            all_results.extend(config_results)
            
        # Save results
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump({
                "quantization_type": self.quantization_type,
                "model_name": self.model_name,
                "total_prompts": len(prompts),
                "total_configs": len(hyperparam_configs),
                "hyperparam_configs": hyperparam_configs,
                "results": all_results
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
    parser = argparse.ArgumentParser(description="Evaluate hallucination rates across quantization + hyperparameters")
    
    parser.add_argument("--model", required=True, help="Hugging Face model name")
    parser.add_argument("--prompts", required=True, help="Path to prompts JSONL file")
    parser.add_argument("--quantization", required=True, 
                       choices=["none", "int8", "nf4", "fp4_double_quant"],
                       help="Quantization type")
    parser.add_argument("--output", required=True, help="Output JSON file path")
    parser.add_argument("--device", default="auto", help="Device to use (auto, cuda, cpu)")
    parser.add_argument("--max-tokens", type=int, default=80, help="Max new tokens to generate")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        
    # Define hyperparameter sweep configurations (optimized for speed)
    hyperparam_configs = [
        # Conservative (low temperature, focused sampling)
        {"temperature": 0.3, "top_p": 0.8, "top_k": 20},

        # Balanced
        {"temperature": 0.7, "top_p": 0.9, "top_k": 50},

        # Creative (high temperature, diverse sampling)
        {"temperature": 1.2, "top_p": 0.95, "top_k": 100},
    ]
    
    logger.info(f"Testing {len(hyperparam_configs)} hyperparameter configurations")
    
    # Load prompts
    prompts = load_prompts(args.prompts)
    logger.info(f"Loaded {len(prompts)} prompts")
    
    # Initialize evaluator
    evaluator = HyperparamHallucinationEvaluator(
        model_name=args.model,
        quantization_type=args.quantization,
        device=args.device
    )
    
    # Load model
    evaluator.load_model()
    
    # Run evaluation
    evaluator.evaluate_prompts_with_hyperparams(prompts, hyperparam_configs, args.output)
    
    logger.info("Hyperparameter sweep evaluation completed successfully")


if __name__ == "__main__":
    main()