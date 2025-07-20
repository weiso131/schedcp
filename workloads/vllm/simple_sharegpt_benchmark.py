#!/usr/bin/env python3
"""
Simplified ShareGPT benchmark using vLLM's official benchmark_throughput.py approach.
This avoids the complexities of our custom implementation.
"""

import argparse
import json
import time
import sys
import os
from pathlib import Path

# Add parent directory to path to import scheduler runner
parent_dir = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(parent_dir))

try:
    from scheduler.scheduler_runner import SchedulerRunner
except ImportError:
    print("Warning: Could not import scheduler_runner. Running without scheduler switching.")
    SchedulerRunner = None

from vllm import LLM, SamplingParams
from vllm.transformers_utils.tokenizer import get_tokenizer


def load_sharegpt_dataset(dataset_path, num_prompts=100):
    """Load ShareGPT dataset and return prompts."""
    with open(dataset_path, 'r') as f:
        dataset = json.load(f)
    
    prompts = []
    for i, item in enumerate(dataset):
        if i >= num_prompts:
            break
        
        if "conversations" in item and len(item["conversations"]) > 0:
            # Get the first user message
            for conv in item["conversations"]:
                if conv.get("from") == "human":
                    prompts.append(conv["value"])
                    break
    
    return prompts


def run_benchmark(model_name, prompts, max_tokens=128, max_model_len=2048):
    """Run vLLM benchmark with given prompts."""
    print(f"\nInitializing vLLM with model: {model_name}")
    print(f"Max model length: {max_model_len}")
    print(f"Max tokens per request: {max_tokens}")
    
    # Initialize LLM
    llm = LLM(
        model=model_name,
        max_model_len=max_model_len,
        trust_remote_code=True,
        download_dir="/tmp/vllm_models",
        gpu_memory_utilization=0.85,
    )
    
    # Set up sampling parameters
    sampling_params = SamplingParams(
        temperature=1.0,
        top_p=1.0,
        max_tokens=max_tokens,
        ignore_eos=True,
    )
    
    # Filter prompts by length to avoid errors
    tokenizer = get_tokenizer(model_name, trust_remote_code=True)
    filtered_prompts = []
    
    for prompt in prompts:
        tokens = tokenizer.encode(prompt)
        if len(tokens) + max_tokens <= max_model_len:
            filtered_prompts.append(prompt)
        else:
            print(f"Skipping prompt with {len(tokens)} tokens (exceeds limit)")
    
    print(f"\nRunning benchmark with {len(filtered_prompts)} prompts...")
    
    # Run generation
    start_time = time.time()
    outputs = llm.generate(filtered_prompts, sampling_params, use_tqdm=True)
    end_time = time.time()
    
    # Calculate metrics
    elapsed_time = end_time - start_time
    total_tokens = sum(len(output.outputs[0].token_ids) for output in outputs)
    
    metrics = {
        "num_prompts": len(filtered_prompts),
        "elapsed_time": elapsed_time,
        "prompts_per_second": len(filtered_prompts) / elapsed_time,
        "total_generated_tokens": total_tokens,
        "tokens_per_second": total_tokens / elapsed_time,
        "avg_tokens_per_prompt": total_tokens / len(filtered_prompts) if filtered_prompts else 0,
    }
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Simple ShareGPT vLLM benchmark")
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.2-3B",
                        help="Model to benchmark")
    parser.add_argument("--dataset", type=str, default="datasets/ShareGPT_V3_unfiltered_cleaned_split.json",
                        help="Path to ShareGPT dataset")
    parser.add_argument("--num-prompts", type=int, default=100,
                        help="Number of prompts to test")
    parser.add_argument("--max-tokens", type=int, default=128,
                        help="Maximum tokens to generate per prompt")
    parser.add_argument("--max-model-len", type=int, default=2048,
                        help="Maximum model sequence length")
    parser.add_argument("--test-schedulers", action="store_true",
                        help="Test with different schedulers")
    
    args = parser.parse_args()
    
    # Load dataset
    print(f"Loading {args.num_prompts} prompts from {args.dataset}...")
    prompts = load_sharegpt_dataset(args.dataset, args.num_prompts)
    print(f"Loaded {len(prompts)} prompts")
    
    if args.test_schedulers and SchedulerRunner:
        # Test with different schedulers
        runner = SchedulerRunner()
        schedulers_to_test = ["default", "scx_lavd", "scx_rusty", "scx_bpfland"]
        
        results = {}
        for scheduler in schedulers_to_test:
            print(f"\n{'='*60}")
            print(f"Testing scheduler: {scheduler}")
            print(f"{'='*60}")
            
            if scheduler != "default":
                if not runner.setup_scheduler(scheduler):
                    print(f"Failed to setup {scheduler}, skipping...")
                    continue
            
            try:
                metrics = run_benchmark(args.model, prompts, args.max_tokens, args.max_model_len)
                results[scheduler] = metrics
                
                print(f"\nResults for {scheduler}:")
                print(f"  Prompts/second: {metrics['prompts_per_second']:.2f}")
                print(f"  Tokens/second: {metrics['tokens_per_second']:.2f}")
                print(f"  Avg tokens/prompt: {metrics['avg_tokens_per_prompt']:.2f}")
                
            except Exception as e:
                print(f"Error with {scheduler}: {e}")
                results[scheduler] = {"error": str(e)}
            
            finally:
                if scheduler != "default":
                    runner.stop_scheduler()
        
        # Print summary
        print(f"\n{'='*60}")
        print("BENCHMARK SUMMARY")
        print(f"{'='*60}")
        for scheduler, metrics in results.items():
            if "error" not in metrics:
                print(f"{scheduler}: {metrics['tokens_per_second']:.2f} tokens/sec")
            else:
                print(f"{scheduler}: ERROR - {metrics['error']}")
    
    else:
        # Just run with current scheduler
        metrics = run_benchmark(args.model, prompts, args.max_tokens, args.max_model_len)
        
        print("\nBenchmark Results:")
        print(f"  Number of prompts: {metrics['num_prompts']}")
        print(f"  Total time: {metrics['elapsed_time']:.2f} seconds")
        print(f"  Prompts/second: {metrics['prompts_per_second']:.2f}")
        print(f"  Tokens/second: {metrics['tokens_per_second']:.2f}")
        print(f"  Avg tokens/prompt: {metrics['avg_tokens_per_prompt']:.2f}")


if __name__ == "__main__":
    main()