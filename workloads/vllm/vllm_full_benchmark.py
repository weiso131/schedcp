#!/usr/bin/env python3
"""
Full vLLM benchmark with 1000 prompts and proper length handling.
Automatically filters prompts to fit within model limits.
"""

import time
import json
import argparse
import sys
from pathlib import Path
from vllm import LLM, SamplingParams
from vllm.transformers_utils.tokenizer import get_tokenizer

# Add parent directory to path for scheduler runner
parent_dir = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(parent_dir))

try:
    from scheduler.scheduler_runner import SchedulerRunner
except ImportError:
    print("Warning: Could not import scheduler_runner. Running without scheduler switching.")
    SchedulerRunner = None


def load_and_filter_prompts(dataset_path, model_name, num_prompts, max_model_len, max_output_tokens):
    """Load ShareGPT dataset and filter prompts by length."""
    print(f"Loading dataset from {dataset_path}...")
    with open(dataset_path, 'r') as f:
        dataset = json.load(f)
    
    # Initialize tokenizer for length checking
    tokenizer = get_tokenizer(model_name, trust_remote_code=True)
    
    valid_prompts = []
    skipped_count = 0
    
    # Calculate max allowed prompt length
    max_prompt_tokens = max_model_len - max_output_tokens - 50  # 50 token buffer
    
    for i, item in enumerate(dataset):
        if len(valid_prompts) >= num_prompts:
            break
            
        if "conversations" in item and len(item["conversations"]) > 0:
            # Get the first user message
            prompt_text = None
            for conv in item["conversations"]:
                if conv.get("from") == "human":
                    prompt_text = conv["value"]
                    break
            
            if prompt_text:
                # Check token length
                tokens = tokenizer.encode(prompt_text)
                if len(tokens) <= max_prompt_tokens:
                    valid_prompts.append(prompt_text)
                else:
                    skipped_count += 1
    
    print(f"Loaded {len(valid_prompts)} valid prompts")
    print(f"Skipped {skipped_count} prompts that were too long")
    
    return valid_prompts


def run_benchmark(model_name, prompts, max_output_tokens=256, max_model_len=4096):
    """Run vLLM benchmark with given prompts."""
    print(f"\nInitializing vLLM:")
    print(f"  Model: {model_name}")
    print(f"  Max model length: {max_model_len}")
    print(f"  Max output tokens: {max_output_tokens}")
    print(f"  Number of prompts: {len(prompts)}")
    
    # Initialize LLM
    llm = LLM(
        model=model_name,
        max_model_len=max_model_len,
        trust_remote_code=True,
        download_dir="/tmp/vllm_models",
        gpu_memory_utilization=0.90,
        enable_prefix_caching=True,
    )
    
    # Set up sampling parameters
    sampling_params = SamplingParams(
        temperature=1.0,
        top_p=1.0,
        max_tokens=max_output_tokens,
        ignore_eos=True,
    )
    
    print("\nRunning benchmark...")
    start_time = time.time()
    
    # Run generation with progress bar
    outputs = llm.generate(prompts, sampling_params, use_tqdm=True)
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    # Calculate metrics
    total_output_tokens = sum(len(output.outputs[0].token_ids) for output in outputs)
    total_input_tokens = sum(len(output.prompt_token_ids) for output in outputs)
    total_tokens = total_input_tokens + total_output_tokens
    
    metrics = {
        "model": model_name,
        "num_prompts": len(outputs),
        "elapsed_time": elapsed_time,
        "prompts_per_second": len(outputs) / elapsed_time,
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "total_tokens": total_tokens,
        "output_tokens_per_second": total_output_tokens / elapsed_time,
        "total_tokens_per_second": total_tokens / elapsed_time,
        "avg_input_tokens": total_input_tokens / len(outputs),
        "avg_output_tokens": total_output_tokens / len(outputs),
    }
    
    return metrics


def print_results(metrics):
    """Print benchmark results in a formatted way."""
    print(f"\n{'='*60}")
    print(f"BENCHMARK RESULTS")
    print(f"{'='*60}")
    print(f"Model: {metrics['model']}")
    print(f"Total prompts: {metrics['num_prompts']}")
    print(f"Total time: {metrics['elapsed_time']:.2f} seconds")
    print(f"\nThroughput:")
    print(f"  Prompts/second: {metrics['prompts_per_second']:.2f}")
    print(f"  Output tokens/second: {metrics['output_tokens_per_second']:.2f}")
    print(f"  Total tokens/second: {metrics['total_tokens_per_second']:.2f}")
    print(f"\nToken statistics:")
    print(f"  Total input tokens: {metrics['total_input_tokens']:,}")
    print(f"  Total output tokens: {metrics['total_output_tokens']:,}")
    print(f"  Avg input tokens/prompt: {metrics['avg_input_tokens']:.1f}")
    print(f"  Avg output tokens/prompt: {metrics['avg_output_tokens']:.1f}")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description="Full vLLM ShareGPT benchmark")
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.2-3B",
                        help="Model to benchmark")
    parser.add_argument("--dataset", type=str, default="datasets/ShareGPT_V3_unfiltered_cleaned_split.json",
                        help="Path to ShareGPT dataset")
    parser.add_argument("--num-prompts", type=int, default=1000,
                        help="Number of prompts to test")
    parser.add_argument("--max-output-tokens", type=int, default=256,
                        help="Maximum output tokens per prompt")
    parser.add_argument("--max-model-len", type=int, default=4096,
                        help="Maximum model sequence length")
    parser.add_argument("--test-scheduler", type=str, default=None,
                        help="Test with a specific scheduler (e.g., scx_lavd)")
    
    args = parser.parse_args()
    
    # Load and filter prompts
    prompts = load_and_filter_prompts(
        args.dataset,
        args.model,
        args.num_prompts,
        args.max_model_len,
        args.max_output_tokens
    )
    
    if len(prompts) < args.num_prompts:
        print(f"\nWarning: Only found {len(prompts)} valid prompts (requested {args.num_prompts})")
        response = input("Continue? (y/n): ")
        if response.lower() != 'y':
            return
    
    # Setup scheduler if requested
    runner = None
    scheduler_proc = None
    if args.test_scheduler and SchedulerRunner:
        runner = SchedulerRunner()
        print(f"\nSetting up scheduler: {args.test_scheduler}")
        try:
            scheduler_proc = runner.start_scheduler(args.test_scheduler)
            print(f"Successfully started {args.test_scheduler}")
        except Exception as e:
            print(f"Failed to start {args.test_scheduler}: {e}")
            return
    
    try:
        # Run benchmark
        metrics = run_benchmark(
            args.model,
            prompts,
            args.max_output_tokens,
            args.max_model_len
        )
        
        # Print results
        print_results(metrics)
        
        # Save results to file
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        results_file = f"results/benchmark_{timestamp}.json"
        with open(results_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"\nResults saved to: {results_file}")
        
    finally:
        # Clean up scheduler
        if args.test_scheduler and runner and scheduler_proc:
            print("\nStopping scheduler...")
            runner.stop_scheduler(args.test_scheduler, scheduler_proc)


if __name__ == "__main__":
    main()