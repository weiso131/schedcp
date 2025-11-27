#!/usr/bin/env python3
"""
Minimal OpenAI client wrapper for llama.cpp server.
Makes llama.cpp server look like vLLM's output format.
"""

import json
import time
import sys
import argparse
import numpy as np
import asyncio
from concurrent.futures import ThreadPoolExecutor
from openai import OpenAI


def load_sharegpt_dataset(dataset_path, num_prompts):
    """Load ShareGPT dataset and extract prompts"""
    with open(dataset_path, 'r') as f:
        data = json.load(f)

    prompts = []
    for item in data[:num_prompts]:
        if 'conversations' in item:
            for conv in item['conversations']:
                if conv.get('from') == 'human':
                    prompts.append(conv['value'])
                    break

    return prompts[:num_prompts]


def process_single_request(client, prompt, idx, model, max_tokens):
    """Process a single request and return result"""
    try:
        request_start = time.time()
        first_token_time = None
        input_tokens = 0
        output_tokens = 0

        stream = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=0.7,
            stream=True,
            stream_options={"include_usage": True}
        )

        for chunk in stream:
            # Track first token time
            if first_token_time is None and chunk.choices and len(chunk.choices) > 0:
                delta = chunk.choices[0].delta
                has_content = (delta.content or
                             getattr(delta, 'reasoning_content', None))
                if has_content:
                    first_token_time = time.time()

            # Get actual token counts from the last chunk
            if chunk.usage:
                input_tokens = chunk.usage.prompt_tokens
                output_tokens = chunk.usage.completion_tokens

        request_end = time.time()

        ttft_ms = (first_token_time - request_start) * 1000 if first_token_time else 0
        total_time_ms = (request_end - request_start) * 1000

        return {
            'ttft_ms': ttft_ms,
            'total_time_ms': total_time_ms,
            'input_tokens': input_tokens,
            'output_tokens': output_tokens,
            'success': True
        }

    except Exception as e:
        print(f"  Request {idx + 1} failed: {e}", file=sys.stderr)
        return {'success': False}


def run_openai_benchmark(server_url, prompts, model="local-model", max_tokens=50000, parallel=1):
    """
    Run benchmark using OpenAI-compatible API.
    Returns metrics in vLLM-compatible format.

    Args:
        parallel: Number of parallel requests (1 = sequential, >1 = concurrent)
    """
    client = OpenAI(base_url=f"{server_url}/v1", api_key="dummy")

    results = []
    total_input_tokens = 0
    total_output_tokens = 0

    print(f"Running benchmark with {len(prompts)} prompts (parallelism={parallel})...")
    start_time = time.time()

    if parallel == 1:
        # Sequential execution (original behavior)
        for idx, prompt in enumerate(prompts):
            result = process_single_request(client, prompt, idx, model, max_tokens)
            results.append(result)

            if result.get('success', False):
                total_input_tokens += result['input_tokens']
                total_output_tokens += result['output_tokens']

            if (idx + 1) % 10 == 0:
                print(f"  Completed {idx + 1}/{len(prompts)} requests")
    else:
        # Parallel execution
        with ThreadPoolExecutor(max_workers=parallel) as executor:
            futures = [
                executor.submit(process_single_request, client, prompt, idx, model, max_tokens)
                for idx, prompt in enumerate(prompts)
            ]

            for idx, future in enumerate(futures):
                result = future.result()
                results.append(result)

                if result.get('success', False):
                    total_input_tokens += result['input_tokens']
                    total_output_tokens += result['output_tokens']

                if (idx + 1) % 10 == 0:
                    print(f"  Completed {idx + 1}/{len(prompts)} requests")

    end_time = time.time()
    benchmark_duration = end_time - start_time

    # Calculate metrics
    successful_results = [r for r in results if r.get('success', False)]
    if not successful_results:
        print("Error: No successful requests", file=sys.stderr)
        return None

    successful_requests = len(successful_results)
    ttfts = [r['ttft_ms'] for r in successful_results]
    tpots = [(r['total_time_ms'] - r['ttft_ms']) / r['output_tokens']
             if r['output_tokens'] > 0 else 0 for r in successful_results]

    # Return vLLM-compatible metrics
    return {
        'successful_requests': successful_requests,
        'benchmark_duration': benchmark_duration,
        'total_input_tokens': total_input_tokens,
        'total_generated_tokens': total_output_tokens,
        'request_throughput': successful_requests / benchmark_duration,
        'output_token_throughput': total_output_tokens / benchmark_duration,
        'total_token_throughput': (total_input_tokens + total_output_tokens) / benchmark_duration,
        'mean_ttft': float(np.mean(ttfts)),
        'median_ttft': float(np.median(ttfts)),
        'p99_ttft': float(np.percentile(ttfts, 99)),
        'mean_tpot': float(np.mean(tpots)),
        'median_tpot': float(np.median(tpots)),
        'p99_tpot': float(np.percentile(tpots, 99)),
        'mean_itl': float(np.mean(tpots)),
        'median_itl': float(np.median(tpots)),
        'p99_itl': float(np.percentile(tpots, 99)),
    }


def format_as_vllm_output(metrics):
    """Format metrics as vLLM benchmark output for parsing"""
    output = []
    output.append(f"Successful requests: {metrics['successful_requests']}")
    output.append(f"Benchmark duration (s): {metrics['benchmark_duration']:.2f}")
    output.append(f"Total input tokens: {metrics['total_input_tokens']}")
    output.append(f"Total generated tokens: {metrics['total_generated_tokens']}")
    output.append(f"Request throughput (req/s): {metrics['request_throughput']:.2f}")
    output.append(f"Output token throughput (tok/s): {metrics['output_token_throughput']:.2f}")
    output.append(f"Total Token throughput (tok/s): {metrics['total_token_throughput']:.2f}")
    output.append(f"Mean TTFT (ms): {metrics['mean_ttft']:.2f}")
    output.append(f"Median TTFT (ms): {metrics['median_ttft']:.2f}")
    output.append(f"P99 TTFT (ms): {metrics['p99_ttft']:.2f}")
    output.append(f"Mean TPOT (ms): {metrics['mean_tpot']:.2f}")
    output.append(f"Median TPOT (ms): {metrics['median_tpot']:.2f}")
    output.append(f"P99 TPOT (ms): {metrics['p99_tpot']:.2f}")
    output.append(f"Mean ITL (ms): {metrics['mean_itl']:.2f}")
    output.append(f"Median ITL (ms): {metrics['median_itl']:.2f}")
    output.append(f"P99 ITL (ms): {metrics['p99_itl']:.2f}")
    return "\n".join(output)


def main():
    parser = argparse.ArgumentParser(
        description='llama.cpp benchmark client with vLLM-compatible output'
    )
    parser.add_argument('--server-url', type=str, default='http://localhost:8000',
                        help='llama.cpp server URL (default: http://localhost:8080)')
    parser.add_argument('--num-prompts', type=int, default=10,
                        help='Number of prompts to test (default: 10)')
    parser.add_argument('--dataset-path', type=str,
                        default='datasets/ShareGPT_V3_unfiltered_cleaned_split.json',
                        help='Path to ShareGPT dataset')
    parser.add_argument('--max-tokens', type=int, default=4000,
                        help='Max tokens per response (default: 4000)')
    parser.add_argument('--model', type=str, default='Qwen/Qwen3-30B-A3B-FP8',
                        help='Model name (default: Qwen/Qwen3-30B-A3B-FP8)')
    parser.add_argument('--parallel', type=int, default=1,
                        help='Number of parallel requests (default: 1 for sequential)')

    args = parser.parse_args()

    # Load dataset
    print(f"Loading dataset from {args.dataset_path}...")
    try:
        prompts = load_sharegpt_dataset(args.dataset_path, args.num_prompts)
        print(f"Loaded {len(prompts)} prompts\n")
    except Exception as e:
        print(f"Error loading dataset: {e}", file=sys.stderr)
        return 1

    # Run benchmark
    print(f"Connecting to llama.cpp server at {args.server_url}...")
    metrics = run_openai_benchmark(
        args.server_url,
        prompts,
        model=args.model,
        max_tokens=args.max_tokens,
        parallel=args.parallel
    )

    if not metrics:
        return 1

    # Print results in vLLM format
    print("\n" + "="*60)
    print(format_as_vllm_output(metrics))
    print("="*60)

    return 0


if __name__ == "__main__":
    sys.exit(main())

