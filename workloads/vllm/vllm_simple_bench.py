#!/usr/bin/env python3
"""
Ultra-simple vLLM benchmark that avoids all the complexity.
Uses vLLM's native throughput measurement.
"""

import time
import json
from vllm import LLM, SamplingParams

# Configuration
MODEL = "meta-llama/Llama-3.2-3B"
DATASET_PATH = "datasets/ShareGPT_V3_unfiltered_cleaned_split.json"
NUM_PROMPTS = 50  # Reduced for faster testing
MAX_TOKENS = 100  # Tokens to generate per prompt
MAX_MODEL_LEN = 1536  # Reduced to avoid OOM

print(f"Simple vLLM Benchmark")
print(f"Model: {MODEL}")
print(f"Dataset: {DATASET_PATH}")
print(f"Prompts: {NUM_PROMPTS}")
print(f"Max tokens per prompt: {MAX_TOKENS}")
print(f"Max model length: {MAX_MODEL_LEN}\n")

# Load prompts
print("Loading prompts...")
with open(DATASET_PATH, 'r') as f:
    dataset = json.load(f)

prompts = []
for i, item in enumerate(dataset[:NUM_PROMPTS]):
    if "conversations" in item:
        for conv in item["conversations"]:
            if conv.get("from") == "human":
                # Truncate very long prompts
                prompt = conv["value"]
                if len(prompt) > 1000:  # Simple length check
                    prompt = prompt[:1000]
                prompts.append(prompt)
                break

print(f"Loaded {len(prompts)} prompts\n")

# Initialize vLLM
print("Initializing vLLM...")
llm = LLM(
    model=MODEL,
    max_model_len=MAX_MODEL_LEN,
    trust_remote_code=True,
    download_dir="/tmp/vllm_models",
    gpu_memory_utilization=0.85,
)

# Set sampling parameters
sampling_params = SamplingParams(
    temperature=1.0,
    top_p=1.0,
    max_tokens=MAX_TOKENS,
)

# Run benchmark
print("\nRunning benchmark...")
start_time = time.time()

outputs = llm.generate(prompts, sampling_params, use_tqdm=True)

end_time = time.time()
elapsed = end_time - start_time

# Calculate metrics
total_tokens = sum(len(output.outputs[0].token_ids) for output in outputs)

print(f"\n{'='*50}")
print(f"BENCHMARK RESULTS")
print(f"{'='*50}")
print(f"Total prompts processed: {len(outputs)}")
print(f"Total time: {elapsed:.2f} seconds")
print(f"Throughput: {len(outputs)/elapsed:.2f} prompts/second")
print(f"Total tokens generated: {total_tokens}")
print(f"Token throughput: {total_tokens/elapsed:.2f} tokens/second")
print(f"Average tokens per prompt: {total_tokens/len(outputs):.2f}")
print(f"{'='*50}")