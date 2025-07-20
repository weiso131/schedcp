#!/usr/bin/env python3
"""
Simple benchmark using vLLM V0 engine
"""

import time
import os
import sys

# Force V0 engine
os.environ['VLLM_USE_V1'] = '0'

# Ensure we're not importing from local directory
if 'vllm' in os.listdir('.'):
    sys.path = [p for p in sys.path if not p.endswith('/vllm')]

from vllm import LLM, SamplingParams

def run_simple_test():
    """Run a simple test with minimal setup"""
    print("=" * 60)
    print("Simple vLLM V0 Benchmark")
    print("=" * 60)
    
    model_name = "facebook/opt-125m"
    
    print(f"Model: {model_name}")
    print("Using V0 engine")
    print()
    
    try:
        # Initialize model with minimal settings
        print("Loading model...")
        start_load = time.time()
        llm = LLM(
            model=model_name,
            trust_remote_code=True,
            max_model_len=512,  # Smaller context
            gpu_memory_utilization=0.5,  # Less memory
            enforce_eager=True  # Disable CUDA graphs
        )
        load_time = time.time() - start_load
        print(f"✓ Model loaded in {load_time:.2f} seconds")
        
        # Simple test
        prompts = [
            "Hello, how are you?",
            "What is the capital of France?",
            "Tell me a joke.",
        ]
        
        sampling_params = SamplingParams(
            temperature=0.8,
            top_p=0.95,
            max_tokens=20
        )
        
        print(f"\nGenerating responses for {len(prompts)} prompts...")
        start = time.time()
        outputs = llm.generate(prompts, sampling_params)
        end = time.time()
        
        print(f"\n✓ Generation completed in {end - start:.2f} seconds")
        
        # Show results
        print("\nResults:")
        print("-" * 60)
        for i, output in enumerate(outputs):
            prompt = output.prompt
            generated_text = output.outputs[0].text
            tokens = len(output.outputs[0].token_ids)
            print(f"Prompt {i+1}: {prompt}")
            print(f"Response: {generated_text}")
            print(f"Tokens: {tokens}")
            print("-" * 60)
        
        # Calculate basic metrics
        total_time = end - start
        total_prompts = len(prompts)
        prompts_per_sec = total_prompts / total_time
        
        print(f"\nMetrics:")
        print(f"Total time: {total_time:.2f}s")
        print(f"Prompts/sec: {prompts_per_sec:.2f}")
        
        return True
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("vLLM V0 Simple Test")
    print("=" * 60)
    
    # Check vLLM
    try:
        import vllm
        print(f"vLLM version: {vllm.__version__}")
        print(f"Using V0 engine: {os.environ.get('VLLM_USE_V1', '1') == '0'}")
    except Exception as e:
        print(f"Failed to import vLLM: {e}")
        sys.exit(1)
    
    # Run test
    if run_simple_test():
        print("\n✓ Test completed successfully!")
        print("\nTo run full benchmarks, you may need to:")
        print("1. Fix CUDA compilation issues")
        print("2. Install missing system dependencies")
        print("3. Use a different GPU or run in CPU mode")
    else:
        print("\n✗ Test failed.")

if __name__ == "__main__":
    main()