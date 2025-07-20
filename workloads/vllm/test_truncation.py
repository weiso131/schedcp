#!/usr/bin/env python3
"""
Test script to verify prompt truncation fix
"""

import os
import sys

# Ensure we're not importing from local directory
if 'vllm' in os.listdir('.'):
    sys.path = [p for p in sys.path if not p.endswith('/vllm')]

from vllm import LLM, SamplingParams


def test_long_prompt():
    """Test with a very long prompt to verify truncation works"""
    print("Testing vLLM with long prompt...")
    
    # Create a very long prompt (over 2048 tokens)
    long_prompt = "Once upon a time, in a land far far away, " * 200
    print(f"Original prompt length: {len(long_prompt)} characters")
    
    # Initialize model
    llm = LLM(
        model="facebook/opt-125m",
        max_model_len=2048,
        gpu_memory_utilization=0.5,
        enforce_eager=True
    )
    
    # Truncate prompt manually to fit within limits
    # Leave room for generation (use 90% of max_model_len)
    max_prompt_chars = int(2048 * 0.9 * 4)  # Rough estimate: 1 token ≈ 4 chars
    if len(long_prompt) > max_prompt_chars:
        truncated_prompt = long_prompt[:max_prompt_chars] + "..."
        print(f"Truncated prompt length: {len(truncated_prompt)} characters")
    else:
        truncated_prompt = long_prompt
    
    # Generate
    sampling_params = SamplingParams(
        temperature=0.8,
        max_tokens=50
    )
    
    try:
        outputs = llm.generate([truncated_prompt], sampling_params)
        print("\n✓ Generation successful!")
        print(f"Generated text: {outputs[0].outputs[0].text[:100]}...")
        return True
    except Exception as e:
        print(f"\n✗ Generation failed: {e}")
        return False


def test_with_wrapper():
    """Test using the TruncatingLLM wrapper"""
    print("\nTesting with TruncatingLLM wrapper...")
    
    # Import the wrapper if it exists
    try:
        sys.path.insert(0, '/root/yunwei37/ai-os/workloads/vllm')
        from truncating_llm import TruncatingLLM
        
        llm = TruncatingLLM(
            model="facebook/opt-125m",
            max_model_len=2048,
            max_prompt_ratio=0.9
        )
        
        # Test with very long prompt
        long_prompt = "The quick brown fox jumps over the lazy dog. " * 500
        print(f"Testing with {len(long_prompt)} character prompt...")
        
        sampling_params = SamplingParams(
            temperature=0.8,
            max_tokens=50
        )
        
        outputs = llm.generate([long_prompt], sampling_params)
        print("✓ Wrapper test successful!")
        print(f"Generated: {outputs[0].outputs[0].text[:100]}...")
        return True
        
    except ImportError:
        print("TruncatingLLM wrapper not found. Run fix_prompt_length.py --create-wrapper first.")
        return False
    except Exception as e:
        print(f"✗ Wrapper test failed: {e}")
        return False


def main():
    print("=" * 60)
    print("vLLM Prompt Truncation Test")
    print("=" * 60)
    
    # Test 1: Manual truncation
    success1 = test_long_prompt()
    
    # Test 2: Wrapper (if available)
    success2 = test_with_wrapper()
    
    print("\n" + "=" * 60)
    print("Test Results:")
    print(f"Manual truncation test: {'PASSED' if success1 else 'FAILED'}")
    print(f"Wrapper test: {'PASSED' if success2 else 'FAILED'}")
    print("=" * 60)


if __name__ == "__main__":
    main()