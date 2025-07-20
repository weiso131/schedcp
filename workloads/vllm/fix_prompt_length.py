#!/usr/bin/env python3
"""
Script to fix vLLM prompt length issues
Truncates prompts that exceed maximum model length
"""

import json
import tiktoken
from typing import List, Dict, Optional
from transformers import AutoTokenizer


def estimate_token_count(text: str, method: str = "approximate") -> int:
    """Estimate token count for a text string"""
    if method == "tiktoken":
        try:
            # Use tiktoken for better accuracy (OpenAI's tokenizer)
            enc = tiktoken.get_encoding("cl100k_base")
            return len(enc.encode(text))
        except:
            pass
    
    # Fallback to approximation
    # Rough estimate: 1 token ≈ 4 characters or 0.75 words
    word_count = len(text.split())
    char_count = len(text)
    return max(int(word_count * 1.3), int(char_count / 4))


def truncate_prompt(prompt: str, max_tokens: int, tokenizer=None) -> str:
    """Truncate prompt to fit within max tokens"""
    if tokenizer:
        # Use actual tokenizer for accurate truncation
        tokens = tokenizer.encode(prompt, add_special_tokens=False)
        if len(tokens) <= max_tokens:
            return prompt
        truncated_tokens = tokens[:max_tokens]
        return tokenizer.decode(truncated_tokens, skip_special_tokens=True)
    else:
        # Approximate truncation based on character count
        estimated_tokens = estimate_token_count(prompt)
        if estimated_tokens <= max_tokens:
            return prompt
        
        # Truncate by reducing characters proportionally
        ratio = max_tokens / estimated_tokens
        target_chars = int(len(prompt) * ratio * 0.9)  # 90% to be safe
        return prompt[:target_chars] + "..."


def fix_sharegpt_dataset(input_file: str, output_file: str, max_prompt_tokens: int = 1800):
    """Fix ShareGPT dataset by truncating long prompts"""
    print(f"Processing {input_file}...")
    
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Try to load a tokenizer for accurate truncation
    tokenizer = None
    try:
        tokenizer = AutoTokenizer.from_pretrained("facebook/opt-125m")
        print("Using OPT tokenizer for accurate truncation")
    except:
        print("Using approximate truncation (tokenizer not available)")
    
    fixed_data = []
    truncated_count = 0
    
    for item in data:
        new_item = item.copy()
        
        # Handle different ShareGPT formats
        if 'prompt' in item:
            original_prompt = item['prompt']
            token_count = estimate_token_count(original_prompt)
            
            if token_count > max_prompt_tokens:
                new_item['prompt'] = truncate_prompt(original_prompt, max_prompt_tokens, tokenizer)
                truncated_count += 1
                
        elif 'conversations' in item:
            new_conversations = []
            for conv in item['conversations']:
                new_conv = conv.copy()
                if conv.get('from') in ['human', 'user']:
                    original_text = conv['value']
                    token_count = estimate_token_count(original_text)
                    
                    if token_count > max_prompt_tokens:
                        new_conv['value'] = truncate_prompt(original_text, max_prompt_tokens, tokenizer)
                        truncated_count += 1
                
                new_conversations.append(new_conv)
            new_item['conversations'] = new_conversations
        
        fixed_data.append(new_item)
    
    # Save fixed dataset
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(fixed_data, f, indent=2, ensure_ascii=False)
    
    print(f"Fixed dataset saved to {output_file}")
    print(f"Truncated {truncated_count} prompts out of {len(data)} total items")
    print(f"Maximum prompt tokens: {max_prompt_tokens}")


def create_vllm_wrapper():
    """Create a wrapper script that handles prompt truncation automatically"""
    wrapper_content = '''#!/usr/bin/env python3
"""
vLLM wrapper with automatic prompt truncation
"""

import os
import sys
from typing import List, Union

# Ensure we're not importing from local directory
if 'vllm' in os.listdir('.'):
    sys.path = [p for p in sys.path if not p.endswith('/vllm')]

from vllm import LLM, SamplingParams
from vllm.inputs import TextPrompt


class TruncatingLLM(LLM):
    """LLM wrapper that automatically truncates long prompts"""
    
    def __init__(self, *args, **kwargs):
        # Extract max_model_len if specified
        self.max_prompt_ratio = kwargs.pop('max_prompt_ratio', 0.9)  # Use 90% of max_model_len for prompts
        super().__init__(*args, **kwargs)
        
        # Calculate maximum prompt length
        self.max_prompt_tokens = int(self.llm_engine.model_config.max_model_len * self.max_prompt_ratio)
        
        print(f"Initialized TruncatingLLM with max_prompt_tokens={self.max_prompt_tokens}")
    
    def _truncate_prompt(self, prompt: str) -> str:
        """Truncate prompt if it exceeds maximum length"""
        try:
            # Try to use the model's tokenizer
            tokenizer = self.llm_engine.tokenizer
            tokens = tokenizer.encode(prompt)
            
            if len(tokens) <= self.max_prompt_tokens:
                return prompt
            
            # Truncate and decode
            truncated_tokens = tokens[:self.max_prompt_tokens]
            truncated_prompt = tokenizer.decode(truncated_tokens)
            
            print(f"Truncated prompt from {len(tokens)} to {len(truncated_tokens)} tokens")
            return truncated_prompt
            
        except Exception as e:
            # Fallback to character-based truncation
            max_chars = self.max_prompt_tokens * 4  # Rough estimate
            if len(prompt) > max_chars:
                print(f"Truncated prompt from {len(prompt)} to {max_chars} characters")
                return prompt[:max_chars] + "..."
            return prompt
    
    def generate(
        self,
        prompts: Union[str, List[str]],
        sampling_params: SamplingParams,
        *args,
        **kwargs
    ):
        """Generate with automatic prompt truncation"""
        # Handle single prompt
        if isinstance(prompts, str):
            prompts = [self._truncate_prompt(prompts)]
        else:
            # Handle list of prompts
            prompts = [self._truncate_prompt(p) for p in prompts]
        
        return super().generate(prompts, sampling_params, *args, **kwargs)


# Example usage
if __name__ == "__main__":
    llm = TruncatingLLM(
        model="facebook/opt-125m",
        max_model_len=2048,
        max_prompt_ratio=0.9  # Use 90% of max length for prompts
    )
    
    # Test with a long prompt
    long_prompt = "This is a test " * 1000  # Very long prompt
    
    sampling_params = SamplingParams(
        temperature=0.8,
        max_tokens=100
    )
    
    outputs = llm.generate([long_prompt], sampling_params)
    print(f"Generated: {outputs[0].outputs[0].text[:100]}...")
'''
    
    with open('/root/yunwei37/ai-os/workloads/vllm/truncating_llm.py', 'w') as f:
        f.write(wrapper_content)
    
    print("Created truncating_llm.py wrapper")


def main():
    """Main function to fix prompt length issues"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Fix vLLM prompt length issues")
    parser.add_argument("--fix-dataset", action="store_true", 
                       help="Fix ShareGPT dataset by truncating long prompts")
    parser.add_argument("--create-wrapper", action="store_true",
                       help="Create TruncatingLLM wrapper class")
    parser.add_argument("--input", default="../datasets/ShareGPT_V3_unfiltered_cleaned_split.json",
                       help="Input dataset file")
    parser.add_argument("--output", default="../datasets/ShareGPT_V3_truncated.json",
                       help="Output dataset file")
    parser.add_argument("--max-prompt-tokens", type=int, default=1800,
                       help="Maximum prompt tokens (default: 1800)")
    
    args = parser.parse_args()
    
    if args.fix_dataset:
        fix_sharegpt_dataset(args.input, args.output, args.max_prompt_tokens)
    
    if args.create_wrapper:
        create_vllm_wrapper()
    
    if not args.fix_dataset and not args.create_wrapper:
        print("Please specify --fix-dataset or --create-wrapper")
        parser.print_help()


if __name__ == "__main__":
    main()