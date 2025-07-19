#!/bin/bash

echo "=== llama.cpp ShareGPT Evaluation Script ==="
echo

# Check if build exists
if [ ! -f "build/bin/llama-server" ]; then
    echo "Building llama.cpp..."
    make build
fi

# Check if model exists
if [ ! -f "models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf" ]; then
    echo "Downloading test model..."
    python download_test_model.py
fi

# Check if ShareGPT dataset exists
if [ ! -f "datasets/sharegpt_benchmark.json" ]; then
    echo "Downloading ShareGPT dataset..."
    python download_sharegpt.py --dataset vicuna --num-samples 100
fi

# Run evaluation
echo "Running ShareGPT evaluation..."
echo "Note: This will test llama.cpp server with different schedulers"
echo

# You can adjust these parameters:
# --num-samples: Number of ShareGPT prompts to test (default: 100)
# --max-concurrent: Maximum concurrent requests (default: 10)
# --production-only: Only test production schedulers
# --schedulers: Specific schedulers to test

python sharegpt_llama_server_eval.py \
    --num-samples 20 \
    --max-concurrent 4 \
    --output-dir results

echo
echo "Evaluation complete! Check results/ directory for:"
echo "  - sharegpt_benchmark_summary.txt: Performance summary"
echo "  - sharegpt_llama_server_performance.png: Performance charts"
echo "  - sharegpt_llama_server_results.json: Detailed results"