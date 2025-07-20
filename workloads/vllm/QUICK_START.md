# vLLM Quick Start for Benchmarking

## Summary

I've created a comprehensive setup for running vLLM benchmarks to measure latency and throughput. The setup includes:

1. **Makefile** - Automates building, installation, and benchmark execution
2. **BENCHMARK_GUIDE.md** - Detailed documentation on running various benchmarks
3. **Downloaded datasets** - ShareGPT and Sonnet datasets ready for testing

## Installation Status

vLLM has been installed via pip. However, there are some compilation issues with the CUDA environment that may require additional system dependencies.

## Quick Commands

### 1. Latency Benchmark
```bash
# Basic latency test
make benchmark-latency MODEL=facebook/opt-125m

# With custom parameters
make benchmark-latency \
    MODEL=meta-llama/Llama-2-7b-hf \
    BATCH_SIZE=4 \
    INPUT_LEN=128 \
    OUTPUT_LEN=256
```

### 2. Throughput Benchmark
```bash
# Using ShareGPT dataset
make benchmark-throughput \
    MODEL=meta-llama/Llama-2-7b-hf \
    NUM_PROMPTS=1000 \
    DATASET=sharegpt \
    DATASET_PATH=datasets/ShareGPT_V3_unfiltered_cleaned_split.json

# Using random dataset
make benchmark-throughput \
    MODEL=facebook/opt-125m \
    NUM_PROMPTS=1000 \
    DATASET=random \
    INPUT_LEN=128 \
    OUTPUT_LEN=128
```

### 3. Serving Benchmark
```bash
# Start server first
vllm serve meta-llama/Llama-2-7b-hf --disable-log-requests

# Then run benchmark
make benchmark-serving \
    MODEL=meta-llama/Llama-2-7b-hf \
    PORT=8000 \
    NUM_PROMPTS=100
```

## Key Metrics

### Latency Benchmark
- **Average Latency**: End-to-end time for processing a batch
- **Percentile Latencies**: P50, P90, P99 for performance distribution
- **Tokens/second**: Generation speed

### Throughput Benchmark
- **Requests/second**: Overall throughput
- **Total tokens/second**: Combined input + output processing speed
- **Output tokens/second**: Generation-only speed

### Serving Benchmark
- **TTFT (Time to First Token)**: Critical for streaming
- **TPOT (Time per Output Token)**: Generation speed after first token
- **ITL (Inter-token Latency)**: Time between consecutive tokens

## Troubleshooting

If you encounter compilation errors:

1. **Install development dependencies**:
```bash
sudo apt-get update
sudo apt-get install -y build-essential python3-dev
```

2. **Check CUDA installation**:
```bash
nvcc --version
nvidia-smi
```

3. **Use CPU-only mode** (if GPU issues):
```bash
export VLLM_TARGET_DEVICE=cpu
```

## Files Created

- `/root/yunwei37/ai-os/workloads/vllm/Makefile` - Build and benchmark automation
- `/root/yunwei37/ai-os/workloads/vllm/BENCHMARK_GUIDE.md` - Comprehensive benchmark documentation
- `/root/yunwei37/ai-os/workloads/vllm/datasets/` - Downloaded benchmark datasets

## Next Steps

1. Fix the compilation issues by installing missing system dependencies
2. Run the benchmarks with your desired models
3. Analyze the results to optimize performance

For more detailed information, see the BENCHMARK_GUIDE.md file.