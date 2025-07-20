# vLLM Benchmark Guide

This guide explains how to build vLLM and run various benchmarks to measure latency and throughput performance.

## Quick Start

```bash
# Install vLLM (recommended)
make install

# Run a quick benchmark
make benchmark-quick

# Download benchmark datasets
make download-datasets
```

## Installation

### Option 1: Install from PyPI (Recommended)
```bash
make install-pip
# or simply
make install
```

### Option 2: Build from Source
```bash
# Install dependencies first
make install-deps

# Build from source
make install-source
# or
INSTALL_TYPE=source make install
```

### Custom Build Options
```bash
# Specify CUDA architectures
CUDA_ARCH="75,80,86,89,90" make install-source

# Control build parallelism
MAX_JOBS=8 make install-source
```

## Benchmark Types

vLLM provides three main types of benchmarks:

1. **Latency Benchmark** - Measures end-to-end latency for processing requests
2. **Throughput Benchmark** - Measures offline batch processing throughput
3. **Serving Benchmark** - Measures online serving performance with concurrent requests

## Running Benchmarks

### 1. Latency Benchmark

Measures the time to process a single batch of requests.

```bash
# Basic usage
make benchmark-latency MODEL=meta-llama/Llama-2-7b-hf

# With custom parameters
make benchmark-latency \
    MODEL=meta-llama/Llama-2-7b-hf \
    BATCH_SIZE=4 \
    INPUT_LEN=128 \
    OUTPUT_LEN=256 \
    WARMUP=10 \
    ITERS=30
```

**Direct Python command:**
```bash
cd vllm
python benchmarks/benchmark_latency.py \
    --model meta-llama/Llama-2-7b-hf \
    --batch-size 1 \
    --input-len 32 \
    --output-len 128 \
    --num-iters-warmup 10 \
    --num-iters 30
```

**Key metrics:**
- Average latency (seconds)
- Percentile latencies (P50, P90, P99)
- Tokens per second

### 2. Throughput Benchmark

Measures offline batch processing performance.

```bash
# Basic usage with ShareGPT dataset
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

**Direct Python command:**
```bash
cd vllm
python benchmarks/benchmark_throughput.py \
    --model meta-llama/Llama-2-7b-hf \
    --dataset-name sharegpt \
    --dataset-path ../datasets/ShareGPT_V3_unfiltered_cleaned_split.json \
    --num-prompts 1000
```

**Key metrics:**
- Requests per second
- Total tokens per second
- Output tokens per second

### 3. Serving Benchmark

Measures online serving performance with concurrent requests.

**Step 1: Start the vLLM server**
```bash
vllm serve meta-llama/Llama-2-7b-hf --disable-log-requests
```

**Step 2: Run the benchmark**
```bash
make benchmark-serving \
    MODEL=meta-llama/Llama-2-7b-hf \
    PORT=8000 \
    NUM_PROMPTS=100 \
    DATASET=sharegpt \
    DATASET_PATH=datasets/ShareGPT_V3_unfiltered_cleaned_split.json
```

**Direct Python command:**
```bash
cd vllm
python benchmarks/benchmark_serving.py \
    --backend vllm \
    --model meta-llama/Llama-2-7b-hf \
    --endpoint /v1/completions \
    --port 8000 \
    --dataset-name sharegpt \
    --dataset-path ../datasets/ShareGPT_V3_unfiltered_cleaned_split.json \
    --num-prompts 100
```

**Key metrics:**
- Request throughput (req/s)
- Token throughput (tok/s)
- Time to First Token (TTFT)
- Time per Output Token (TPOT)
- Inter-token Latency (ITL)

## Dataset Options

### Available Datasets

1. **ShareGPT** - Real conversation data
   ```bash
   make download-datasets  # Downloads ShareGPT automatically
   ```

2. **Random** - Synthetic random tokens
   ```bash
   DATASET=random INPUT_LEN=128 OUTPUT_LEN=128 make benchmark-throughput ...
   ```

3. **Sonnet** - Shakespeare's sonnets
   ```bash
   DATASET=sonnet DATASET_PATH=vllm/benchmarks/sonnet.txt make benchmark-throughput ...
   ```

4. **Custom Dataset** - Your own JSONL file
   ```bash
   # Format: {"prompt": "Your prompt here"}
   DATASET=custom DATASET_PATH=your_data.jsonl make benchmark-throughput ...
   ```

## Advanced Usage

### Benchmarking with Different Models

```bash
# Small model for testing
make benchmark-throughput MODEL=facebook/opt-125m NUM_PROMPTS=1000

# Large model
make benchmark-throughput MODEL=meta-llama/Llama-2-70b-hf NUM_PROMPTS=100

# Custom/local model
make benchmark-throughput MODEL=/path/to/your/model NUM_PROMPTS=100
```

### Benchmarking with GPU Memory Constraints

```bash
# Limit GPU memory usage
export CUDA_VISIBLE_DEVICES=0
export VLLM_ENFORCE_EAGER=1  # Disable CUDA graphs for lower memory usage

make benchmark-throughput MODEL=meta-llama/Llama-2-7b-hf NUM_PROMPTS=100
```

### Benchmarking with Quantization

```bash
# Using quantized models
vllm serve meta-llama/Llama-2-7b-hf \
    --quantization awq \
    --disable-log-requests

# Then run serving benchmark
make benchmark-serving MODEL=meta-llama/Llama-2-7b-hf
```

### Benchmarking with Tensor Parallelism

```bash
# For multi-GPU setups
vllm serve meta-llama/Llama-2-70b-hf \
    --tensor-parallel-size 4 \
    --disable-log-requests

make benchmark-serving MODEL=meta-llama/Llama-2-70b-hf
```

## Interpreting Results

### Latency Metrics
- **Average Latency**: Mean time to process a batch
- **P99 Latency**: 99th percentile latency (worst-case for 99% of requests)
- **Throughput**: Tokens generated per second

### Throughput Metrics
- **Requests/s**: Number of requests processed per second
- **Total tokens/s**: Input + output tokens per second
- **Output tokens/s**: Generated tokens per second

### Serving Metrics
- **TTFT (Time to First Token)**: Critical for streaming applications
- **TPOT (Time per Output Token)**: Generation speed after first token
- **ITL (Inter-token Latency)**: Time between consecutive tokens

## Performance Tips

1. **Warmup is Important**: Always include warmup iterations to get stable results
2. **Batch Size**: Larger batch sizes generally improve throughput but increase latency
3. **Input/Output Length**: Longer sequences require more memory and compute
4. **Model Size**: Smaller models are faster but less capable
5. **Quantization**: Can significantly improve performance with minimal quality loss

## Troubleshooting

### Out of Memory Errors
```bash
# Reduce batch size
BATCH_SIZE=1 make benchmark-latency MODEL=...

# Use smaller sequence lengths
INPUT_LEN=64 OUTPUT_LEN=64 make benchmark-throughput MODEL=...
```

### CUDA Not Found
```bash
# Specify CUDA home
export CUDA_HOME=/usr/local/cuda-11.8
make install-source
```

### Slow Performance
```bash
# Check GPU utilization
nvidia-smi

# Use performance mode
sudo nvidia-smi -pm 1
sudo nvidia-smi -pl 300  # Set power limit
```

## Example Benchmark Workflow

```bash
# 1. Install vLLM
make install

# 2. Download datasets
make download-datasets

# 3. Run latency benchmark
make benchmark-latency MODEL=meta-llama/Llama-2-7b-hf

# 4. Run throughput benchmark
make benchmark-throughput \
    MODEL=meta-llama/Llama-2-7b-hf \
    NUM_PROMPTS=1000 \
    DATASET=sharegpt \
    DATASET_PATH=datasets/ShareGPT_V3_unfiltered_cleaned_split.json

# 5. Start server for serving benchmark
vllm serve meta-llama/Llama-2-7b-hf --disable-log-requests &

# 6. Run serving benchmark
make benchmark-serving \
    MODEL=meta-llama/Llama-2-7b-hf \
    NUM_PROMPTS=100
```

## Saving Results

To save benchmark results:

```bash
# Redirect output to file
make benchmark-throughput MODEL=... 2>&1 | tee benchmark_results.txt

# For serving benchmarks with detailed results
cd vllm
python benchmarks/benchmark_serving.py \
    --model meta-llama/Llama-2-7b-hf \
    --save-result \
    --result-dir ./results \
    --result-filename benchmark_$(date +%Y%m%d_%H%M%S).json
```

## References

- [vLLM Documentation](https://docs.vllm.ai)
- [vLLM GitHub Repository](https://github.com/vllm-project/vllm)
- [Benchmarking Best Practices](https://docs.vllm.ai/en/latest/dev/benchmarks.html)