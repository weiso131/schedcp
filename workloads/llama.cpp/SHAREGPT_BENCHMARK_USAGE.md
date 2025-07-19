# ShareGPT Benchmark Usage Guide

## Overview

The updated ShareGPT benchmark script (`sharegpt_llama_server_eval.py`) now supports multiple datasets, configurable test parameters, system info logging, and server output logging.

## New Features

### 1. Dataset Support

- **Default**: Uses `sharegpt_vicuna.json` if available, falls back to `sharegpt_benchmark.json`
- **Custom**: Specify any dataset with `--dataset path/to/dataset.json`

### 2. Automatic Output Directory Naming

Output directories are automatically named with:
- Dataset name
- Number of samples (s)
- Max concurrent requests (c)  
- Timestamp

Example: `results/sharegpt_vicuna_s100_c10_20240119_143025/`

### 3. System Information Collection

Each benchmark run now collects and logs:
- Platform and OS information
- CPU cores and frequency
- Memory capacity
- Kernel version
- Python version
- Timestamp

### 4. Server Logging

Enable server logging with `--server-logs` flag:
- Server stdout saved to: `server_logs.stdout`
- Server stderr saved to: `server_logs.stderr`

### 5. Configurable Test Parameters

- `--num-samples`: Number of prompts to test (default: 100)
- `--max-concurrent`: Maximum concurrent requests (default: 10)

## Usage Examples

### Basic Usage (with defaults)
```bash
python sharegpt_llama_server_eval.py
```
This will:
- Use `sharegpt_vicuna.json` dataset (or fallback)
- Test 100 samples with 10 concurrent requests
- Create timestamped output directory

### Custom Dataset
```bash
python sharegpt_llama_server_eval.py \
    --dataset datasets/sharegpt_benchmark.json \
    --num-samples 50 \
    --max-concurrent 5
```

### Production Testing with Logging
```bash
python sharegpt_llama_server_eval.py \
    --production-only \
    --server-logs \
    --num-samples 200 \
    --max-concurrent 20
```

### Specific Schedulers
```bash
python sharegpt_llama_server_eval.py \
    --schedulers scx_rusty scx_lavd scx_bpfland \
    --num-samples 30 \
    --server-logs
```

### Custom Output Directory
```bash
python sharegpt_llama_server_eval.py \
    --output-dir results/my_custom_test \
    --server-logs
```

## Output Structure

```
results/sharegpt_vicuna_s100_c10_20240119_143025/
├── sharegpt_llama_server_results.json    # Results with metadata
├── sharegpt_llama_server_performance.png # Performance charts
├── sharegpt_benchmark_summary.txt        # Human-readable summary
├── server_logs.stdout                    # Server stdout (if --server-logs)
└── server_logs.stderr                    # Server stderr (if --server-logs)
```

## Results File Structure

The JSON results file now includes:
```json
{
  "system_info": {
    "system": {...},
    "cpu": {...},
    "memory": {...},
    "kernel": "...",
    "timestamp": "..."
  },
  "dataset": "sharegpt_vicuna",
  "config": {
    "num_samples": 100,
    "max_concurrent": 10,
    "model": "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
    "server_binary": "llama-server",
    "schedulers_tested": 5
  },
  "results": {
    "default": {...},
    "scx_rusty": {...},
    ...
  }
}
```

## Summary Report

The text summary now includes:
- System information (platform, CPU, memory, kernel)
- Test configuration (dataset, samples, concurrency)
- Timestamp
- Per-scheduler results

## Tips for Testing

1. **Start Small**: Test with fewer samples first
   ```bash
   python sharegpt_llama_server_eval.py --num-samples 10 --max-concurrent 2
   ```

2. **Monitor Resources**: The system info helps correlate performance with hardware
   
3. **Debug Failures**: Use `--server-logs` to capture server errors
   
4. **Compare Datasets**: Test both datasets to see performance differences
   ```bash
   # Test 1
   python sharegpt_llama_server_eval.py --dataset datasets/sharegpt_vicuna.json
   
   # Test 2  
   python sharegpt_llama_server_eval.py --dataset datasets/sharegpt_benchmark.json
   ```

5. **Reduce Concurrency**: If seeing many failures, reduce concurrent requests
   ```bash
   python sharegpt_llama_server_eval.py --max-concurrent 5
   ```

## Understanding System Info

The benchmark now captures system details using:
- `platform` module for OS/architecture info
- `psutil` for CPU/memory metrics
- `uname -a` for kernel details

This helps when comparing results across different systems or configurations.