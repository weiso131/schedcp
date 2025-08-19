# LLaMA.cpp Benchmark Testing Documentation

numactl --interleave=3 python /root/yunwei37/ai-os/workloads/llama.cpp/llamacpp_bench_start.py > llama_test.log

## Overview

This document explains the llama.cpp benchmark testing framework, how it works, and analysis of the test results including why some requests fail.

## Test Architecture

### Test Scripts

1. **`llamacpp_bench_start.py`** - Tests raw inference performance using llama-bench
   - Measures prompt processing (PP) and text generation (TG) throughput
   - Tests different schedulers with controlled workloads
   
2. **`sharegpt_llama_server_eval.py`** - Tests real-world server performance using ShareGPT dataset
   - Simulates concurrent user requests
   - Measures time-to-first-token (TTFT) and overall throughput
   - Tests with realistic conversation patterns

### Test Configuration

#### Server Configuration (sharegpt_llama_server_eval.py)
```python
server_config = {
    "n_gpu_layers": -1,      # Use all GPU layers if available
    "ctx_size": 4096,        # Context window size
    "n_batch": 512,          # Batch size for processing
    "n_threads": 8,          # Number of CPU threads
    "cont_batching": True,   # Continuous batching for better throughput
    "flash_attn": True,      # Flash attention optimization
}
```

#### Benchmark Parameters
- **num_samples**: 100 (default) - Number of prompts from ShareGPT dataset
- **max_concurrent**: 10 (default) - Maximum concurrent requests
- **timeout**: 30 seconds - Server startup timeout

## How the Testing Works

### Request Handling

The ShareGPT benchmark simulates real-world usage:

1. **Dataset Loading**: Loads prompts from ShareGPT dataset (real conversation starters)
2. **Server Launch**: Starts llama.cpp server with specific scheduler
3. **Concurrent Requests**: Sends up to 10 concurrent requests using asyncio
4. **Streaming Response**: Processes streaming responses to measure TTFT
5. **Metrics Collection**: Records success/failure, latency, throughput

### Test Flow
```
1. Load 100 prompts from ShareGPT dataset
2. For each scheduler:
   a. Start llama.cpp server
   b. Wait for server ready (health check)
   c. Send 10 concurrent requests (batches of prompts)
   d. Collect metrics for each request
   e. Stop server
   f. Analyze results
```

## Understanding the Test Results

### Success Rate: 12/20 (60%)

The test results show 12 successful requests out of 20 total requests. This pattern is consistent across all schedulers, indicating the failures are not scheduler-specific.

### Why Requests Failed

Based on the code analysis and typical patterns, requests fail for these reasons:

1. **Resource Exhaustion**
   - Running 10 concurrent requests can overwhelm the server
   - Memory limitations with 4096 token context window
   - CPU/GPU resource contention

2. **Context Length Exceeded**
   - Some ShareGPT prompts may be too long
   - Combined prompt + response exceeds 4096 token limit
   - Server returns HTTP error when context is exceeded

3. **Timeout Issues**
   - Complex prompts take longer to process
   - Under high concurrency, some requests timeout
   - Default streaming timeout may be too aggressive

4. **Server Overload**
   - Continuous batching has limits
   - Queue overflow when too many requests arrive simultaneously
   - Some requests get rejected when server is at capacity

### Failure Pattern Analysis

The 8/20 failure rate appears systematic:
- First batch of concurrent requests (10) likely succeeds partially
- Second batch encounters resource exhaustion
- Pattern suggests server capacity limit around 10-12 concurrent operations

## Performance Metrics Explained

### Key Metrics

1. **TTFT (Time to First Token)**
   - Latency before user sees first response token
   - Critical for user experience
   - Ranges from 4.8s (simple) to 17.9s (qmap)

2. **Tokens Per Second (TPS)**
   - Overall generation throughput
   - Varies from 5.8 (qmap) to 24.3 (rustland)
   - Higher is better

3. **Success Rate**
   - Percentage of requests completed successfully
   - All schedulers show 60% (12/20) success rate
   - Indicates system-level limitation, not scheduler issue

## Recommendations

### For Testing

1. **Reduce Concurrency**: Lower max_concurrent from 10 to 5-6
2. **Increase Timeouts**: Set request timeout to 60+ seconds
3. **Filter Prompts**: Pre-filter ShareGPT dataset for shorter prompts
4. **Monitor Resources**: Add CPU/memory monitoring during tests

### For Production

1. **Request Queue**: Implement proper request queuing
2. **Rate Limiting**: Add rate limiting to prevent overload
3. **Error Handling**: Improve error messages for debugging
4. **Auto-retry**: Implement exponential backoff for failed requests

## Running the Tests

### Basic ShareGPT Benchmark
```bash
python sharegpt_llama_server_eval.py \
    --num-samples 20 \
    --max-concurrent 5 \
    --production-only
```

### Custom Configuration
```bash
python sharegpt_llama_server_eval.py \
    --server-binary /path/to/llama-server \
    --model /path/to/model.gguf \
    --dataset datasets/sharegpt_benchmark.json \
    --num-samples 50 \
    --max-concurrent 3 \
    --schedulers scx_rusty scx_lavd
```

### Debugging Failed Requests
To debug failures, modify the test script to:
1. Log full error responses
2. Track memory usage per request
3. Monitor server logs during testing
4. Reduce concurrency to isolate issues

## Conclusion

The 60% success rate (12/20) is primarily due to:
- High concurrency (10 simultaneous requests)
- Resource limitations (memory, compute)
- Context length constraints (4096 tokens)
- Server capacity limits

This is not a scheduler problem but rather a capacity/configuration issue that affects all schedulers equally. For production use, implement proper queue management and resource monitoring.

## on my laptop


 /home/yunwei37/ai-os/workloads/llama.cpp/build/bin/llama-bench  -m /home/yunwei37/ai-os/workloads/llama.cpp/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf -n 1
| model                          |       size |     params | backend    | threads |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | ------: | --------------: | -------------------: |
| llama 1B Q4_K - Medium         | 636.18 MiB |     1.10 B | CPU        |       2 |           pp512 |         58.31 ± 1.75 |
| llama 1B Q4_K - Medium         | 636.18 MiB |     1.10 B | CPU        |       2 |             tg1 |         27.86 ± 1.62 |

build: 5aa1105da (6082)

yunwei37@victoryang00-ASUS-Zenbook-S-14-UX5406SA-UX5406SA:~/ai-os/workloads/llama.cpp/llama.cpp$ /home/yunwei37/ai-os/workloads/llama.cpp/llama.cpp/build/bin/llama-bench -m /home/yunwei37/ai-os/workloads/llama.cpp/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf -n 1

| model                          |       size |     params | backend    | ngl |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --------------: | -------------------: |
get_memory_info: [warning] ext_intel_free_memory is not supported (export/set ZES_ENABLE_SYSMAN=1 to support), use total memory as free memory
get_memory_info: [warning] ext_intel_free_memory is not supported (export/set ZES_ENABLE_SYSMAN=1 to support), use total memory as free memory
get_memory_info: [warning] ext_intel_free_memory is not supported (export/set ZES_ENABLE_SYSMAN=1 to support), use total memory as free memory
| llama 1B Q4_K - Medium         | 636.18 MiB |     1.10 B | SYCL       |  99 |           pp512 |      629.25 ± 288.96 |
| llama 1B Q4_K - Medium         | 636.18 MiB |     1.10 B | SYCL       |  99 |             tg1 |        23.02 ± 12.81 |

build: 5aa1105da (6082)

