# Scheduler Performance Testing with schbench

This directory contains tools for testing all schedulers in the SchedCP project using schbench.

## Setup

1. Build schbench:
   ```bash
   cd ../schbench
   make
   ```

2. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

Basic usage:
```bash
./test_schedulers.py
```

Test only production schedulers:
```bash
./test_schedulers.py --production-only
```

Customize test parameters:
```bash
./test_schedulers.py --runtime 60 --message-threads 4 --message-groups 8
```

## Output

The script generates:
- Performance comparison figures (PNG)
- JSON results file with detailed metrics
- Console summary of all scheduler performance

## Test Parameters

- **runtime**: Test duration in seconds (default: 30)
- **message-threads**: Number of message threads (default: 2)
- **message-groups**: Number of message groups (default: 4)
- **sleeptime**: Sleep time in microseconds (default: 10000)
- **cputime**: CPU time in microseconds (default: 10000)

## Schedulers Tested

The script tests all available schedulers in the SchedCP project, including:
- scx_simple, scx_rusty, scx_bpfland, scx_flash
- scx_lavd, scx_layered, scx_nest, scx_p2dq
- scx_flatcg, and more

Results show latency percentiles, throughput, and combined performance scores.