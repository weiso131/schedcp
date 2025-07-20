# Long-tail Scheduler Test Cases

This directory contains test cases that demonstrate workloads with long-tail characteristics where custom scheduling can provide significant performance improvements.

## Overview

Each test case simulates a real-world scenario where:
- 99 tasks complete quickly (short-tail)
- 1 task takes significantly longer (long-tail)
- Standard scheduling leads to CPU underutilization
- Custom scheduling can improve end-to-end time by 25-40%

## Available Test Cases

### 1. Pigz Compression (`pigz_compression/`)

- **Scenario**: Backup systems compressing mixed-size files
- **Workload**: 99 small files (1MB) + 1 large file (2GB)
- **Tool**: pigz parallel compression
- **Expected**: Large file compression dominates runtime

### 2. FFmpeg Transcode (`ffmpeg_transcode/`)

- **Scenario**: Video platform processing mixed-length uploads
- **Workload**: 10 short videos (1s) + 1 long video (30s)
- **Tool**: ffmpeg video transcoding
- **Expected**: Long video transcoding dominates runtime

### 3. Spark Shuffle Simulation (`spark_shuffle/`)
- **Scenario**: Analytics with data skew (hot keys)
- **Workload**: 99 small partitions + 1 hot partition (100x larger)
- **Tool**: Python multiprocessing simulation
- **Expected**: Hot partition processing dominates runtime

## Quick Start

1. **Run all tests**:
   ```bash
   make run-all
   ```

2. **Run individual test**:
   ```bash
   make run-pigz_compression
   ```

3. **Analyze results**:
   ```bash
   make analyze-pigz_compression
   ```

4. **Clean up**:
   ```bash
   make clean
   ```

## Directory Structure

Each test case follows this structure:
```
testcase_name/
├── Makefile          # Build and run commands
├── run_test.sh       # Main test runner script
├── .gitignore        # Ignore generated data
├── data/             # Generated test data (ignored)
├── output/           # Test outputs (ignored)
└── results/          # Analysis results (ignored)
```

## Process Analysis

Each test case includes automatic process monitoring that tracks:
- **CPU Time**: Time spent on CPU per process
- **Wall Clock Time**: Total runtime per process
- **Process Hierarchy**: Parent-child relationships
- **Timeline**: When processes start/stop
- **Long-tail Detection**: Processes running >5 seconds

The analysis identifies:
- Which processes are long-tail vs short-tail
- Potential scheduler optimization benefits
- Concurrent process execution patterns

## Results Interpretation

After running a test, look for:

1. **Long-tail Processes**: Tasks taking >5 seconds CPU time
2. **Imbalance**: One task taking 10-100x longer than others
3. **Scheduler Benefit**: Estimated improvement from isolation
4. **Concurrency**: How many tasks run simultaneously

Example output:
```
Long-tail Analysis:
Total processes: 100
Long runners (>5s CPU): 1
Short runners (≤5s CPU): 99

Scheduler Optimization Potential:
Current estimated time: 149.5s
Optimized estimated time: 100.2s
Potential improvement: 33.0%
```

## Requirements

- **Python 3** with `psutil` package
- **Tools**: Depends on test case (pigz, ffmpeg, etc.)
- **System**: Linux with at least 2 CPU cores
- **Disk Space**: ~3GB for largest test data

## Adding New Test Cases

To add a new test case:

1. Create directory: `mkdir new_testcase/`
2. Add Makefile with standard targets: `generate-data`, `run-test`, `analyze`, `clean`
3. Create `.gitignore` for generated data
4. Use `../common/analyze.py` for process monitoring
5. Update main Makefile `TEST_CASES` variable

## Integration with Schedulers

These test cases are designed to work with:
- **sched_ext** custom schedulers
- **CPU affinity** tools (taskset, cgroups)
- **Container orchestrators** (Kubernetes, Docker)
- **Batch schedulers** (SLURM, PBS)

The process analysis helps validate scheduler effectiveness by showing before/after improvements in long-tail workload performance.