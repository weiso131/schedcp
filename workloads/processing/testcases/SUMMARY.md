# Complete Long-tail Scheduler Test Cases

## Overview

This directory contains 12 comprehensive test cases that demonstrate workloads with long-tail characteristics where custom scheduling can provide significant performance improvements.

## Test Cases Summary

| Test Case | Scenario | Workload Pattern | Long-tail Task |
|-----------|----------|------------------|----------------|
| **pigz_compression** | Backup systems | 99 small files (1MB) + 1 large file (2GB) | Large file compression |
| **ffmpeg_transcode** | Video platform | 10 short videos (1s) + 1 long video (30s) | Long video transcoding |
| **pytest_suite** | CI/CD pipelines | 99 fast tests + 1 slow integration test | Database setup test |
| **git_compression** | Repository maintenance | 99 small commits + 1 large binary blob | Large delta compression |
| **rocksdb_compaction** | Database operations | Many small SST files + 1 large compaction | L0→L1 compaction |
| **file_checksum** | Security scans | 99 small files (10MB) + 1 large file (1GB) | Large file checksum |
| **spark_shuffle** | Analytics | 99 small partitions + 1 hot key partition | Hot key processing |
| **sort_compress** | Log processing | 10 small files + 1 large file (1M lines) | Large file sort/compress |
| **dask_groupby** | Customer analytics | Normal customers + 1 major customer | Hot key aggregation |
| **duckdb_import** | Data warehousing | 10 normal CSVs + 1 month-end CSV | Large CSV import |
| **pandas_etl** | Log analysis | 10 small logs + 1 large log (100K lines) | Large log processing |
| **flink_join** | Retail analytics | Normal products + 1 popular product | Hot key joins |

## Key Features

### 🚀 **Easy Execution**
```bash
make run-all              # Run all tests
make run-pigz_compression # Run specific test  
make analyze-all          # Analyze all results
```

### 📊 **Comprehensive Analysis**
- Real-time process monitoring during test execution
- Long-tail vs short-tail task classification
- Scheduler optimization potential estimation
- CPU utilization and timeline analysis

### 🔧 **Realistic Workloads**
- Each test simulates real-world scenarios
- Configurable data sizes for different testing needs
- No complex dependencies - uses standard tools
- Reproducible synthetic data generation

### 📈 **Performance Insights**
- Identifies processes running >5 seconds (long-tail)
- Estimates 25-40% improvement potential from scheduling
- Shows CPU time vs wall clock time breakdown
- Demonstrates concurrency bottlenecks

## Quick Start

1. **Prerequisites**: Python 3, psutil package, standard tools (pigz, ffmpeg, etc.)

2. **Run a test**:
   ```bash
   cd pigz_compression/
   make run-test
   ```

3. **View results**:
   ```bash
   make analyze
   ```

## Expected Results Pattern

All tests follow the same pattern:
- **Current**: One long task blocks completion (~149 time units)
- **Optimized**: Long task isolated, short tasks parallel (~100 time units)  
- **Improvement**: ~33% reduction in end-to-end time

## Integration with Schedulers

These tests work with:
- **sched_ext custom schedulers** (BPF-based)
- **CPU affinity tools** (taskset, cgroups)
- **Container schedulers** (Kubernetes, Docker)
- **HPC schedulers** (SLURM, PBS)

## Directory Structure
```
testcases/
├── common/                 # Shared analysis tools
│   ├── analyze.py         # Process monitoring
│   └── analyze_results.py # Results analysis
├── pigz_compression/      # File compression test
├── ffmpeg_transcode/      # Video transcoding test
├── pytest_suite/         # Test suite execution
├── git_compression/       # Repository operations
├── rocksdb_compaction/    # Database compaction
├── file_checksum/         # Security scanning
├── spark_shuffle/         # Analytics workload
├── sort_compress/         # Log processing
├── dask_groupby/          # Customer analytics
├── duckdb_import/         # Data warehousing
├── pandas_etl/            # ETL pipelines
├── flink_join/            # Stream processing
├── Makefile              # Master build system
└── README.md             # Detailed documentation
```

Each test case directory contains:
- `Makefile` - Build and run automation
- `.gitignore` - Ignore generated data
- Generated scripts and data files

## Analysis Output Example

```
=== Process Runtime Analysis ===
PID      Command              CPU Time    Wall Time   Avg CPU%
--------------------------------------------------------------------
12345    pigz large.iso       95.23s      96.45s      98.7%
12346    pigz file001.dat     0.15s       0.18s       83.3%
12347    pigz file002.dat     0.14s       0.16s       87.5%
...

Long-tail Analysis:
Total processes: 100
Long runners (>5s CPU): 1
Short runners (≤5s CPU): 99

Scheduler Optimization Potential:
Current estimated time: 149.1s
Optimized estimated time: 100.2s
Potential improvement: 32.8%
```

This comprehensive test suite provides everything needed to evaluate and demonstrate the effectiveness of custom scheduling solutions for long-tail workloads.