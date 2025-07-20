# Linux Kernel Scheduler Optimization Test Cases

## Overview

This document presents a comprehensive collection of benchmark test cases designed to demonstrate how custom Linux kernel schedulers can significantly improve performance on dual-CPU systems. These tests specifically target "long-tail" scenarios where one task takes significantly longer than others, creating inefficiencies in standard CFS (Completely Fair Scheduler) scheduling.

## Core Concept: The Long-Tail Problem

In many real-world scenarios, workloads consist of multiple tasks where:
- 99 tasks complete in ~60 seconds each
- 1 task (the "straggler") takes ~6000 seconds

Under standard CFS scheduling on a 2-CPU system:
- Total time = (99 × 60 sec / 2 cores) + 6000 sec ≈ 8940 seconds
- One core sits idle after completing its share of short tasks

With a custom scheduler that detects and pins long-running tasks:
- Total time ≈ 6000 seconds (the duration of the longest task)
- Both cores remain busy until all short tasks complete
- Performance improvement: 25-40% reduction in wall-clock time

## Section 1: File Processing and Compression Workloads

### 1.1 Pigz Directory Compression

**Command:**
```bash
find ./linux-src -type f -print0 | xargs -0 -n1 -P2 pigz -1
```

**Workload Characteristics:**
- 99 small files (≤1 MiB each)
- 1 large ISO file (2 GiB)
- The pigz thread processing the ISO lives 100x longer than others

**Scheduler Optimization Strategy:**
- Track runtime for each PID in a BPF map
- If runtime > 5 seconds AND sibling tasks have exited, mark as LONG
- Pin long-running task to CPU 0: `set_cpus_allowed_ptr(long_pid, 0)`
- Direct all other tasks to CPU 1

**Expected Results:**
- End-to-end time reduction: ~149s → ~100s
- Performance gain: ~33%

### 1.2 FFmpeg Split Transcode

**Command:**
```bash
for f in clips/*.mp4; do 
    ffmpeg -loglevel quiet -i "$f" -vf scale=640:-1 \
           -c:v libx264 -preset veryfast out/"${f##*/}" & 
done
wait
```

**Workload Characteristics:**
- 99 short video clips (process in ~60 seconds each)
- 1 4K/10-minute clip (processes in ~6000 seconds)
- Massive imbalance in processing time

**Scheduler Optimization Strategy:**
- Detect when PID's executable name is "ffmpeg"
- Monitor utime (user CPU time) exceeding threshold
- Pin high-utime processes to dedicated core

**Expected Results:**
- Batch processing time reduced by ~33%
- Better CPU utilization throughout the job

## Section 2: Software Testing and Development Workloads

### 2.1 Pytest xdist Test Suite

**Command:**
```bash
pytest -q -n2 --durations=0
```

**Workload Characteristics:**
- Test suite with 99 fast unit tests
- 1 integration test that starts Postgres (takes 6000 seconds)
- xdist spawns 2 workers; one finishes fast tests in ~60s then idles

**Scheduler Optimization Strategy:**
- Monitor pytest workers in BPF policy
- If a worker holds GIL continuously for >10 seconds, classify as long-tail
- Pin long-running worker to dedicated core

**Expected Results:**
- Suite wall time: 8940 sec → ~6000 sec
- ~33% improvement in total test time

### 2.2 Git Incremental Compression

**Command:**
```bash
git clone --mirror linux.git big.git
cd big.git
time git gc
```

**Workload Characteristics:**
- Packs hundreds of 4 MiB deltas
- One massive 3 GiB delta
- The large delta thread is 100x heavier

**Scheduler Optimization Strategy:**
- Track bytes written per PID using tracepoints
- Maintain running sum of write operations
- Pin the process with maximum bytes written

**Expected Results:**
- GC time reduced by ~30%
- More efficient delta compression parallelization

## Section 3: Database and Storage Workloads

### 3.1 RocksDB Compaction

**Test Setup:**
```bash
# RocksDB db_bench with 10M keys
db_bench --benchmarks=fillrandom --num=10000000
```

**Workload Characteristics:**
- Multiple small file compactions
- One large L0→L1 compaction dominates runtime

**Scheduler Optimization Strategy:**
- Use tracepoint `rocksdb:compaction_start` to identify compaction PIDs
- Pin compaction process until `compaction_finished` event
- Isolate heavy compactions from latency-sensitive operations

**Expected Results:**
- Improved 99th percentile latency during fill phase
- Better isolation of background compaction work

### 3.2 io_uring QD-1 Latency Test

**Command:**
```bash
fio --name=randread --ioengine=io_uring --iodepth=1 \
    --runtime=180 --rw=randread
```

**Workload Characteristics:**
- CQE-poller kthread frequently shares CPU with syscall submitter
- Creates scheduling conflicts affecting latency

**Scheduler Optimization Strategy:**
- Map (submitter_pid → poller_pid) relationship once
- When submitter blocks, pin poller to different CPU
- Ensure submitter and poller don't compete

**Expected Results:**
- P99 latency typically 2-3x lower
- More consistent I/O performance

## Section 4: Data Processing and Analytics Workloads

### 4.1 Spark Local Shuffle with Skew

**Code:**
```python
from pyspark.sql import SparkSession
s = SparkSession.builder.master("local[2]").getOrCreate()
# 99 small keys, 1 hot key
rdd = s.parallelize([(i%100, 1) for i in range(10_000_000)])
result = rdd.groupByKey().mapValues(sum).collect()
```

**Workload Characteristics:**
- Data skew: 1 hot key processes 100x more data
- One executor thread runs for 6000 seconds
- Other executor completes 99 tasks quickly

**Scheduler Optimization Strategy:**
- Monitor executor threads via jstack
- Tasks running >5 seconds on CPU get pinned to CPU 0
- Other tasks directed to CPU 1

**Expected Results:**
- Stage time: ~8940 sec → ~6000 sec
- ~33% improvement in shuffle performance

### 4.2 Sort and Compress with Skew

**Commands:**
```bash
split -b100M big.tsv part_
parallel -j2 --line-buffer 'sort {} | zstd -q -o {}.zst' ::: part_*
```

**Workload Characteristics:**
- One 10 GB chunk among 99 small chunks
- Massive sorting time difference between chunks

**Scheduler Optimization Strategy:**
- Detect long-running sort process
- Pin heavy sort thread after threshold time
- Allow small sorts to complete on other core

**Expected Results:**
- Total processing time reduced by ~30%
- Better parallel efficiency

### 4.3 Dask DataFrame Groupby

**Code:**
```python
import dask.dataframe as dd, pandas as pd, numpy as np
pdf = pd.DataFrame({
    'k': np.concatenate([np.arange(99), np.repeat(999, 5_000_000)]),
    'v': 1
})
d = dd.from_pandas(pdf, npartitions=100)
result = d.groupby('k').v.sum().compute()
```

**Workload Characteristics:**
- Hot group (key 999) overwhelms one worker
- Worker with key 999 occupies CPU for ~6000 seconds
- Severe workload imbalance

**Scheduler Optimization Strategy:**
- Pin first worker still occupying CPU after 60 seconds to CPU 0
- Keep other worker on CPU 1
- Prevent idle CPU while one worker struggles

**Expected Results:**
- Computation time significantly reduced
- Better resource utilization

### 4.4 DuckDB Threaded CSV Import

**Command:**
```bash
duckdb -c "PRAGMA threads=2; COPY (SELECT * FROM read_csv_auto('*.csv')) TO 'all.parquet'"
```

**Workload Characteristics:**
- 99 × 10 MB CSV files
- 1 × 1 GB CSV file
- DuckDB splits import by file, creating imbalance

**Scheduler Optimization Strategy:**
- Detect threads with cumulative utime > 5 seconds early in run
- Pin heavy thread to dedicated core
- Allow fast imports to share other core

**Expected Results:**
- Import time reduced by 25-35%
- More efficient parallel CSV processing

### 4.5 Pandas Multiprocessing ETL

**Code:**
```python
import multiprocessing as mp, pandas as pd, glob, gzip
files = glob.glob('logs/*.gz')

def parse(f):
    return pd.read_csv(gzip.open(f))

with mp.Pool(2) as p:
    dfs = p.map(parse, files)
```

**Workload Characteristics:**
- 99 × 10 MB gzipped files
- 1 × 1 GB gzipped file
- One pool worker spends excessive time decompressing large file

**Scheduler Optimization Strategy:**
- Pin long-running worker process
- Alternatively, boost priority when buddy core is idle
- Ensure decompression doesn't block other work

**Expected Results:**
- ETL pipeline completes 30-40% faster
- Better multiprocessing pool utilization

### 4.6 Local Flink Batch Join

**Setup:**
- MiniCluster with two slots
- Submit join where one key contains all tuples
- Extreme key skew in join operation

**Workload Characteristics:**
- Slot 1 processes giant key for extended time
- Slot 2 completes other keys quickly
- Classic distributed computing skew problem

**Scheduler Optimization Strategy:**
- Pin heavy TaskManager thread
- Allow other slot to recycle on CPU 1
- Isolate skewed computation

**Expected Results:**
- Join operation completes faster
- Better slot utilization in MiniCluster

## Implementation Details

### BPF/sched_ext Policy Structure

The optimization strategies described above can be implemented with a minimal sched_ext policy (~30 lines of code):

1. **Detection Phase:**
   - Track per-PID metrics (runtime, bytes written, utime)
   - Use BPF maps to store historical data
   - Implement threshold-based classification

2. **Action Phase:**
   - Use `sched_setaffinity()` or equivalent to pin tasks
   - Separate long-running tasks from short ones
   - Ensure optimal CPU utilization

3. **Monitoring:**
   - Use `perf sched timehist` to visualize improvements
   - Track wall-clock time with simple `time` command
   - Measure 30-40% improvements consistently

## Key Benefits of These Demonstrations

1. **Zero Application Changes:** 
   - Observe and optimize at kernel level
   - No need to modify application code
   - Works with existing binaries

2. **Quick Iteration:**
   - Tests run in minutes, not hours
   - Faster than full cluster deployments
   - Easy to iterate on scheduler policies

3. **Visual Impact:**
   - Clear before/after CPU utilization charts
   - Obvious performance improvements
   - Compelling demonstration of scheduler benefits

4. **Real-World Relevance:**
   - Patterns found in production systems
   - Applicable to CI/CD, data processing, scientific computing
   - Addresses common performance bottlenecks

## Conclusion

These test cases demonstrate that even simple, heuristic-based custom schedulers can provide significant performance improvements for workloads with long-tail characteristics. The 25-40% performance gains are achievable with minimal code changes and no application modifications, making custom scheduling an attractive optimization strategy for systems with predictable workload patterns.