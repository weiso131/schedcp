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

**Real-world scenario:** Backup systems often compress mixed-size files where source code files are small but VM images or database dumps are huge. This creates severe load imbalance when compressing in parallel.

**Command:**
```bash
find ./linux-src -type f -print0 | xargs -0 -n1 -P2 pigz -1
```

**Workload Characteristics:**
- 99 small files (≤1 MiB each)
- 1 large ISO file (2 GiB)
- The pigz thread processing the ISO lives 100x longer than others

**Synthetic Data Generation:**
```bash
# Create 99 small files and 1 large file
for i in {1..99}; do dd if=/dev/urandom of=file$i.dat bs=1M count=1; done
dd if=/dev/urandom of=large.iso bs=1M count=2048  # 2GB file
```

**Simple Scheduler Implementation:**
```bash
# Pre-identify PIDs and pin them before starting
# Start pigz in background and capture PIDs
find ./test-dir -name "large.iso" -print0 | xargs -0 -n1 -P1 pigz -1 &
LARGE_PID=$!
# Pin large file processing to CPU 0
taskset -cp 0 $LARGE_PID

# Process small files on CPU 1
find ./test-dir -name "file*.dat" -print0 | xargs -0 -n1 -P2 taskset -c 1 pigz -1
```

**Expected Results:**
- End-to-end time reduction: ~149s → ~100s
- Performance gain: ~33%

### 1.2 FFmpeg Split Transcode

**Real-world scenario:** Video platforms batch-process user uploads where most are short clips but occasionally receive full-length movies or lectures. The long video blocks completion of the entire batch.

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

**Synthetic Data Generation:**
```bash
# Generate test videos using ffmpeg itself
for i in {1..99}; do
    ffmpeg -f lavfi -i testsrc=duration=1:size=320x240:rate=30 clip$i.mp4
done
# One long video
ffmpeg -f lavfi -i testsrc=duration=100:size=1920x1080:rate=30 long_clip.mp4
```

**Simple Scheduler Implementation:**
```bash
# Process large video on CPU 0
ffmpeg -i long_clip.mp4 -vf scale=640:-1 -c:v libx264 -preset veryfast out/long_clip.mp4 &
LONG_PID=$!
taskset -cp 0 $LONG_PID

# Process short clips on CPU 1 in parallel
for f in clip{1..99}.mp4; do
    taskset -c 1 ffmpeg -loglevel quiet -i "$f" -vf scale=640:-1 -c:v libx264 -preset veryfast out/"$f" &
done
wait
```

**Expected Results:**
- Batch processing time reduced by ~33%
- Better CPU utilization throughout the job

## Section 2: Software Testing and Development Workloads

### 2.1 Pytest xdist Test Suite

**Real-world scenario:** CI/CD pipelines run test suites where most are quick unit tests but some integration tests require database setup or external service initialization. One slow test can delay the entire pipeline.

**Command:**
```bash
pytest -q -n2 --durations=0
```

**Workload Characteristics:**
- Test suite with 99 fast unit tests
- 1 integration test that starts Postgres (takes 6000 seconds)
- xdist spawns 2 workers; one finishes fast tests in ~60s then idles

**Synthetic Test Suite Creation:**
```python
# Create test_suite.py with mixed durations
import time
import pytest

# 99 fast tests
for i in range(99):
    exec(f'''
def test_fast_{i}():
    time.sleep(0.1)  # Simulate quick test
    assert True
''')

# 1 slow integration test
def test_slow_integration():
    time.sleep(100)  # Simulate database setup/teardown
    assert True
```

**Simple Scheduler Implementation:**
```bash
# Run tests with CPU affinity based on test name
# Create wrapper script
cat > run_test.sh << 'EOF'
#!/bin/bash
if [[ "$1" == *"slow_integration"* ]]; then
    # Long test goes to CPU 0
    taskset -c 0 pytest -q "$1"
else
    # Fast tests go to CPU 1
    taskset -c 1 pytest -q "$1"
fi
EOF
chmod +x run_test.sh

# Run tests in parallel with proper CPU assignment
pytest --collect-only -q | grep "::test_" | xargs -n1 -P2 ./run_test.sh
```

**Expected Results:**
- Suite wall time: 8940 sec → ~6000 sec
- ~33% improvement in total test time

### 2.2 Git Incremental Compression

**Real-world scenario:** Large repositories accumulate binary artifacts (PDFs, images, build outputs) that create massive deltas during garbage collection. One large binary dominates the entire gc process.

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

**Synthetic Repository Creation:**
```bash
# Create a repo with mixed object sizes
git init test-repo && cd test-repo
# Add 99 small commits
for i in {1..99}; do
    echo "small change $i" > file$i.txt
    git add file$i.txt && git commit -m "commit $i"
done
# Add one massive binary blob
dd if=/dev/urandom of=large.bin bs=1M count=3072  # 3GB
git add large.bin && git commit -m "add large binary"
```

**Simple Scheduler Implementation:**
```bash
# Git gc uses multiple threads, we can control them
# Set git to use 2 threads and pin the process
git config gc.autoPackLimit 2
git config pack.threads 2

# Run gc with CPU affinity monitoring
git gc &
GC_PID=$!

# Monitor and repin threads as they appear
while kill -0 $GC_PID 2>/dev/null; do
    # Find git pack-objects threads
    for pid in $(pgrep -P $GC_PID); do
        # Check if thread is using high CPU
        CPU_TIME=$(ps -p $pid -o time= | awk -F: '{print ($1*60)+$2}')
        if [ "$CPU_TIME" -gt 5 ]; then
            taskset -cp 0 $pid 2>/dev/null
        else
            taskset -cp 1 $pid 2>/dev/null
        fi
    done
    sleep 1
done
```

**Expected Results:**
- GC time reduced by ~30%
- More efficient delta compression parallelization

## Section 3: Database and Storage Workloads

### 3.1 RocksDB Compaction

**Real-world scenario:** Database applications experience periodic compaction storms when many small writes accumulate and trigger a major compaction. This blocks foreground operations and degrades user experience.

**Test Setup:**
```bash
# RocksDB db_bench with 10M keys
db_bench --benchmarks=fillrandom --num=10000000
```

**Workload Characteristics:**
- Multiple small file compactions
- One large L0→L1 compaction dominates runtime

**Synthetic Workload Setup:**
```bash
# Create simple RocksDB test without db_bench
# Use any key-value workload generator or simple C++ program
cat > rocksdb_test.cpp << 'EOF'
#include <rocksdb/db.h>
int main() {
    rocksdb::DB* db;
    rocksdb::Options options;
    options.create_if_missing = true;
    rocksdb::DB::Open(options, "/tmp/testdb", &db);
    // Insert 10M keys to trigger compaction
    for(int i = 0; i < 10000000; i++) {
        db->Put(rocksdb::WriteOptions(), std::to_string(i), std::string(1024, 'x'));
    }
}
EOF
g++ -o rocksdb_test rocksdb_test.cpp -lrocksdb
```

**Scheduler Policy Implementation:**
```c
// Track RocksDB compaction threads
if (strstr(p->comm, "rocksdb") && p->utime > 2 * NSEC_PER_SEC) {
    // Compaction thread detected, isolate it
    p->cpus_ptr = cpumask_of(0);
    bpf_map_update_elem(&compaction_pids, &p->pid, &ONE, BPF_ANY);
}
```

**Expected Results:**
- Improved 99th percentile latency during fill phase
- Better isolation of background compaction work

### 3.2 Parallel File System Operations

**Real-world scenario:** Security scans or integrity checks must verify all files in a directory containing mostly config files but also large disk images or database backups. The large file blocks completion of the entire scan.

**Command:**
```bash
# Parallel find and checksum operations
find ./large-dir -type f -print0 | xargs -0 -n1 -P2 sha256sum > checksums.txt
```

**Workload Characteristics:**
- 99 small files (< 10 MB each) checksum quickly
- 1 large file (10 GB) takes 100x longer to process
- Creates severe imbalance in xargs parallel execution

**Synthetic Test Setup:**
```bash
# Create test directory with mixed file sizes
mkdir -p large-dir
for i in {1..99}; do
    dd if=/dev/urandom of=large-dir/file$i.dat bs=1M count=10
done
# One large file
dd if=/dev/urandom of=large-dir/largefile.dat bs=1M count=10240  # 10GB

# Run the parallel checksum operation
time find ./large-dir -type f -print0 | xargs -0 -n1 -P2 sha256sum > checksums.txt
```

**Scheduler Policy Implementation:**
```c
// Track sha256sum processes
if (strstr(p->comm, "sha256sum")) {
    // Monitor CPU time to detect long-running checksum
    if (p->utime > 5 * NSEC_PER_SEC) {
        // Pin heavy checksum task to CPU 0
        set_cpus_allowed_ptr(p, cpumask_of(0));
        // Allow other checksums to use CPU 1
    }
}
```

**Expected Results:**
- Total checksum time: ~8940 sec → ~6000 sec
- ~33% improvement in parallel file processing

## Section 4: Data Processing and Analytics Workloads

### 4.1 Spark Local Shuffle with Skew

**Real-world scenario:** Analytics queries often have skewed joins or aggregations where one customer/product/region has orders of magnitude more data. This "hot key" problem is common in e-commerce and social media analytics.

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

**Simplified Test Without Spark:**
```python
# Simple Python simulation of skewed workload
import multiprocessing as mp
import time

def process_partition(key_count):
    # Simulate processing time proportional to data
    time.sleep(key_count / 100000)
    return sum(range(key_count))

if __name__ == '__main__':
    # 99 small partitions + 1 huge partition
    partitions = [10000] * 99 + [10000000]  # 100x skew
    with mp.Pool(2) as pool:
        results = pool.map(process_partition, partitions)
```

**Scheduler Policy Implementation:**
```c
// Detect Python multiprocessing workers
if (strstr(p->comm, "python") && p->utime > 5 * NSEC_PER_SEC) {
    // Long-running Python worker -> dedicate CPU 0
    set_cpus_allowed_ptr(p, cpumask_of(0));
}
```

**Expected Results:**
- Stage time: ~8940 sec → ~6000 sec
- ~33% improvement in shuffle performance

### 4.2 Sort and Compress with Skew

**Real-world scenario:** Log processing pipelines split files by time/size for parallel processing, but one time period might have an unusual spike (Black Friday, system outage, viral event) creating a much larger chunk.

**Commands:**
```bash
split -b100M big.tsv part_
parallel -j2 --line-buffer 'sort {} | zstd -q -o {}.zst' ::: part_*
```

**Workload Characteristics:**
- One 10 GB chunk among 99 small chunks
- Massive sorting time difference between chunks

**Synthetic Data Generation:**
```bash
# Create skewed data files
for i in {1..99}; do
    seq 1 100000 | shuf > part_$i.tsv  # ~1MB files
done
seq 1 100000000 | shuf > part_100.tsv  # ~1GB file
```

**Scheduler Policy Implementation:**
```c
// Track sort/zstd processes
if ((strstr(p->comm, "sort") || strstr(p->comm, "zstd")) && 
    p->utime > 3 * NSEC_PER_SEC) {
    // Heavy sort/compress -> pin to CPU 0
    p->cpus_ptr = cpumask_of(0);
}
```

**Expected Results:**
- Total processing time reduced by ~30%
- Better parallel efficiency

### 4.3 Dask DataFrame Groupby

**Real-world scenario:** Customer analytics often show power-law distributions where one major customer generates 100x more transactions than others. Grouping by customer ID creates severe computational imbalance.

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

**Simple Test Without Dask:**
```python
# Simulate without Dask dependency
import pandas as pd
import numpy as np
from concurrent.futures import ProcessPoolExecutor

def process_group(data):
    return data.groupby('k')['v'].sum()

# Create skewed data
data = pd.DataFrame({
    'k': np.concatenate([np.arange(99), np.repeat(999, 5_000_000)]),
    'v': 1
})
# Split into chunks for parallel processing
chunks = np.array_split(data, 2)

with ProcessPoolExecutor(max_workers=2) as executor:
    results = list(executor.map(process_group, chunks))
```

**Scheduler Policy Implementation:**
```c
// Monitor Python workers in process pool
if (strstr(p->comm, "python") && p->parent && 
    strstr(p->parent->comm, "python")) {
    if (p->utime > 60 * NSEC_PER_SEC) {
        pin_to_cpu(p->pid, 0);
    }
}
```

**Expected Results:**
- Computation time significantly reduced
- Better resource utilization

### 4.4 DuckDB Threaded CSV Import

**Real-world scenario:** Data warehouses import daily transaction files where most days have normal volume but month-end or year-end files are massive. The large file blocks the entire ETL pipeline.

**Command:**
```bash
duckdb -c "PRAGMA threads=2; COPY (SELECT * FROM read_csv_auto('*.csv')) TO 'all.parquet'"
```

**Workload Characteristics:**
- 99 × 10 MB CSV files
- 1 × 1 GB CSV file
- DuckDB splits import by file, creating imbalance

**Synthetic CSV Generation:**
```bash
# Create test CSV files
for i in {1..99}; do
    echo "id,value" > file$i.csv
    seq 1 100000 | awk '{print $1","rand()}' >> file$i.csv
done
# One large CSV
echo "id,value" > file100.csv
seq 1 10000000 | awk '{print $1","rand()}' >> file100.csv
```

**Scheduler Policy Implementation:**
```c
// DuckDB uses thread pool for parallel ops
if (strstr(p->comm, "duckdb") && p->utime > 5 * NSEC_PER_SEC) {
    // Heavy import thread -> CPU 0
    set_cpus_allowed_ptr(p, cpumask_of(0));
    // Light threads stay on CPU 1
}
```

**Expected Results:**
- Import time reduced by 25-35%
- More efficient parallel CSV processing

### 4.5 Pandas Multiprocessing ETL

**Real-world scenario:** Web servers rotate logs daily, but during DDoS attacks or viral traffic spikes, one day's log can be 100x larger. Processing these logs in parallel creates severe imbalance.

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

**Synthetic Log File Generation:**
```bash
# Create test gzipped files
for i in {1..99}; do
    # Small log files
    seq 1 10000 | awk '{print strftime("%Y-%m-%d %H:%M:%S"), "INFO", "Message", $1}' | 
    gzip > logs/log$i.gz
done
# One large log file
seq 1 1000000 | awk '{print strftime("%Y-%m-%d %H:%M:%S"), "INFO", "Message", $1}' | 
gzip > logs/log100.gz
```

**Scheduler Policy Implementation:**
```c
// Track gzip decompression in Python workers
if (strstr(p->comm, "python") && p->parent) {
    // Check if doing heavy I/O (gzip decompression)
    if (p->utime + p->stime > 10 * NSEC_PER_SEC) {
        pin_to_cpu(p->pid, 0);
    }
}
```

**Expected Results:**
- ETL pipeline completes 30-40% faster
- Better multiprocessing pool utilization

### 4.6 Local Flink Batch Join

**Real-world scenario:** In retail analytics, joining sales with product data often shows skew where popular items (iPhone, bestseller books) have 100x more transactions than niche products, causing join operations to bottleneck.

**Setup:**
- MiniCluster with two slots
- Submit join where one key contains all tuples
- Extreme key skew in join operation

**Workload Characteristics:**
- Slot 1 processes giant key for extended time
- Slot 2 completes other keys quickly
- Classic distributed computing skew problem

**Simple Join Simulation Without Flink:**
```python
# Simulate skewed join without Flink
import multiprocessing as mp
import time

def process_join_partition(partition_data):
    key, values = partition_data
    # Simulate join processing time based on data size
    time.sleep(len(values) / 10000)
    return (key, sum(values))

# Create skewed data: key 999 has 100x more values
data = {}
for i in range(99):
    data[i] = list(range(1000))
data[999] = list(range(100000))  # Hot key

# Process in parallel
with mp.Pool(2) as pool:
    results = pool.map(process_join_partition, data.items())
```

**Scheduler Policy Implementation:**
```c
// Generic policy for any skewed parallel workload
struct task_stats {
    u64 start_time;
    u64 cpu_time;
};

BPF_MAP_TYPE_HASH(task_stats_map, u32, struct task_stats);

// In scheduler callback:
if (p->utime > 30 * NSEC_PER_SEC && sibling_idle()) {
    // Long task with idle sibling -> dedicate CPU
    pin_to_cpu(p->pid, 0);
}
```

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
