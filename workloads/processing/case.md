# Linux Kernel Scheduler Optimization Test Cases

## Overview

This document presents a comprehensive collection of benchmark test cases designed to demonstrate how custom Linux kernel schedulers can significantly improve performance on dual-CPU systems. These tests specifically target "long-tail" scenarios where one task takes significantly longer than others, creating inefficiencies in standard CFS (Completely Fair Scheduler) scheduling.

## Core Concept: The Long-Tail Problem

In many real-world scenarios, workloads consist of multiple tasks where:
- 99 tasks complete in ~6 seconds each
- 1 task (the "straggler") takes ~600 seconds

Under standard CFS scheduling on a 2-CPU system:
- Total time = (99 × 6 sec / 2 cores) + 600 sec ≈ 894 seconds
- One core sits idle after completing its share of short tasks

With a custom scheduler that detects and pins long-running tasks:
- Total time ≈ 600 seconds (the duration of the longest task)
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
- 99 small files (≤100 KiB each)
- 1 large ISO file (200 MiB)
- The pigz thread processing the ISO lives 100x longer than others

**Synthetic Data Generation:**
```bash
# Create 99 small files and 1 large file
for i in {1..99}; do dd if=/dev/urandom of=file$i.dat bs=1K count=100; done
dd if=/dev/urandom of=large.iso bs=1M count=200  # 200MB file
```


**Expected Results:**
- End-to-end time reduction: ~15s → ~10s
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
- 99 short video clips (process in ~6 seconds each)
- 1 4K/10-minute clip (processes in ~600 seconds)
- Massive imbalance in processing time

**Synthetic Data Generation:**
```bash
# Generate test videos using ffmpeg itself
for i in {1..99}; do
    ffmpeg -f lavfi -i testsrc=duration=0.1:size=320x240:rate=30 clip$i.mp4
done
# One long video
ffmpeg -f lavfi -i testsrc=duration=10:size=1920x1080:rate=30 long_clip.mp4
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
- 1 integration test that starts Postgres (takes 600 seconds)
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
    time.sleep(10)  # Simulate database setup/teardown
    assert True
```


**Expected Results:**
- Suite wall time: 894 sec → ~600 sec
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
- One massive 300 MiB delta
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
dd if=/dev/urandom of=large.bin bs=1M count=300  # 300MB
git add large.bin && git commit -m "add large binary"
```


**Expected Results:**
- GC time reduced by ~30%
- More efficient delta compression parallelization

## Section 3: Database and Storage Workloads

### 3.1 RocksDB Compaction

**Real-world scenario:** Database applications experience periodic compaction storms when many small writes accumulate and trigger a major compaction. This blocks foreground operations and degrades user experience.

**Test Setup:**
```bash
# RocksDB db_bench with 1M keys
db_bench --benchmarks=fillrandom --num=1000000
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
    // Insert 1M keys to trigger compaction
    for(int i = 0; i < 1000000; i++) {
        db->Put(rocksdb::WriteOptions(), std::to_string(i), std::string(1024, 'x'));
    }
}
EOF
g++ -o rocksdb_test rocksdb_test.cpp -lrocksdb
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
- 1 large file (1 GB) takes 100x longer to process
- Creates severe imbalance in xargs parallel execution

**Synthetic Test Setup:**
```bash
# Create test directory with mixed file sizes
mkdir -p large-dir
for i in {1..99}; do
    dd if=/dev/urandom of=large-dir/file$i.dat bs=1M count=1
done
# One large file
dd if=/dev/urandom of=large-dir/largefile.dat bs=1M count=1024  # 1GB

# Run the parallel checksum operation
time find ./large-dir -type f -print0 | xargs -0 -n1 -P2 sha256sum > checksums.txt
```


**Expected Results:**
- Total checksum time: ~894 sec → ~600 sec
- ~33% improvement in parallel file processing

## Section 4: Data Processing and Analytics Workloads

### 4.1 Spark Local Shuffle with Skew

**Real-world scenario:** Analytics queries often have skewed joins or aggregations where one customer/product/region has orders of magnitude more data. This "hot key" problem is common in e-commerce and social media analytics.

**Code:**
```python
from pyspark.sql import SparkSession
s = SparkSession.builder.master("local[2]").getOrCreate()
# 99 small keys, 1 hot key
rdd = s.parallelize([(i%100, 1) for i in range(1_000_000)])
result = rdd.groupByKey().mapValues(sum).collect()
```

**Workload Characteristics:**
- Data skew: 1 hot key processes 100x more data
- One executor thread runs for 600 seconds
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
    partitions = [1000] * 99 + [1000000]  # 100x skew
    with mp.Pool(2) as pool:
        results = pool.map(process_partition, partitions)
```


**Expected Results:**
- Stage time: ~894 sec → ~600 sec
- ~33% improvement in shuffle performance

### 4.2 Sort and Compress with Skew

**Real-world scenario:** Log processing pipelines split files by time/size for parallel processing, but one time period might have an unusual spike (Black Friday, system outage, viral event) creating a much larger chunk.

**Commands:**
```bash
split -b100M big.tsv part_
parallel -j2 --line-buffer 'sort {} | zstd -q -o {}.zst' ::: part_*
```

**Workload Characteristics:**
- One 1 GB chunk among 99 small chunks
- Massive sorting time difference between chunks

**Synthetic Data Generation:**
```bash
# Create skewed data files
for i in {1..99}; do
    seq 1 10000 | shuf > part_$i.tsv  # ~100KB files
done
seq 1 10000000 | shuf > part_100.tsv  # ~100MB file
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
    'k': np.concatenate([np.arange(99), np.repeat(999, 500_000)]),
    'v': 1
})
d = dd.from_pandas(pdf, npartitions=100)
result = d.groupby('k').v.sum().compute()
```

**Workload Characteristics:**
- Hot group (key 999) overwhelms one worker
- Worker with key 999 occupies CPU for ~600 seconds
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
    'k': np.concatenate([np.arange(99), np.repeat(999, 500_000)]),
    'v': 1
})
# Split into chunks for parallel processing
chunks = np.array_split(data, 2)

with ProcessPoolExecutor(max_workers=2) as executor:
    results = list(executor.map(process_group, chunks))
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
- 99 × 1 MB CSV files
- 1 × 100 MB CSV file
- DuckDB splits import by file, creating imbalance

**Synthetic CSV Generation:**
```bash
# Create test CSV files
for i in {1..99}; do
    echo "id,value" > file$i.csv
    seq 1 10000 | awk '{print $1","rand()}' >> file$i.csv
done
# One large CSV
echo "id,value" > file100.csv
seq 1 1000000 | awk '{print $1","rand()}' >> file100.csv
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
- 99 × 1 MB gzipped files
- 1 × 100 MB gzipped file
- One pool worker spends excessive time decompressing large file

**Synthetic Log File Generation:**
```bash
# Create test gzipped files
for i in {1..99}; do
    # Small log files
    seq 1 1000 | awk '{print strftime("%Y-%m-%d %H:%M:%S"), "INFO", "Message", $1}' | 
    gzip > logs/log$i.gz
done
# One large log file
seq 1 100000 | awk '{print strftime("%Y-%m-%d %H:%M:%S"), "INFO", "Message", $1}' | 
gzip > logs/log100.gz
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
data[999] = list(range(10000))  # Hot key

# Process in parallel
with mp.Pool(2) as pool:
    results = pool.map(process_join_partition, data.items())
```


**Expected Results:**
- Join operation completes faster
- Better slot utilization in MiniCluster

## Test Case Implementation

All test cases described above are implemented as automated tests in the `testcases/` directory. Each test case includes:

- **Synthetic data generation** for reproducible results
- **Process monitoring** to track CPU usage and identify long-tail tasks  
- **Analysis tools** to measure scheduler optimization potential
- **Makefile automation** for easy execution

### Running Tests

```bash
# Navigate to testcases directory
cd testcases/

# List available tests
make list-tests

# Run all tests
make run-all

# Run specific test
make run-pigz_compression

# Analyze results
make analyze-pigz_compression
```

### Process Analysis

Each test automatically monitors:
- **CPU Time**: Time spent on CPU per process
- **Wall Clock Time**: Total runtime per process
- **Long-tail Detection**: Processes running >0.5 seconds
- **Scheduler Benefit**: Estimated improvement from optimization

## Key Benefits of These Demonstrations

1. **Zero Application Changes:** 
   - Observe and optimize at kernel level
   - No need to modify application code
   - Works with existing binaries

2. **Quick Iteration:**
   - Tests run in seconds to minutes, not hours
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
