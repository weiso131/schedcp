# Log Processing with Skewed Chunks

**ID:** `log_processing`

**Category:** data_processing

**Description:** Processing log files with different sizes and compression

## Workload Purpose & Characteristics

This workload represents a typical log analysis pipeline using Unix command-line tools, simulating server log processing for IP address analytics. The scenario includes 39 processes analyzing 500K log entries and 1 process handling 7M entries (14:1 ratio), demonstrating the challenges of processing unevenly distributed log chunks in production systems.

## Key Performance Characteristics

- **Pipeline-based processing**: Multiple processes connected via pipes
- **Mixed workload types**: Compression (CPU), grep (I/O+CPU), sort (memory+CPU)
- **Memory pressure from sorting**: Large sorts require significant memory
- **I/O patterns**: Sequential reads, intermediate pipe I/O, final writes
- **Process coordination**: Pipeline stages must coordinate efficiently

## Optimization Goals

1. **Minimize pipeline completion time**: Reduce end-to-end processing time
2. **Optimize pipeline throughput**: Ensure efficient data flow between stages
3. **Prioritize large log processing**: Keep the 7M entry pipeline running smoothly
4. **Balance resource allocation**: Distribute CPU/memory across pipeline stages
5. **Prevent pipeline stalls**: Avoid bottlenecks at any processing stage

## Scheduling Algorithm

The optimal scheduler for log processing pipelines should implement:

1. **Pipeline detection**: Identify related processes (gzip, zcat, grep, awk, sort, uniq) as a group
2. **Size-based prioritization**: Detect large vs small pipelines by log file size or process names
3. **Stage-aware scheduling**:
   - Compression stages: 10ms slices for CPU-intensive work
   - Sort stages: 15ms slices with memory consideration
   - Other stages: 5ms slices for I/O operations
4. **Pipeline coherence**: Keep pipeline stages on nearby cores for efficient communication
5. **Memory-aware dispatch**: Consider available memory when scheduling sort operations

## Dependencies

- gzip
- grep
- awk
- sort
- uniq

## Small Setup Commands

```bash
mkdir -p log_chunks
seq 1 500000 | awk '{print strftime("%Y-%m-%d %H:%M:%S"), "[INFO]", "Request from", "192.168.1."int(rand()*255), "processed in", int(rand()*100), "ms"}' > log_chunks/small.log
```

## Large Setup Commands

```bash
mkdir -p log_chunks
seq 1 7000000 | awk '{print strftime("%Y-%m-%d %H:%M:%S"), "[INFO]", "Request from", "192.168.1."int(rand()*255), "processed in", int(rand()*100), "ms"}' > log_chunks/large.log
```

## Small Execution Commands

```bash
gzip -c log_chunks/small.log | zcat | grep -E '\[INFO\]' | awk '{print $4}' | sort | uniq -c | sort -nr > log_chunks/small_ips.txt
```

## Large Execution Commands

```bash
gzip -c log_chunks/large.log | zcat | grep -E '\[INFO\]' | awk '{print $4}' | sort | uniq -c | sort -nr > log_chunks/large_ips.txt
```

## Cleanup Commands

```bash
rm -rf log_chunks/
```
