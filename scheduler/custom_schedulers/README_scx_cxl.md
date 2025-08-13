# SCX_CXL: CXL Bandwidth-Aware Scheduler

A CXL PMU-aware scheduler with DAMON integration for MoE VectorDB workloads, implementing intelligent bandwidth control and memory access pattern optimization.

## Features

### Core Capabilities
- **CXL PMU Metrics Integration**: Real-time monitoring of memory bandwidth, latency, and cache hit rates
- **DAMON Memory Monitoring**: Dynamic memory access pattern tracking for intelligent scheduling decisions
- **Bandwidth-Aware Scheduling**: Token bucket algorithm for precise read/write bandwidth control
- **MoE VectorDB Optimization**: Specialized scheduling for machine learning and vector database workloads
- **Dynamic Priority Adjustment**: Automatic priority boosting based on workload characteristics

### Task Classification
The scheduler automatically identifies and optimizes for:
- **MoE VectorDB Tasks**: Python, faiss, milvus, weaviate processes
- **Bandwidth Test Tasks**: double_bandwidth, memtest, stress tools
- **Read-Intensive Tasks**: Workloads with >70MB/s read bandwidth
- **Write-Intensive Tasks**: Workloads with >70MB/s write bandwidth
- **Kworker Tasks**: Kernel worker threads with dynamic promotion
- **Regular Tasks**: Standard processes with baseline scheduling

### Dispatch Queues
- **VECTORDB_DSQ (ID: 3)**: Highest priority for ML/VectorDB workloads
- **READ_INTENSIVE_DSQ (ID: 1)**: Optimized for read-heavy operations
- **WRITE_INTENSIVE_DSQ (ID: 2)**: Optimized for write-heavy operations
- **FALLBACK_DSQ (ID: 0)**: Default queue for regular tasks

## Building

```bash
cd scheduler/scx/scheds/c
meson setup build --prefix=/usr
meson compile -C build
```

## Usage

### Basic Usage
```bash
# Run with default settings
sudo ./scx_cxl

# Run with custom bandwidth limits
sudo ./scx_cxl -r 800 -w 600

# Run with verbose output
sudo ./scx_cxl -v

# Run with all features disabled (minimal mode)
sudo ./scx_cxl -d -c -b
```

### Command Line Options
```
  -h, --help            Show help message
  -v, --verbose         Enable verbose output and statistics
  -s, --slice-us        Time slice in microseconds (default: 20000)
  -n, --nr-cpus         Number of CPUs to use (default: all)
  -r, --read-bw         Max read bandwidth in MB/s (default: 1000)
  -w, --write-bw        Max write bandwidth in MB/s (default: 1000)
  -d, --disable-damon   Disable DAMON integration
  -c, --disable-cxl     Disable CXL-aware CPU selection
  -b, --disable-bw-ctrl Disable bandwidth control
  -m, --monitor         Monitor interval in seconds (default: 1)
  -D, --damon-path      Path to DAMON sysfs (default: /sys/kernel/mm/damon/admin)
```

## Architecture

### Memory Access Pattern Tracking
```c
struct memory_access_pattern {
    u64 nr_accesses;         // Total number of memory accesses
    u64 avg_access_size;     // Average size per access
    u64 read_bytes;          // Total bytes read
    u64 write_bytes;         // Total bytes written
    u32 locality_score;      // 0-100, higher = better locality
    u32 working_set_size;    // Working set in KB
    enum io_pattern;         // READ_HEAVY, WRITE_HEAVY, MIXED
};
```

### CXL PMU Metrics
```c
struct cxl_pmu_metrics {
    u64 memory_bandwidth;    // Total bandwidth in MB/s
    u64 cache_hit_rate;      // Cache hit percentage (0-100)
    u64 memory_latency;      // Access latency in nanoseconds
    u64 cxl_utilization;     // CXL bus utilization (0-100)
    u64 read_bandwidth;      // Read bandwidth in MB/s
    u64 write_bandwidth;     // Write bandwidth in MB/s
};
```

### Priority Boost Algorithm
- **Bandwidth Test Tasks**: +30 priority
- **MoE VectorDB Tasks**: +25 priority
- **Latency Sensitive**: +25 priority
- **Promoted Kworkers**: +20 priority
- **Read/Write Intensive**: +15 priority (if >70MB/s)
- **High Locality Score**: +5 bonus (if >80)

### CPU Selection Strategy
The scheduler uses a scoring system for CXL-aware CPU selection:
1. Base score: 100 points
2. CXL-attached CPU bonus: +50 for VectorDB tasks
3. Read/Write optimization: +30 for matching workloads
4. Low memory latency: +20 if <100ns
5. High cache hit rate: +15 if >80%
6. Load balancing: -5 per active intensive task

### Bandwidth Control
Token bucket algorithm with:
- Configurable max read/write bandwidth
- 10ms refill interval
- Per-task token tracking
- Automatic throttling when quota exceeded

## Performance Tuning

### For AI/ML Workloads
```bash
# Optimize for vector database operations
sudo ./scx_cxl -r 1500 -w 800 -s 15000
```

### For Read-Heavy Workloads
```bash
# Maximize read bandwidth
sudo ./scx_cxl -r 2000 -w 500 -s 10000
```

### For Write-Heavy Workloads
```bash
# Maximize write bandwidth
sudo ./scx_cxl -r 500 -w 2000 -s 10000
```

### For Low Latency
```bash
# Minimize scheduling latency
sudo ./scx_cxl -s 5000 -n 8
```

## Testing

Run the test suite:
```bash
sudo ./test_scx_cxl.sh
```

## Monitoring

With verbose mode enabled, the scheduler provides real-time statistics:
- Total enqueues and dispatches
- VectorDB task count
- Bandwidth-limited task count
- DAMON update frequency
- CXL migration count

## Requirements

- Linux kernel 6.12+ with sched-ext support
- CONFIG_SCHED_CLASS_EXT=y
- CONFIG_DAMON_SYSFS=y (optional, for DAMON support)
- libbpf >= 1.2.2
- Root privileges for BPF operations

## Troubleshooting

### DAMON Not Available
If you see "DAMON not available", ensure your kernel is compiled with:
```
CONFIG_DAMON=y
CONFIG_DAMON_SYSFS=y
```

### BPF Load Failures
Check kernel logs for detailed error messages:
```bash
dmesg | grep -i bpf
```

### Performance Issues
1. Check CPU frequency scaling settings
2. Verify NUMA configuration
3. Monitor system memory pressure
4. Adjust bandwidth limits based on hardware capabilities

## Integration with ktransformers

This scheduler implements all functionality from the ktransformers eBPF module, providing:
- Bandwidth-aware scheduling for transformer inference
- Memory access pattern optimization for attention mechanisms
- CXL memory tier awareness for model loading
- Dynamic priority adjustment for inference threads

## License

GPL-2.0