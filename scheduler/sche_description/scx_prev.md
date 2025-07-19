# scx_prev

## Overview

scx_prev is a variation on scx_simple that implements an optimized CPU selection strategy. Unlike scx_simple and scx_rusty which prioritize finding fully idle cores, scx_prev prioritizes selecting an idle previous CPU. This seemingly simple change in CPU selection policy can significantly improve performance for certain workloads.

### Key Features

- **Previous CPU Priority**: Checks if the task's previous CPU is idle before searching for other idle CPUs
- **Simple and Efficient**: Minimal overhead scheduler focusing on a single optimization
- **Statistics Tracking**: Maintains statistics on CPU selection outcomes (prev_cpu hits, idle_cpu selections, failures)
- **Performance Optimized**: Specifically tuned for OLTP workloads on simple topology systems

### Architecture

The scheduler implements a straightforward CPU selection algorithm:
1. First checks if the task's previous CPU is idle using `scx_bpf_test_and_clear_cpu_idle()`
2. If not, searches for any idle CPU using `scx_bpf_pick_idle_cpu()`
3. Falls back to the previous CPU if no idle CPU is found
4. Inserts selected tasks into the local DSQ for immediate dispatch

This approach leverages CPU cache locality by preferring the previous CPU when possible, which can reduce cache misses and improve performance for workloads that benefit from cache warmth.

## Typical Use Case

scx_prev excels in environments with:
- **OLTP Workloads**: Particularly effective for Online Transaction Processing workloads
- **Simple Topology**: Best suited for non-NUMA systems with a single Last Level Cache (LLC)
- **Cache-Sensitive Applications**: Workloads that benefit from maintaining CPU cache warmth

The scheduler has been shown to outperform the in-kernel fair class scheduler, scx_simple, and scx_rusty on OLTP workloads when run on systems with simple topology.

## Production Ready?

While scx_prev has not been extensively tested in production environments, its similarity to scx_simple and focused optimization suggest it could be production-ready for specific workloads on hardware with simple topology. The scheduler's simplicity reduces the risk of unexpected behavior, making it a reasonable choice for environments matching its design criteria.

## Command Line Options

```
A variation on scx_simple with CPU selection that prioritizes an idle
previous CPU over finding a fully idle core.

See the top-level comment in .bpf.c for more details.

Usage: scx_prev [-i sec] [-v]

  -h            Display this help and exit
  -i            Sampling interval for statistics in seconds
  -v            Print libbpf debug messages
```