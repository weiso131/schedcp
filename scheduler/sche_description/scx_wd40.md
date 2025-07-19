# scx_wd40

## Overview

scx_wd40 is an experimental fork of the scx_rusty scheduler that uses BPF arenas to simplify scheduler development. It's a multi-domain, BPF/userspace hybrid scheduler where the BPF portion does simple round robin in each domain, and the userspace portion (written in Rust) calculates the load factor of each domain and informs BPF of how tasks should be load balanced accordingly.

## Description

scx_wd40 represents a significant architectural shift in sched_ext scheduler design by leveraging BPF arenas for direct memory sharing between kernel and userspace components. This experimental scheduler is based on scx_rusty but reimplements many components to demonstrate modular scheduler design patterns.

The scheduler architecture consists of:

1. **BPF Arena-based Shared Memory**: Direct memory sharing between BPF and userspace eliminates complex data synchronization
2. **Multi-Domain Structure**: CPUs are organized into domains based on cache topology (LLC by default)
3. **Hybrid Scheduling**: Simple BPF-side scheduling with sophisticated userspace load balancing
4. **Modular Components**: Separate modules for placement, deadline tracking, and load balancing

Key implementation features:
- Uses BPF arenas to share complex data structures directly between kernel and userspace
- Implements deadline-based task ordering within domains
- Provides greedy work stealing both within and across NUMA nodes
- Separates concerns between fast-path BPF operations and complex userspace decisions

## Features

- **BPF Arena Memory Sharing**: Revolutionary use of BPF arenas for zero-copy data sharing
- **Modular BPF Design**: Separate BPF programs for different scheduling aspects (placement, deadline, load balancing)
- **Cache-aware Domains**: Automatic domain creation based on LLC topology
- **Two-tier Load Balancing**: High-frequency tuning (100ms) and low-frequency balancing (2s)
- **Greedy Work Stealing**: Configurable task stealing within and across NUMA boundaries
- **Deadline Tracking**: Per-task deadline management for fair scheduling
- **NUMA-aware Operations**: Optimized for multi-socket systems
- **Detailed Statistics**: Comprehensive scheduling metrics via scx_stats

## Goals

This scheduler ultimately aims to demonstrate how to build modular BPF schedulers to enable easy code reuse between scheduler codebases. The main way of achieving this is through the use of BPF arenas that make it possible to directly share memory between the userspace and kernel scheduler components. This in turn lets us offload most of the complexity of the scheduler to userspace. Userspace components can be more easily combined, as opposed to scheduler BPF methods that are often mutually exclusive.

## Use Cases

scx_wd40 is primarily intended for:
- Demonstrating BPF arena usage in production-like schedulers
- Testing modular scheduler design patterns
- Exploring advanced BPF features in scheduling contexts
- Research into hybrid kernel/userspace scheduler architectures
- Prototyping new scheduling algorithms with minimal kernel code

## Production Readiness

No. This scheduler heavily uses BPF arenas and as such routinely requires a bleeding-edge kernel toolchain to even run and verify. Additional limitations include:
- Requires latest kernel with full BPF arena support
- Experimental features may have stability issues
- Not extensively tested across diverse workloads
- Assumes equal processing power across all domains
- Limited production deployment experience

## Command Line Options

```
scx_wd40: A fork of the scx_rusty multi-domain scheduler

Usage: scx_wd40 [OPTIONS]

Options:
  -u, --slice-us-underutil <SLICE_US_UNDERUTIL>
          Scheduling slice duration for under-utilized hosts, in microseconds [default: 20000]
  -o, --slice-us-overutil <SLICE_US_OVERUTIL>
          Scheduling slice duration for over-utilized hosts, in microseconds [default: 1000]
  -i, --interval <INTERVAL>
          Load balance interval in seconds [default: 2.0]
  -I, --tune-interval <TUNE_INTERVAL>
          The tuner runs at a higher frequency than the load balancer to dynamically tune scheduling behavior. Tuning interval in seconds [default: 0.1]
  -l, --load-half-life <LOAD_HALF_LIFE>
          The half-life of task and domain load running averages in seconds [default: 1.0]
  -c, --cache-level <CACHE_LEVEL>
          Build domains according to how CPUs are grouped at this cache level [default: 3]
  -g, --greedy-threshold <GREEDY_THRESHOLD>
          When non-zero, enable greedy task stealing [default: 1]
      --greedy-threshold-x-numa <GREEDY_THRESHOLD_X_NUMA>
          When non-zero, enable greedy task stealing across NUMA nodes [default: 0]
  -h, --help
          Print help
```
