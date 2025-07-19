# scx_sdt

## Overview

scx_sdt (Scheduler with Data Tracking) is a simple demonstration scheduler that showcases the use of BPF arenas for per-task data management. It implements basic FIFO scheduling while tracking detailed statistics about task lifecycle and scheduling decisions.

## Description

scx_sdt is designed to demonstrate advanced BPF features, particularly the BPF arena allocator system for managing per-task data. The scheduler implements a straightforward scheduling policy while maintaining comprehensive statistics for each task.

The scheduler operates using:
1. **BPF Arena Allocation**: Uses the new BPF arena feature to allocate per-task statistics structures
2. **Shared DSQ**: All tasks are placed in a single shared dispatch queue (DSQ) with FIFO ordering
3. **Statistics Tracking**: Maintains detailed per-task and global statistics about scheduling events
4. **Default CPU Selection**: Uses the kernel's default CPU selection logic with idle detection

Key implementation details:
- Allocates per-task statistics structures using `scx_task_alloc()` from the BPF arena
- Tracks five key metrics: enqueue events, task initialization, task exit, idle CPU selections, and busy CPU selections
- Aggregates per-task statistics to global counters on task exit
- Implements minimal scheduling logic to focus on demonstrating arena usage

## Features

- **BPF Arena Demonstration**: Shows how to use BPF arenas for dynamic memory allocation
- **Per-Task Data Management**: Allocates and manages per-task statistics structures
- **Statistics Collection**: Comprehensive tracking of scheduling events
- **Simple FIFO Scheduling**: Uses a single shared queue for all tasks
- **Default CPU Selection**: Leverages kernel's built-in CPU selection logic
- **Global Statistics Aggregation**: Combines per-task stats into global counters

## Use Cases

This scheduler is primarily educational and useful for:
- Learning how to use BPF arenas for dynamic memory allocation
- Understanding per-task data management in BPF schedulers
- Demonstrating statistics collection in sched_ext
- Testing BPF arena allocator functionality
- Serving as a template for more complex schedulers that need per-task state

## Production Readiness

This is primarily an example scheduler for educational purposes and is not suitable for production use. It lacks:
- Priority handling
- Load balancing
- NUMA awareness
- Cgroup support
- Any form of fairness or deadline guarantees

The scheduler's main value is in demonstrating BPF arena usage patterns rather than providing an effective scheduling policy.
## Command Line Options

```
/root/yunwei37/ai-os/scheduler/sche_bin/scx_sdt: invalid option -- '-'
A simple sched_ext scheduler.

See the top-level comment in .bpf.c for more details.

Usage: scx_sdt [-f] [-v]

  -v            Print libbpf debug messages
  -h            Display this help and exit
```
