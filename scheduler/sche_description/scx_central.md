# scx_central

## Overview

A "central" scheduler where scheduling decisions are made from a single CPU. This scheduler illustrates how scheduling decisions can be dispatched from a single CPU, allowing other cores to run with infinite slices, without timer ticks, and without having to incur the overhead of making scheduling decisions.

## Description

scx_central implements a centralized scheduling architecture where one designated CPU (the "central" CPU) makes all scheduling decisions for the entire system. This design allows all other CPUs to run tasks with infinite time slices (SCX_SLICE_INF) without timer interrupts, significantly reducing scheduling overhead.

The scheduler operates by having the central CPU's dispatch logic handle task assignment for all CPUs in the system. Non-central CPUs simply execute tasks without making scheduling decisions, which can lead to better cache locality and reduced context switching overhead.

## Features

- **Single-CPU Decision Making**: All scheduling logic concentrated on one CPU
- **Infinite Time Slices**: Tasks run without timer-based preemption on non-central CPUs
- **Reduced Overhead**: Eliminates scheduling overhead on worker CPUs
- **FIFO Ordering**: Simple first-in-first-out task ordering
- **Configurable Central CPU**: Can designate which CPU acts as the scheduler
- **Timer-based Preemption**: Tasks are preempted every 20ms via timer callback

## Use Cases

This scheduler could theoretically be useful for any workload that benefits from minimizing scheduling overhead and timer ticks. An example of where this could be particularly useful is running VMs, where running with infinite slices and no timer ticks allows the VM to avoid unnecessary expensive vmexits.

## Production Readiness

Not yet. While tasks are run with an infinite slice (`SCX_SLICE_INF`), they're preempted every 20ms in a timer callback. The scheduler also puts the core scheduling logic inside of the central / scheduling CPU's `ops.dispatch()` path, and does not yet have any kind of priority mechanism.


## Command Line Options

```
/root/yunwei37/ai-os/scheduler/sche_bin/scx_central: invalid option -- '-'
A central FIFO sched_ext scheduler.

See the top-level comment in .bpf.c for more details.

Usage: scx_central [-s SLICE_US] [-c CPU]

  -s SLICE_US   Override slice duration
  -c CPU        Override the central CPU (default: 0)
  -v            Print libbpf debug messages
  -h            Display this help and exit
```
