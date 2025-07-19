# scx_nest

## Overview

A scheduler based on the following Inria-Paris paper: [OS Scheduling with Nest: Keeping Tasks Close Together on Warm Cores](https://hal.inria.fr/hal-03612592/file/paper.pdf). The core idea of the scheduler is to make scheduling decisions which encourage work to run on cores that are expected to have high frequency. This scheduler currently will only perform well on single CCX / single-socket hosts.

## Description

scx_nest implements a core-packing scheduler that aims to maximize CPU frequency by concentrating work on a subset of "warm" cores. The scheduler divides CPU cores into two nests:

1. **Primary Nest**: A compact set of cores where most work is scheduled. These cores maintain high frequencies due to consistent utilization.
2. **Reserve Nest**: Additional cores that are used when the primary nest becomes saturated.

The scheduler dynamically adjusts the size of the primary nest based on workload demands. When cores in the primary nest are fully utilized, work spills over to the reserve nest. Conversely, when load decreases, cores are removed from the primary nest after a configurable idle delay.

This approach is based on the observation that modern CPUs can achieve higher boost frequencies when only a subset of cores are active, making it beneficial for certain workloads to pack tasks onto fewer cores rather than spreading them across all available cores.

## Features

- **Dynamic Nest Sizing**: Automatically adjusts the number of active cores based on load
- **Warm Core Preference**: Prioritizes scheduling on cores that are already active
- **Configurable Idle Delay**: Controls how quickly idle cores are removed from the primary nest
- **Reserve Core Management**: Maintains a configurable reserve of additional cores
- **Hyperthreading Awareness**: Can optionally prefer fully idle cores over sibling threads
- **Placement Failure Handling**: Aggressively expands the primary nest after repeated placement failures

## Use Cases

`scx_nest` is designed to optimize workloads that have somewhat low CPU utilization and which can benefit from running on a subset of cores on the host so as to keep the frequencies high on those cores. Some workloads may perform better by spreading work across many cores to avoid thrashing the cache, etc. Determining whether a workload is well-suited to `scx_nest` will likely require experimentation.

Ideal for:
- Latency-sensitive applications with moderate CPU usage
- Workloads that benefit from high single-core performance
- Systems where power efficiency is important
- Applications with bursty CPU demands

## Production Readiness

This scheduler could be used in a production environment, assuming the hardware constraints enumerated above (single CCX / single-socket hosts).



## Command Line Options

```
/root/yunwei37/ai-os/scheduler/sche_bin/scx_nest: invalid option -- '-'
A Nest sched_ext scheduler.

See the top-level comment in .bpf.c for more details.

Usage: scx_nest [-p] [-d DELAY] [-m <max>] [-i ITERS]

  -d DELAY_US   Delay (us), before removing an idle core from the primary nest (default 2000us / 2ms)
  -m R_MAX      Maximum number of cores in the reserve nest (default 5)
  -i ITERS      Number of successive placement failures tolerated before trying to aggressively expand primary nest (default 2), or 0 to disable
  -s SLICE_US   Override slice duration in us (default 20000us / 20ms)
  -I            First try to find a fully idle core, and then any idle core, when searching nests. Default behavior is to ignore hypertwins and check for any idle core.
  -v            Print libbpf debug messages
  -h            Display this help and exit
```
