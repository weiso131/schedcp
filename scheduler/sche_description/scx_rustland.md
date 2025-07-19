# scx_rustland

This is a single user-defined scheduler used within [sched_ext](https://github.com/sched-ext/scx/tree/main), which is a Linux kernel feature which enables implementing kernel thread schedulers in BPF and dynamically loading them. [Read more about sched_ext](https://github.com/sched-ext/scx/tree/main).

## Overview

scx_rustland is based on scx_rustland_core, a BPF component that abstracts
the low-level sched_ext functionalities. The actual scheduling policy is
entirely implemented in user space and it is written in Rust.

## How To Install

Available as a [Rust crate](https://crates.io/crates/scx_rustland): `cargo add scx_rustland`

## Typical Use Case

scx_rustland is designed to prioritize interactive workloads over background
CPU-intensive workloads. For this reason the typical use case of this scheduler
involves low-latency interactive applications, such as gaming, video
conferencing and live streaming.

scx_rustland is also designed to be an "easy to read" template that can be used
by any developer to quickly experiment more complex scheduling policies fully
implemented in Rust.

## Production Ready?

For performance-critical production scenarios, other schedulers are likely
to exhibit better performance, as offloading all scheduling decisions to
user-space comes with a certain cost (even if it's minimal).

However, a scheduler entirely implemented in user-space holds the potential for
seamless integration with sophisticated libraries, tracing tools, external
services (e.g., AI), etc.

Hence, there might be situations where the benefits outweigh the overhead,
justifying the use of this scheduler in a production environment.

## Demo

[scx_rustland-terraria](https://github.com/sched-ext/scx/assets/1051723/42ec3bf2-9f1f-4403-80ab-bf5d66b7c2d5)

The key takeaway of this demo is to demonstrate that , despite the overhead of
running a scheduler in user-space, we can still obtain interesting results and,
in this particular case, even outperform the default Linux scheduler (EEVDF) in
terms of application responsiveness (fps), while a CPU intensive workload
(parallel kernel build) is running in the background.

## Command Line Options

```
scx_rustland: user-space scheduler written in Rust

scx_rustland is designed to prioritize interactive workloads over background CPU-intensive
workloads. For this reason the typical use case of this scheduler involves low-latency interactive
applications, such as gaming, video conferencing and live streaming.

scx_rustland is also designed to be an "easy to read" template that can be used by any developer to
quickly experiment more complex scheduling policies fully implemented in Rust.

The scheduler is based on scx_rustland_core, which implements the low level sched-ext
functionalities.

The scheduling policy implemented in user-space is a based on a deadline, evaluated as following:

deadline = vruntime + exec_runtime

Where, vruntime reflects the task's total runtime scaled by weight (ensuring fairness), while
exec_runtime accounts the CPU time used since the last sleep (capturing responsiveness). Tasks are
then dispatched from the lowest to the highest deadline.

This approach favors latency-sensitive tasks: those that frequently sleep will accumulate less
exec_runtime, resulting in earlier deadlines. In contrast, CPU-intensive tasks that don’t sleep
accumulate a larger exec_runtime and thus get scheduled later.

All the tasks are stored in a BTreeSet (TaskTree), using the deadline as the ordering key. Once the
order of execution is determined all tasks are sent back to the BPF counterpart (scx_rustland_core)
to be dispatched.

The BPF dispatcher is completely agnostic of the particular scheduling policy implemented in
user-space. For this reason developers that are willing to use this scheduler to experiment
scheduling policies should be able to simply modify the Rust component, without having to deal with
any internal kernel / BPF details.

=== Troubleshooting ===

- Reduce the time slice (option `-s`) if you experience lag or cracking audio.

Usage: scx_rustland [OPTIONS]

Options:
  -s, --slice-us <SLICE_US>
          Scheduling slice duration in microseconds
          
          [default: 20000]

  -S, --slice-us-min <SLICE_US_MIN>
          Scheduling minimum slice duration in microseconds
          
          [default: 1000]

  -l, --percpu-local
          If set, per-CPU tasks are dispatched directly to their only eligible CPU. This can help
          enforce affinity-based isolation for better performance

  -p, --partial
          If specified, only tasks which have their scheduling policy set to SCHED_EXT using
          sched_setscheduler(2) are switched. Otherwise, all tasks are switched

      --exit-dump-len <EXIT_DUMP_LEN>
          Exit debug dump buffer length. 0 indicates default
          
          [default: 0]

  -v, --verbose
          Enable verbose output, including libbpf details. Moreover, BPF scheduling events will be
          reported in tracefs (e.g., /sys/kernel/tracing/trace_pipe)

      --stats <STATS>
          Enable stats monitoring with the specified interval

      --monitor <MONITOR>
          Run in stats monitoring mode with the specified interval. Scheduler is not launched

      --help-stats
          Show descriptions for statistics

  -V, --version
          Print scheduler version and exit

  -h, --help
          Print help (see a summary with '-h')
```
