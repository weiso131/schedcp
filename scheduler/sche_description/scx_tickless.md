# scx_tickless

This is a single user-defined scheduler used within [sched_ext](https://github.com/sched-ext/scx/tree/main), which is a Linux kernel feature which enables implementing kernel thread schedulers in BPF and dynamically loading them. [Read more about sched_ext](https://github.com/sched-ext/scx/tree/main).

## Overview

scx_tickless is a server-oriented scheduler designed for cloud computing,
virtualization, and high-performance computing workloads.

The scheduler works by routing all scheduling events through a pool of
primary CPUs assigned to handle these events. This allows disabling the
scheduler's tick on other CPUs, reducing OS noise.

By default, only CPU 0 is included in the pool of primary CPUs. However,
the pool size can be adjusted using the `--primary-domain CPUMASK` option.
On systems with a large number of CPUs, allocating multiple CPUs to the
primary pool may be beneficial.

Tasks are placed into a global queue and the primary CPUs are responsible
for distributing them to the other "tickless" CPUs. Preemption events are
sent from the primary CPUs via IPC only when a "tickless" CPU is being
contended by multiple tasks.

The primary CPUs also perform the check for a contended CPU and the
frequency of this check can be adjusted with the `--frequency FREQ` option.
This effectively determines the tick frequency on the "tickless" CPUs when
multiple tasks are competing for them.

NOTE: in order to effectively disable ticks on the "tickless" CPUs the
kernel must be booted with `nohz_full`. Keep in mind that `nohz_full`
introduces syscall overhead, so this may regress latency-sensitive
workloads.

## Typical Use Case

Typical use cases include cloud computing, virtualization and high
performance computing workloads. This scheduler is not designed for
latency-sensitive workloads.

## Production Ready?

This scheduler is still experimental.

## Command Line Options

```
Usage: scx_tickless [OPTIONS]

Options:
      --exit-dump-len <EXIT_DUMP_LEN>
          Exit debug dump buffer length. 0 indicates default
          
          [default: 0]

  -m, --primary-domain <PRIMARY_DOMAIN>
          Define the set of CPUs, represented as a bitmask in hex (e.g., 0xff), dedicated to process
          scheduling events
          
          [default: 0x1]

  -s, --slice-us <SLICE_US>
          Maximum scheduling slice duration in microseconds (applied only when multiple tasks are
          contending the same CPU)
          
          [default: 20000]

  -f, --frequency <FREQUENCY>
          Frequency of the tick triggered on the scheduling CPUs to check for task time slice
          expiration (0 == CONFIG_HZ).
          
          A higher frequency can increase the overall system responsiveness but it can also
          introduce more scheduling overhead and load on the primary CPUs.
          
          [default: 0]

  -p, --prefer-same-cpu
          Try to keep tasks running on the same CPU
          
          This can help to improve cache locality at the cost of introducing some extra overhead in
          the scheduler (and increase the load on the primary CPUs).

  -n, --nosmt
          Disable SMT topology awareness

      --stats <STATS>
          Enable stats monitoring with the specified interval

      --monitor <MONITOR>
          Run in stats monitoring mode with the specified interval. Scheduler is not launched

  -v, --verbose
          Enable verbose output, including libbpf details

  -V, --version
          Print scheduler version and exit

      --help-stats
          Show descriptions for statistics

  -h, --help
          Print help (see a summary with '-h')
```
