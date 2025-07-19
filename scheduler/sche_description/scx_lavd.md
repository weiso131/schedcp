# scx_lavd

This is a single user-defined scheduler used within [sched_ext](https://github.com/sched-ext/scx/tree/main), which is a Linux kernel feature which enables implementing kernel thread schedulers in BPF and dynamically loading them. [Read more about sched_ext](https://github.com/sched-ext/scx/tree/main).

## Overview

scx_lavd is a BPF scheduler that implements an LAVD (Latency-criticality Aware
Virtual Deadline) scheduling algorithm. While LAVD is new and still evolving,
its core ideas are 1) measuring how much a task is latency critical and 2)
leveraging the task's latency-criticality information in making various
scheduling decisions (e.g., task's deadline, time slice, etc.). As the name
implies, LAVD is based on the foundation of deadline scheduling. This scheduler
consists of the BPF part and the rust part. The BPF part makes all the
scheduling decisions; the rust part provides high-level information (e.g., CPU
topology) to the BPF code, loads the BPF code and conducts other chores (e.g.,
printing sampled scheduling decisions).

## Typical Use Case

scx_lavd is initially motivated by gaming workloads. It aims to improve
interactivity and reduce stuttering while playing games on Linux. Hence, this
scheduler's typical use case involves highly interactive applications, such as
gaming, which requires high throughput and low tail latencies. 

## Production Ready?

Yes, scx_lavd should be performant across various CPU architectures. It creates
a separate scheduling domain per-LLC, per-core type (e.g., P or E core on
Intel, big or LITTLE on ARM), and per-NUMA domain, so the default balanced
profile or autopilot mode should be performant. It mainly targets single CCX
/ single-socket systems.


## Command Line Options

```
scx_lavd: Latency-criticality Aware Virtual Deadline (LAVD) scheduler

The rust part is minimal. It processes command line options and logs out scheduling statistics. The
BPF part makes all the scheduling decisions. See the more detailed overview of the LAVD design at
main.bpf.c.

Usage: scx_lavd [OPTIONS]

Options:
      --autopilot
          Automatically decide the scheduler's power mode (performance vs. powersave vs. balanced),
          CPU preference order, etc, based on system load. The options affecting the power mode and
          the use of core compaction (--autopower, --performance, --powersave, --balanced,
          --no-core-compaction) cannot be used with this option. When no option is specified, this
          is a default mode

      --autopower
          Automatically decide the scheduler's power mode (performance vs. powersave vs. balanced)
          based on the system's active power profile. The scheduler's power mode decides the CPU
          preference order and the use of core compaction, so the options affecting these
          (--autopilot, --performance, --powersave, --balanced, --no-core-compaction) cannot be used
          with this option

      --performance
          Run the scheduler in performance mode to get maximum performance. This option cannot be
          used with other conflicting options (--autopilot, --autopower, --balanced, --powersave,
          --no-core-compaction) affecting the use of core compaction

      --powersave
          Run the scheduler in powersave mode to minimize powr consumption. This option cannot be
          used with other conflicting options (--autopilot, --autopower, --performance, --balanced,
          --no-core-compaction) affecting the use of core compaction

      --balanced
          Run the scheduler in balanced mode aiming for sweetspot between power and performance.
          This option cannot be used with other conflicting options (--autopilot, --autopower,
          --performance, --powersave, --no-core-compaction) affecting the use of core compaction

      --slice-max-us <SLICE_MAX_US>
          Maximum scheduling slice duration in microseconds
          
          [default: 5000]

      --slice-min-us <SLICE_MIN_US>
          Minimum scheduling slice duration in microseconds
          
          [default: 500]

      --preempt-shift <PREEMPT_SHIFT>
          Limit the ratio of preemption to the roughly top P% of latency-critical tasks. When N is
          given as an argument, P is 0.5^N * 100. The default value is 6, which limits the
          preemption for the top 1.56% of latency-critical tasks
          
          [default: 6]

      --cpu-pref-order <CPU_PREF_ORDER>
          List of CPUs in preferred order (e.g., "0-3,7,6,5,4"). The scheduler uses the CPU
          preference mode only when the core compaction is enabled (i.e., balanced or powersave mode
          is specified as an option or chosen in the autopilot or autopower mode). When
          "--cpu-pref-order" is given, it implies "--no-use-em"
          
          [default: ]

      --no-use-em
          Do not use the energy model in making CPU preference order decisions

      --no-futex-boost
          Do not boost futex holders

      --no-preemption
          Disable preemption

      --no-wake-sync
          Disable an optimization for synchronous wake-up

      --no-core-compaction
          Disable core compaction so the scheduler uses all the online CPUs. The core compaction
          attempts to minimize the number of actively used CPUs for unaffinitized tasks, respecting
          the CPU preference order. Normally, the core compaction is enabled by the power mode
          (i.e., balanced or powersave mode is specified as an option or chosen in the autopilot or
          autopower mode). This option cannot be used with the other options that control the core
          compaction (--autopilot, --autopower, --performance, --balanced, --powersave)

      --no-freq-scaling
          Disable controlling the CPU frequency

      --stats <STATS>
          Enable stats monitoring with the specified interval

      --monitor <MONITOR>
          Run in stats monitoring mode with the specified interval. Scheduler is not launched

      --monitor-sched-samples <MONITOR_SCHED_SAMPLES>
          Run in monitoring mode. Show the specified number of scheduling samples every second

  -v, --verbose...
          Enable verbose output, including libbpf details. Specify multiple times to increase
          verbosity

  -V, --version
          Print scheduler version and exit

      --help-stats
          Show descriptions for statistics

  -h, --help
          Print help (see a summary with '-h')
```
