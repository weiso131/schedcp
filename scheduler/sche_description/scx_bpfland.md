# scx_bpfland

This is a single user-defined scheduler used within [sched_ext](https://github.com/sched-ext/scx/tree/main), which is a Linux kernel feature which enables implementing kernel thread schedulers in BPF and dynamically loading them. [Read more about sched_ext](https://github.com/sched-ext/scx/tree/main).

## Overview

scx_bpfland: a vruntime-based sched_ext scheduler that prioritizes interactive
workloads.

This scheduler is derived from scx_rustland, but it is fully implemented in BPF.
It has a minimal user-space Rust part to process command line options, collect
metrics and log out scheduling statistics. The BPF part makes all the
scheduling decisions.

Tasks are categorized as either interactive or regular based on their average
rate of voluntary context switches per second. Tasks that exceed a specific
voluntary context switch threshold are classified as interactive. Interactive
tasks are prioritized in a higher-priority queue, while regular tasks are
placed in a lower-priority queue. Within each queue, tasks are sorted based on
their weighted runtime: tasks that have higher weight (priority) or use the CPU
for less time (smaller runtime) are scheduled sooner, due to their a higher
position in the queue.

Moreover, each task gets a time slice budget. When a task is dispatched, it
receives a time slice equivalent to the remaining unused portion of its
previously allocated time slice (with a minimum threshold applied). This gives
latency-sensitive workloads more chances to exceed their time slice when needed
to perform short bursts of CPU activity without being interrupted (i.e.,
real-time audio encoding / decoding workloads).

## Typical Use Case

Interactive workloads, such as gaming, live streaming, multimedia, real-time
audio encoding/decoding, especially when these workloads are running alongside
CPU-intensive background tasks.

In this scenario scx_bpfland ensures that interactive workloads maintain a high
level of responsiveness.

## Production Ready?

The scheduler is based on scx_rustland, implementing nearly the same scheduling
algorithm with minor changes and optimizations to be fully implemented in BPF.

Given that the scx_rustland scheduling algorithm has been extensively tested,
this scheduler can be considered ready for production use.

## Command Line Options

```
scx_bpfland: a vruntime-based sched_ext scheduler that prioritizes interactive workloads.

This scheduler is derived from scx_rustland, but it is fully implemented in BPF. It has a minimal
user-space part written in Rust to process command line options, collect metrics and log out
scheduling statistics.

The BPF part makes all the scheduling decisions (see src/bpf/main.bpf.c).

Usage: scx_bpfland [OPTIONS]

Options:
      --exit-dump-len <EXIT_DUMP_LEN>
          Exit debug dump buffer length. 0 indicates default
          
          [default: 0]

  -s, --slice-us <SLICE_US>
          Maximum scheduling slice duration in microseconds
          
          [default: 20000]

  -S, --slice-us-min <SLICE_US_MIN>
          Minimum scheduling slice duration in microseconds
          
          [default: 1000]

  -l, --slice-us-lag <SLICE_US_LAG>
          Maximum time slice lag in microseconds.
          
          A positive value can help to enhance the responsiveness of interactive tasks, but it can
          also make performance more "spikey".
          
          A negative value can make performance more consistent, but it can also reduce the
          responsiveness of interactive tasks (by smoothing the effect of the vruntime scheduling
          and making the task ordering closer to a FIFO).
          
          [default: 20000]

  -t, --throttle-us <THROTTLE_US>
          Throttle the running CPUs by periodically injecting idle cycles.
          
          This option can help extend battery life on portable devices, reduce heating, fan noise
          and overall energy consumption (0 = disable).
          
          [default: 0]

  -I, --idle-resume-us <IDLE_RESUME_US>
          Set CPU idle QoS resume latency in microseconds (-1 = disabled).
          
          Setting a lower latency value makes CPUs less likely to enter deeper idle states,
          enhancing performance at the cost of higher power consumption. Alternatively, increasing
          the latency value may reduce performance, but also improve power efficiency.
          
          [default: -1]

  -n, --no-preempt
          Disable preemption.
          
          Never allow tasks to be directly dispatched. This can help to increase fairness over
          responsiveness.

  -p, --local-pcpu
          Enable per-CPU tasks prioritization.
          
          This allows to prioritize per-CPU tasks that usually tend to be de-prioritized (since they
          can't be migrated when their only usable CPU is busy). Enabling this option can introduce
          unfairness and potentially trigger stalls, but it can improve performance of server-type
          workloads (such as large parallel builds).

  -k, --local-kthreads
          Enable kthreads prioritization (EXPERIMENTAL).
          
          Enabling this can improve system performance, but it may also introduce noticeable
          interactivity issues or unfairness in scenarios with high kthread activity, such as heavy
          I/O or network traffic.
          
          Use it only when conducting specific experiments or if you have a clear understanding of
          its implications.

  -w, --no-wake-sync
          Disable direct dispatch during synchronous wakeups.
          
          Enabling this option can lead to a more uniform load distribution across available cores,
          potentially improving performance in certain scenarios. However, it may come at the cost
          of reduced efficiency for pipe-intensive workloads that benefit from tighter
          producer-consumer coupling.

  -m, --primary-domain <PRIMARY_DOMAIN>
          Specifies the initial set of CPUs, represented as a bitmask in hex (e.g., 0xff), that the
          scheduler will use to dispatch tasks, until the system becomes saturated, at which point
          tasks may overflow to other available CPUs.
          
          Special values: - "auto" = automatically detect the CPUs based on the active power profile
          - "performance" = automatically detect and prioritize the fastest CPUs - "powersave" =
          automatically detect and prioritize the slowest CPUs - "all" = all CPUs assigned to the
          primary domain - "none" = no prioritization, tasks are dispatched on the first CPU
          available
          
          [default: auto]

      --disable-l2
          Disable L2 cache awareness

      --disable-l3
          Disable L3 cache awareness

      --disable-smt
          Disable SMT awareness

      --disable-numa
          Disable NUMA rebalancing

  -f, --cpufreq
          Enable CPU frequency control (only with schedutil governor).
          
          With this option enabled the CPU frequency will be automatically scaled based on the load.

      --stats <STATS>
          Enable stats monitoring with the specified interval

      --monitor <MONITOR>
          Run in stats monitoring mode with the specified interval. Scheduler is not launched

  -d, --debug
          Enable BPF debugging via /sys/kernel/tracing/trace_pipe

  -v, --verbose
          Enable verbose output, including libbpf details

  -V, --version
          Print scheduler version and exit

      --help-stats
          Show descriptions for statistics

  -h, --help
          Print help (see a summary with '-h')
```
