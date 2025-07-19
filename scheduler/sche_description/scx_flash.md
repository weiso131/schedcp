# scx_flash

This is a single user-defined scheduler used within [sched_ext](https://github.com/sched-ext/scx/tree/main), which is a Linux kernel feature which enables implementing kernel thread schedulers in BPF and dynamically loading them. [Read more about sched_ext](https://github.com/sched-ext/scx/tree/main).

## Overview

A scheduler that focuses on ensuring fairness among tasks and performance
predictability.

It operates using an earliest deadline first (EDF) policy, where each task is
assigned a "latency" weight. This weight is dynamically adjusted based on how
often a task release the CPU before its full time slice is used. Tasks that
release the CPU early are given a higher latency weight, prioritizing them over
tasks that fully consume their time slice.

## Typical Use Case

The combination of dynamic latency weights and EDF scheduling ensures
responsive and consistent performance, even in overcommitted systems.

This makes the scheduler particularly well-suited for latency-sensitive
workloads, such as multimedia or real-time audio processing.

## Production Ready?

Yes.

## Command Line Options

```
scx_flash is scheduler that focuses on ensuring fairness and performance predictability.

It operates using an earliest deadline first (EDF) policy. The deadline of each task deadline is
defined as:

    deadline = vruntime + exec_vruntime

`vruntime` represents the task's accumulated runtime, inversely scaled by its weight, while
`exec_vruntime` accounts for the vruntime accumulated since the last sleep event.

Fairness is ensured through `vruntime`, whereas `exec_vruntime` helps prioritize latency-sensitive
tasks. Tasks that are frequently blocked waiting for an event (typically latency-sensitive)
accumulate a smaller `exec_vruntime` compared to tasks that continuously consume CPU without
interruption.

As a result, tasks with a smaller `exec_vruntime` will have a shorter deadline and will be
dispatched earlier, ensuring better responsiveness for latency-sensitive tasks.

Moreover, tasks can accumulate a maximum `vruntime` credit while they're sleeping, based on how
often they voluntarily release the CPU (`avg_nvcsw`). This allows prioritizing frequent sleepers
over less-frequent ones.

Usage: scx_flash [OPTIONS]

Options:
      --exit-dump-len <EXIT_DUMP_LEN>
          Exit debug dump buffer length. 0 indicates default
          
          [default: 0]

  -s, --slice-us <SLICE_US>
          Maximum scheduling slice duration in microseconds
          
          [default: 4096]

  -S, --slice-us-min <SLICE_US_MIN>
          Minimum scheduling slice duration in microseconds
          
          [default: 128]

  -l, --slice-us-lag <SLICE_US_LAG>
          Maximum runtime budget that a task can accumulate while sleeping (in microseconds).
          
          Increasing this value can help to enhance the responsiveness of interactive tasks, but it
          can also make performance more "spikey".
          
          [default: 4096]

  -r, --run-us-lag <RUN_US_LAG>
          Maximum runtime penalty that a task can accumulate while running (in microseconds).
          
          Increasing this value can help to enhance the responsiveness of interactive tasks, but it
          can also make performance more "spikey".
          
          [default: 32768]

  -c, --max-avg-nvcsw <MAX_AVG_NVCSW>
          Maximum rate of voluntary context switches.
          
          Increasing this value can help prioritize interactive tasks with a higher sleep frequency
          over interactive tasks with lower sleep frequency.
          
          Decreasing this value makes the scheduler more robust and fair.
          
          (0 = disable voluntary context switch prioritization).
          
          [default: 128]

  -C, --cpu-busy-thresh <CPU_BUSY_THRESH>
          Utilization percentage to consider a CPU as busy (-1 = auto).
          
          A value close to 0 forces tasks to migrate quickier, increasing work conservation and
          potentially system responsiveness.
          
          A value close to 100 makes tasks more sticky to their CPU, increasing cache-sensivite and
          server-type workloads.
          
          In auto mode (-1) the scheduler autoomatically tries to determine the optimal value in
          function of the current workload.
          
          [default: -1]

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
          
          [default: 32]

  -T, --tickless
          Enable tickless mode.
          
          This option enables tickless mode: tasks get an infinite time slice and they are preempted
          only in case of CPU contention. This can help reduce the OS noise and provide a better
          level of performance isolation.

  -R, --rr-sched
          Enable round-robin scheduling.
          
          Each task is given a fixed time slice (defined by --slice-us) and run in a cyclic, fair
          order.

  -p, --local-pcpu
          Enable per-CPU tasks prioritization.
          
          Enabling this option allows to prioritize per-CPU tasks that usually tend to be
          de-prioritized, since they can't be migrated when their only usable CPU is busy. This
          improves fairness, but it can also reduce the overall system throughput.
          
          This option is recommended for gaming or latency-sensitive workloads.

  -y, --sticky-cpu
          Enable CPU stickiness.
          
          Enabling this option can reduce the amount of task migrations, but it can also make
          performance less consistent on systems with hybrid cores.
          
          This option has no effect if the primary scheduling domain includes all the CPUs (e.g.,
          `--primary-domain all`).

  -n, --native-priority
          Native tasks priorities.
          
          By default, the scheduler normalizes task priorities to avoid large gaps that could lead
          to stalls or starvation. This option disables normalization and uses the default Linux
          priority range instead.

  -k, --local-kthreads
          Enable per-CPU kthread prioritization.
          
          Enabling this can improve system performance, but it may also introduce interactivity
          issues or unfairness in scenarios with high kthread activity, such as heavy I/O or network
          traffic.

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
          - "turbo" = automatically detect and prioritize the CPUs with the highest max frequency -
          "performance" = automatically detect and prioritize the fastest CPUs - "powersave" =
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
