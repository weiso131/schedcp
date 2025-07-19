# scx_chaos

## Overview

scx_chaos is a variant of the scx_p2dq scheduler that adds chaos testing capabilities. It is based on the P2DQ (Preemptive Distributed Queuing) scheduler implementation.

## Description

This scheduler is designed for testing and debugging purposes, introducing controlled chaos into scheduling decisions to help identify potential issues in the scheduling framework or workloads.

## Features

- Based on scx_p2dq implementation
- Includes chaos testing functionality
- Useful for stress testing and debugging

## Use Case

This scheduler is primarily intended for development and testing environments where you want to stress test scheduling behavior and identify potential race conditions or edge cases.

## Production Ready?

No, this scheduler is designed for testing purposes only and should not be used in production environments.
## Command Line Options

```
scx_chaos: A general purpose sched_ext scheduler designed to amplify race conditions

WARNING: This scheduler is a very early alpha, and hasn't been production tested yet. The CLI in
particular is likely very unstable and does not guarantee compatibility between versions.

scx_chaos is a general purpose scheduler designed to run apps with acceptable performance. It has a
series of features designed to add latency in paths in an application. All control is through the
CLI. Running without arguments will not attempt to introduce latency and can set a baseline for
performance impact. The other command line arguments allow for specifying latency inducing
behaviours which attempt to induce a crash.

Unlike most other schedulers, you can also run scx_chaos with a named target. For example: scx_chaos
-- ./app_that_might_crash --arg1 --arg2 In this mode the scheduler will automatically detach after
the application exits, unless run with `--repeat-failure` where it will restart the application on
failure.

Usage: scx_chaos [OPTIONS] [ARGS]...

Options:
      --repeat-failure
          Whether to continue on failure of the command under test

      --repeat-success
          Whether to continue on successful exit of the command under test

      --ppid-targeting <PPID_TARGETING>
          Whether to focus on the named task and its children instead of the entire system. Only
          takes effect if pid or args provided
          
          [default: true]
          [possible values: true, false]

  -v, --verbose...
          Enable verbose output, including libbpf details. Specify multiple times to increase
          verbosity

      --version
          Print version and exit

      --stats <STATS>
          Enable stats monitoring with the specified interval

      --monitor <MONITOR>
          Run in stats monitoring mode with the specified interval. Scheduler is not launched

  -h, --help
          Print help (see a summary with '-h')

Random Delays:
      --random-delay-frequency <RANDOM_DELAY_FREQUENCY>
          Chance of randomly delaying a process

      --random-delay-min-us <RANDOM_DELAY_MIN_US>
          Minimum time to add for random delay

      --random-delay-max-us <RANDOM_DELAY_MAX_US>
          Maximum time to add for random delay

Perf Degradation:
      --degradation-frequency <DEGRADATION_FREQUENCY>
          Chance of degradating a process

      --degradation-frac7 <DEGRADATION_FRAC7>
          Amount to degradate a process
          
          [default: 0]

CPU Frequency:
      --cpufreq-frequency <CPUFREQ_FREQUENCY>
          Chance of randomly delaying a process

      --cpufreq-min <CPUFREQ_MIN>
          Minimum CPU frequency for scaling

      --cpufreq-max <CPUFREQ_MAX>
          Maximum CPU frequency for scaling

Kprobe Random Delays:
      --kprobes-for-random-delays <KPROBES_FOR_RANDOM_DELAYS>...
          Introduce random delays in the scheduler whenever a provided kprobe is hit

      --kprobe-random-delay-frequency <KPROBE_RANDOM_DELAY_FREQUENCY>
          Chance of kprobe random delays. Must be between 0 and 1

General Scheduling:
  -k, --disable-kthreads-local
          Disables per-cpu kthreads directly dispatched into local dsqs

  -a, --autoslice
          Enables autoslice tuning

  -r, --interactive-ratio <INTERACTIVE_RATIO>
          Ratio of interactive tasks for autoslice tuning, percent value from 1-99
          
          [default: 10]

      --deadline
          Enables deadline scheduling

  -e, --eager-load-balance
          DEPRECATED

  -f, --freq-control
          Enables CPU frequency control

  -g, --greedy-idle-disable <GREEDY_IDLE_DISABLE>
          ***DEPRECATED*** Disables greedy idle CPU selection, may cause better load balancing on
          multi-LLC systems
          
          [default: true]
          [possible values: true, false]

  -y, --interactive-sticky
          Interactive tasks stay sticky to their CPU if no idle CPU is found

      --interactive-fifo
          Interactive tasks are FIFO scheduled

  -d, --dispatch-pick2-disable
          Disables pick2 load balancing on the dispatch path

      --dispatch-lb-busy <DISPATCH_LB_BUSY>
          Enables pick2 load balancing on the dispatch path when LLC utilization is under the
          specified utilization
          
          [default: 75]

      --dispatch-lb-interactive <DISPATCH_LB_INTERACTIVE>
          Enables pick2 load balancing on the dispatch path for interactive tasks
          
          [default: true]
          [possible values: true, false]

      --keep-running
          Enable tasks to run beyond their timeslice if the CPU is idle

      --interactive-dsq <INTERACTIVE_DSQ>
          Use a separate DSQ for interactive tasks
          
          [default: true]
          [possible values: true, false]

      --wakeup-lb-busy <WAKEUP_LB_BUSY>
          DEPRECATED
          
          [default: 0]

      --wakeup-llc-migrations
          Allow LLC migrations on the wakeup path

      --select-idle-in-enqueue
          Allow selecting idle in enqueue path

      --idle-resume-us <IDLE_RESUME_US>
          Set idle QoS resume latency based in microseconds

      --max-dsq-pick2 <MAX_DSQ_PICK2>
          Only pick2 load balance from the max DSQ
          
          [default: false]
          [possible values: true, false]

  -s, --min-slice-us <MIN_SLICE_US>
          Scheduling min slice duration in microseconds
          
          [default: 100]

      --lb-slack-factor <LB_SLACK_FACTOR>
          Slack factor for load balancing, load balancing is not performed if load is within slack
          factor percent
          
          [default: 5]

  -l, --min-llc-runs-pick2 <MIN_LLC_RUNS_PICK2>
          Number of runs on the LLC before a task becomes eligbile for pick2 migration on the wakeup
          path
          
          [default: 1]

  -t, --dsq-time-slices <DSQ_TIME_SLICES>
          Manual definition of slice intervals in microseconds for DSQs, must be equal to number of
          dumb_queues

  -x, --dsq-shift <DSQ_SHIFT>
          DSQ scaling shift, each queue min timeslice is shifted by the scaling shift
          
          [default: 4]

  -m, --min-nr-queued-pick2 <MIN_NR_QUEUED_PICK2>
          Minimum number of queued tasks to use pick2 balancing, 0 to always enabled
          
          [default: 0]

  -q, --dumb-queues <DUMB_QUEUES>
          Number of dumb DSQs
          
          [default: 3]

  -i, --init-dsq-index <INIT_DSQ_INDEX>
          Initial DSQ for tasks
          
          [default: 0]

Test Command:
  -p, --pid <PID>
          Stop the scheduler if specified process terminates

  [ARGS]...
          Program to run under the chaos scheduler
          
          Runs a program under test and tracks when it terminates, similar to most debuggers. Note
          that the scheduler still attaches for every process on the system.
```
