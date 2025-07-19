# scx_p2dq

## Overview
A simple pick 2 load balancing scheduler with (dumb) multi-layer queueing.

The p2dq scheduler is a simple load balancing scheduler that uses a pick two
algorithm for load balancing. A fixed number of DSQs are created per LLC with
incremental slice intervals. If a task is able to consume the majority of the
assigned slice is it dispatched to a DSQ with a longer slice. Tasks that do not
consume more than half the slice are moved to shorter slice DSQs. The DSQs with
the shortest slice lengths are then determined to be "interactive". All DSQs on
the same LLC share the same vtime and there is special handling for
(non)interactive tasks for load balancing purposes.

The scheduler handles all scheduling decisions in BPF and the userspace
component is only for metric reporting.

## Use Cases
p2dq can perform well in a variety of workloads including interactive workloads
such as gaming, batch processing and server applications. Tuning of of p2dq for
each use case is required.

### Configuration
The main idea behind p2dq is being able to classify which tasks are interactive
and using a separate dispatch queue (DSQ) for them. Non interactive tasks
can have special properties such as being able to be load balanced across
LLCs/NUMA nodes. The `--autoslice` option will attempt to scale DSQ time slices
based on the `--interactive-ratio`. DSQ time slices can also be set manually
if the duration/distribution of tasks that are considered to be interactive is
known in advance. `scxtop` can be used to get an understanding of time slice
utilization so that DSQs can be properly configured. For desktop systems keeping
the interactive ratio small (ex: <5) and using a small number of queues (2) will
give a general performance with autoslice enabled.

## Command Line Options

```
scx_p2dq: A pick 2 dumb queuing load balancing scheduler.

The BPF part does simple vtime or round robin scheduling in each domain while tracking average load
of each domain and duty cycle of each task.

Usage: scx_p2dq [OPTIONS]

Options:
  -v, --verbose...
          Enable verbose output, including libbpf details. Specify multiple times to increase
          verbosity

      --stats <STATS>
          Enable stats monitoring with the specified interval

      --monitor <MONITOR>
          Run in stats monitoring mode with the specified interval. Scheduler is not launched

      --version
          Print version and exit

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

  -h, --help
          Print help (see a summary with '-h')
```
