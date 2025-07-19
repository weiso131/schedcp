# scx_rusty

This is a single user-defined scheduler used within [sched_ext](https://github.com/sched-ext/scx/tree/main), which is a Linux kernel feature which enables implementing kernel thread schedulers in BPF and dynamically loading them. [Read more about sched_ext](https://github.com/sched-ext/scx/tree/main).

## Overview

A multi-domain, BPF / user space hybrid scheduler. The BPF portion of the
scheduler does a simple round robin in each domain, and the user space portion
(written in Rust) calculates the load factor of each domain, and informs BPF of
how tasks should be load balanced accordingly.

## How To Install

Available as a [Rust crate](https://crates.io/crates/scx_rusty): `cargo add scx_rusty`

## Typical Use Case

Rusty is designed to be flexible, accommodating different architectures and
workloads. Various load balancing thresholds (e.g. greediness, frequency, etc),
as well as how Rusty should partition the system into scheduling domains, can
be tuned to achieve the optimal configuration for any given system or workload.

## Production Ready?

Yes. If tuned correctly, rusty should be performant across various CPU
architectures and workloads. By default, rusty creates a separate scheduling
domain per-LLC, so its default configuration may be performant as well. Note
however that scx_rusty does not yet disambiguate between LLCs in different NUMA
nodes, so it may perform better on multi-CCX machines where all the LLCs share
the same socket, as opposed to multi-socket machines.

Note as well that you may run into an issue with infeasible weights, where a
task with a very high weight may cause the scheduler to incorrectly leave cores
idle because it thinks they're necessary to accommodate the compute for a
single task. This can also happen in CFS, and should soon be addressed for
scx_rusty.

## Command Line Options

```
scx_rusty: A multi-domain BPF / userspace hybrid scheduler

The BPF part does simple vtime or round robin scheduling in each domain while tracking average load
of each domain and duty cycle of each task.

The userspace part performs two roles. First, it makes higher frequency (100ms) tuning decisions. It
identifies CPUs which are not too heavily loaded and marks them so that they can pull tasks from
other overloaded domains on the fly.

Second, it drives lower frequency (2s) load balancing. It determines whether load balancing is
necessary by comparing domain load averages. If there are large enough load differences, it examines
upto 1024 recently active tasks on the domain to determine which should be migrated.

The overhead of userspace operations is low. Load balancing is not performed frequently, but
work-conservation is still maintained through tuning and greedy execution. Load balancing itself is
not that expensive either. It only accesses per-domain load metrics to determine the domains that
need load balancing, as well as limited number of per-task metrics for each pushing domain.

An earlier variant of this scheduler was used to balance across six domains, each representing a
chiplet in a six-chiplet AMD processor, and could match the performance of production setup using
CFS.

WARNING: scx_rusty currently assumes that all domains have equal processing power and at similar
distances from each other. This limitation will be removed in the future.

Usage: scx_rusty [OPTIONS]

Options:
  -u, --slice-us-underutil <SLICE_US_UNDERUTIL>
          Scheduling slice duration for under-utilized hosts, in microseconds
          
          [default: 20000]

  -o, --slice-us-overutil <SLICE_US_OVERUTIL>
          Scheduling slice duration for over-utilized hosts, in microseconds
          
          [default: 1000]

  -i, --interval <INTERVAL>
          Load balance interval in seconds
          
          [default: 2.0]

  -I, --tune-interval <TUNE_INTERVAL>
          The tuner runs at a higher frequency than the load balancer to dynamically tune scheduling
          behavior. Tuning interval in seconds
          
          [default: 0.1]

  -l, --load-half-life <LOAD_HALF_LIFE>
          The half-life of task and domain load running averages in seconds
          
          [default: 1.0]

  -c, --cache-level <CACHE_LEVEL>
          Build domains according to how CPUs are grouped at this cache level as determined by
          /sys/devices/system/cpu/cpuX/cache/indexI/id
          
          [default: 3]

  -C, --cpumasks <CPUMASKS>...
          Instead of using cache locality, set the cpumask for each domain manually. Provide
          multiple --cpumasks, one for each domain. E.g. --cpumasks 0xff_00ff --cpumasks 0xff00 will
          create two domains, with the corresponding CPUs belonging to each domain. Each CPU must
          belong to precisely one domain

  -g, --greedy-threshold <GREEDY_THRESHOLD>
          When non-zero, enable greedy task stealing. When a domain is idle, a cpu will attempt to
          steal tasks from another domain as follows:
          
          1. Try to consume a task from the current domain 2. Try to consume a task from another
          domain in the current NUMA node (or globally, if running on a single-socket system), if
          the domain has at least this specified number of tasks enqueued.
          
          See greedy_threshold_x_numa to enable task stealing across NUMA nodes. Tasks stolen in
          this manner are not permanently stolen from their domain.
          
          [default: 1]

      --greedy-threshold-x-numa <GREEDY_THRESHOLD_X_NUMA>
          When non-zero, enable greedy task stealing across NUMA nodes. The order of greedy task
          stealing follows greedy-threshold as described above, and greedy-threshold must be nonzero
          to enable task stealing across NUMA nodes
          
          [default: 0]

      --no-load-balance
          Disable load balancing. Unless disabled, userspace will periodically calculate the load
          factor of each domain and instruct BPF which processes to move

  -k, --kthreads-local
          Put per-cpu kthreads directly into local dsq's

  -b, --balanced-kworkers
          In recent kernels (>=v6.6), the kernel is responsible for balancing kworkers across L3
          cache domains. Exclude them from load-balancing to avoid conflicting operations. Greedy
          executions still apply

  -f, --fifo-sched
          Use FIFO scheduling instead of weighted vtime scheduling

  -D, --direct-greedy-under <DIRECT_GREEDY_UNDER>
          Idle CPUs with utilization lower than this will get remote tasks directly pushed onto
          them. 0 disables, 100 always enables
          
          [default: 90.0]

  -K, --kick-greedy-under <KICK_GREEDY_UNDER>
          Idle CPUs with utilization lower than this may get kicked to accelerate stealing when a
          task is queued on a saturated remote domain. 0 disables, 100 enables always
          
          [default: 100.0]

  -r, --direct-greedy-numa
          Whether tasks can be pushed directly to idle CPUs on NUMA nodes different than their
          domain's node. If direct-greedy-under is disabled, this option is a no-op. Otherwise, if
          this option is set to false (default), tasks will only be directly pushed to idle CPUs if
          they reside on the same NUMA node as the task's domain

  -p, --partial
          If specified, only tasks which have their scheduling policy set to SCHED_EXT using
          sched_setscheduler(2) are switched. Otherwise, all tasks are switched

      --mempolicy-affinity
          Enables soft NUMA affinity for tasks that use set_mempolicy. This may improve performance
          in some scenarios when using mempolicies

      --stats <STATS>
          Enable stats monitoring with the specified interval

      --monitor <MONITOR>
          Run in stats monitoring mode with the specified interval. The scheduler is not launched

      --exit-dump-len <EXIT_DUMP_LEN>
          Exit debug dump buffer length. 0 indicates default
          
          [default: 0]

  -v, --verbose...
          Enable verbose output, including libbpf details. Specify multiple times to increase
          verbosity

      --version
          Print version and exit

      --help-stats
          Show descriptions for statistics

      --perf <PERF>
          Tunable for prioritizing CPU performance by configuring the CPU frequency governor. Valid
          values are [0, 1024]. Higher values prioritize performance, lower values prioritize energy
          efficiency. When in doubt, use 0 or 1024
          
          [default: 0]

  -h, --help
          Print help (see a summary with '-h')
```
