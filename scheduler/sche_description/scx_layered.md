# scx_layered

This is a single user-defined scheduler used within [sched_ext](https://github.com/sched-ext/scx/tree/main), which is a Linux kernel feature which enables implementing kernel thread schedulers in BPF and dynamically loading them. [Read more about sched_ext](https://github.com/sched-ext/scx/tree/main).

## Overview

A highly configurable multi-layer BPF / user space hybrid scheduler.

scx_layered allows the user to classify tasks into multiple layers, and apply
different scheduling policies to those layers. For example, a layer could be
created of all tasks that are part of the `user.slice` cgroup slice, and a
policy could be specified that ensures that the layer is given at least 80% CPU
utilization for some subset of CPUs on the system.

## How To Install

Available as a [Rust crate](https://crates.io/crates/scx_layered): `cargo add scx_layered`

## Typical Use Case

scx_layered is designed to be highly customizable, and can be targeted for
specific applications. For example, if you had a high-priority service that
required priority access to all but 1 physical core to ensure acceptable p99
latencies, you could specify that the service would get priority access to all
but 1 core on the system. If that service ends up not utilizing all of those
cores, they could be used by other layers until they're needed.

## Production Ready?

Yes. If tuned correctly, scx_layered should be performant across various CPU
architectures and workloads.

That said, you may run into an issue with infeasible weights, where a task with
a very high weight may cause the scheduler to incorrectly leave cores idle
because it thinks they're necessary to accommodate the compute for a single
task. This can also happen in CFS, and should soon be addressed for
scx_layered.

## Tuning scx_layered
`scx_layered` is designed with specific use cases in mind and may not perform
as well as a general purpose scheduler for all workloads. It does have topology
awareness, which can be disabled with the `-t` flag. This may impact
performance on NUMA machines, as layers will be able to span NUMA nodes by
default. For configuring `scx_layered` to span across multiple NUMA nodes simply
setting all nodes in the `nodes` field of the config.

For controlling the performance level of different levels (i.e. CPU frequency)
the `perf` field can be set. This must be used in combination with the
`schedutil` frequency governor. The value should be from 0-1024 with 1024 being
maximum performance. Depending on the system hardware it will translate to
frequency, which can also trigger turbo boosting if the value is high enough
and turbo is enabled.

Layer affinities can be defined using the `nodes` or `llcs` layer configs. This
allows for restricting a layer to a NUMA node or LLC. Layers will by default
attempt to grow within the same NUMA node, however this may change to suppport
different layer growth strategies in the future. When tuning the `util_range`
for a layer there should be some consideration for how the layer should grow.
For example, if the `util_range` lower bound is too high, it may lead to the
layer shrinking excessively. This could be ideal for core compaction strategies
for a layer, but may poorly utilize hardware, especially in low system
utilization. The upper bound of the `util_range` controls how the layer grows,
if set too aggressively the layer could grow fast and prevent other layers from
utilizing CPUs. Lastly, the `slice_us` can be used to tune the timeslice
per layer. This is useful if a layer has more latency sensitive tasks, where
timeslices should be shorter. Conversely if a layer is largely CPU bound with
less concerns of latency it may be useful to increase the `slice_us` parameter.

`scx_layered` can provide performance wins, for certain workloads when
sufficient tuning on the layer config.

## Command Line Options

```
scx_layered: A highly configurable multi-layer sched_ext scheduler

scx_layered allows classifying tasks into multiple layers and applying
different scheduling policies to them. The configuration is specified in
json and composed of two parts - matches and policies.

Matches
=======

Whenever a task is forked or its attributes are changed, the task goes
through a series of matches to determine the layer it belongs to. A
match set is composed of OR groups of AND blocks. An example:

  "matches": [
    [
      {
        "CgroupPrefix": "system.slice/"
      }
    ],
    [
      {
        "CommPrefix": "fbagent"
      },
      {
        "NiceAbove": 0
      }
    ]
  ],

The outer array contains the OR groups and the inner AND blocks, so the
above matches:

- Tasks which are in the cgroup sub-hierarchy under "system.slice".

- Or tasks whose comm starts with "fbagent" and have a nice value > 0.

Currently, the following matches are supported:

- CgroupPrefix: Matches the prefix of the cgroup that the task belongs
  to. As this is a string match, whether the pattern has the trailing
  '/' makes a difference. For example, "TOP/CHILD/" only matches tasks
  which are under that particular cgroup while "TOP/CHILD" also matches
  tasks under "TOP/CHILD0/" or "TOP/CHILD1/".

- CommPrefix: Matches the task's comm prefix.

- PcommPrefix: Matches the task's thread group leader's comm prefix.

- NiceAbove: Matches if the task's nice value is greater than the
  pattern.

- NiceBelow: Matches if the task's nice value is smaller than the
  pattern.

- NiceEquals: Matches if the task's nice value is exactly equal to
  the pattern.

- UIDEquals: Matches if the task's effective user id matches the value

- GIDEquals: Matches if the task's effective group id matches the value.

- PIDEquals: Matches if the task's pid matches the value.

- PPIDEquals: Matches if the task's ppid matches the value.

- TGIDEquals: Matches if the task's tgid matches the value.

- NSPIDEquals: Matches if the task's namespace id and pid matches the values.

- NSEquals: Matches if the task's namespace id matches the values.

- IsGroupLeader: Bool. When true, matches if the task is group leader
  (i.e. PID == TGID), aka the thread from which other threads are made.
  When false, matches if the task is *not* the group leader (i.e. the rest).

- CmdJoin: Matches when the task uses pthread_setname_np to send a join/leave
command to the scheduler. See examples/cmdjoin.c for more details.

- UsedGpuTid: Bool. When true, matches if the tasks which have used
  gpus by tid.

- UsedGpuPid: Bool. When true, matches if the tasks which have used gpu
  by tgid/pid.

- [EXPERIMENTAL] AvgRuntime: (u64, u64). Match tasks whose average runtime
  is within the provided values [min, max).

While there are complexity limitations as the matches are performed in
BPF, it is straightforward to add more types of matches.

Templates
---------

Templates let us create a variable number of layers dynamically at initialization
time out of a cgroup name suffix/prefix. Sometimes we know there are multiple
applications running on a machine, each with their own cgroup but do not know the
exact names of the applications or cgroups, e.g., in cloud computing contexts where
workloads are placed on machines dynamically and run under cgroups whose name is
autogenerated. In that case, we cannot hardcode the cgroup match rules when writing
the configuration. We thus cannot easily prevent tasks from different cgroups from
falling into the same layer and affecting each other's performance.


Templates offer a solution to this problem by generating one layer for each such cgroup,
provided these cgroups share a suffix, and that the suffix is unique to them. Templates
have a cgroup suffix rule that we use to find the relevant cgroups in the system. For each
such cgroup, we copy the layer config and add a matching rule that matches just this cgroup.


Policies
========

The following is an example policy configuration for a layer.

  "kind": {
    "Confined": {
      "cpus_range": [1, 8],
      "util_range": [0.8, 0.9]
    }
  }

It's of "Confined" kind, which tries to concentrate the layer's tasks
into a limited number of CPUs. In the above case, the number of CPUs
assigned to the layer is scaled between 1 and 8 so that the per-cpu
utilization is kept between 80% and 90%. If the CPUs are loaded higher
than 90%, more CPUs are allocated to the layer. If the utilization drops
below 80%, the layer loses CPUs.

Currently, the following policy kinds are supported:

- Confined: Tasks are restricted to the allocated CPUs. The number of
  CPUs allocated is modulated to keep the per-CPU utilization in
  "util_range". The range can optionally be restricted with the
  "cpus_range" property.

- Grouped: Similar to Confined but tasks may spill outside if there are
  idle CPUs outside the allocated ones. The range can optionally be
  restricted with the "cpus_range" property.

- Open: Prefer the CPUs which are not occupied by Confined or Grouped
  layers. Tasks in this group will spill into occupied CPUs if there are
  no unoccupied idle CPUs.

All layers take the following options:

- min_exec_us: Minimum execution time in microseconds. Whenever a task
  is scheduled in, this is the minimum CPU time that it's charged no
  matter how short the actual execution time may be.

- yield_ignore: Yield ignore ratio. If 0.0, yield(2) forfeits a whole
  execution slice. 0.25 yields three quarters of an execution slice and
  so on. If 1.0, yield is completely ignored.

- slice_us: Scheduling slice duration in microseconds.

- fifo: Use FIFO queues within the layer instead of the default vtime.

- preempt: If true, tasks in the layer will preempt tasks which belong
  to other non-preempting layers when no idle CPUs are available.

- preempt_first: If true, tasks in the layer will try to preempt tasks
  in their previous CPUs before trying to find idle CPUs.

- exclusive: If true, tasks in the layer will occupy the whole core. The
  other logical CPUs sharing the same core will be kept idle. This isn't
  a hard guarantee, so don't depend on it for security purposes.

- allow_node_aligned: Put node aligned tasks on layer DSQs instead of lo
  fallback. This is a hack to support node-affine tasks without making
  the whole scheduler node aware and should only be used with open
  layers on non-saturated machines to avoid possible stalls.

- prev_over_idle_core: On SMT enabled systems, prefer using the same CPU
  when picking a CPU for tasks on this layer, even if that CPUs SMT
  sibling is processing a task.

- weight: Weight of the layer, which is a range from 1 to 10000 with a
  default of 100. Layer weights are used during contention to prevent
  starvation across layers. Weights are used in combination with
  utilization to determine the infeasible adjusted weight with higher
  weights having a larger adjustment in adjusted utilization.

- disallow_open_after_us: Duration to wait after machine reaches saturation
  before confining tasks in Open layers.

- cpus_range_frac: Array of 2 floats between 0 and 1.0. Lower and upper
  bound fractions of all CPUs to give to a layer. Mutually exclusive
  with cpus_range.

- disallow_preempt_after_us: Duration to wait after machine reaches saturation
  before confining tasks to preempt.

- xllc_mig_min_us: Skip cross-LLC migrations if they are likely to run on
  their existing LLC sooner than this.

- idle_smt: *** DEPRECATED ****

- growth_algo: When a layer is allocated new CPUs different algorithms can
  be used to determine which CPU should be allocated next. The default
  algorithm is a "sticky" algorithm that attempts to spread layers evenly
  across cores.

- perf: CPU performance target. 0 means no configuration. A value
  between 1 and 1024 indicates the performance level CPUs running tasks
  in this layer are configured to using scx_bpf_cpuperf_set().

- idle_resume_us: Sets the idle resume QoS value. CPU idle time governors are expected to
  regard the minimum of the global (effective) CPU latency limit and the effective resume
  latency constraint for the given CPU as the upper limit for the exit latency of the idle
  states. See the latest kernel docs for more details:
  https://www.kernel.org/doc/html/latest/admin-guide/pm/cpuidle.html

- nodes: If set the layer will use the set of NUMA nodes for scheduling
  decisions. If unset then all available NUMA nodes will be used. If the
  llcs value is set the cpuset of NUMA nodes will be or'ed with the LLC
  config.

- llcs: If set the layer will use the set of LLCs (last level caches)
  for scheduling decisions. If unset then all LLCs will be used. If
  the nodes value is set the cpuset of LLCs will be or'ed with the nodes
  config.


Similar to matches, adding new policies and extending existing ones
should be relatively straightforward.

Configuration example and running scx_layered
=============================================

An scx_layered config is composed of layer configs. A layer config is
composed of a name, a set of matches, and a policy block. Running the
following will write an example configuration into example.json.

  $ scx_layered -e example.json

Note that the last layer in the configuration must have an empty match set
as a catch-all for tasks which haven't been matched into previous layers.

The configuration can be specified in multiple json files and
command line arguments, which are concatenated in the specified
order. Each must contain valid layer configurations.

By default, an argument to scx_layered is interpreted as a JSON string. If
the argument is a pointer to a JSON file, it should be prefixed with file:
or f: as follows:

  $ scx_layered file:example.json
  ...
  $ scx_layered f:example.json

Monitoring Statistics
=====================

Run with `--stats INTERVAL` to enable stats monitoring. There is
also an scx_stat server listening on /var/run/scx/root/stat that can
be monitored by running `scx_layered --monitor INTERVAL` separately.

  ```bash
  $ scx_layered --monitor 1
  tot= 117909 local=86.20 open_idle= 0.21 affn_viol= 1.37 proc=6ms
  busy= 34.2 util= 1733.6 load=  21744.1 fallback_cpu=  1
    batch    : util/frac=   11.8/  0.7 load/frac=     29.7:  0.1 tasks=  2597
               tot=   3478 local=67.80 open_idle= 0.00 preempt= 0.00 affn_viol= 0.00
               cpus=  2 [  2,  2] 04000001 00000000
    immediate: util/frac= 1218.8/ 70.3 load/frac=  21399.9: 98.4 tasks=  1107
               tot=  68997 local=90.57 open_idle= 0.26 preempt= 9.36 affn_viol= 0.00
               cpus= 50 [ 50, 50] fbfffffe 000fffff
    normal   : util/frac=  502.9/ 29.0 load/frac=    314.5:  1.4 tasks=  3512
               tot=  45434 local=80.97 open_idle= 0.16 preempt= 0.00 affn_viol= 3.56
               cpus= 50 [ 50, 50] fbfffffe 000fffff
  ```

Global statistics: see [`SysStats`]

Per-layer statistics: see [`LayerStats`]

Usage: scx_layered [OPTIONS] [SPECS]...

Arguments:
  [SPECS]...
          Layer specification. See --help

Options:
  -s, --slice-us <SLICE_US>
          Scheduling slice duration in microseconds
          
          [default: 20000]

  -M, --max-exec-us <MAX_EXEC_US>
          Maximum consecutive execution time in microseconds. A task may be allowed to keep
          executing on a CPU for this long. Note that this is the upper limit and a task may have to
          moved off the CPU earlier. 0 indicates default - 20 * slice_us
          
          [default: 0]

  -i, --interval <INTERVAL>
          Scheduling interval in seconds
          
          [default: 0.1]

  -n, --no-load-frac-limit
          ***DEPRECATED*** Disable load-fraction based max layer CPU limit. recommended

      --exit-dump-len <EXIT_DUMP_LEN>
          Exit debug dump buffer length. 0 indicates default
          
          [default: 0]

  -v, --verbose...
          Enable verbose output, including libbpf details. Specify multiple times to increase
          verbosity

  -t, --disable-topology[=<DISABLE_TOPOLOGY>]
          Disable topology awareness. When enabled, the "nodes" and "llcs" settings on a layer are
          ignored. Defaults to false on topologies with multiple NUMA nodes or LLCs, and true
          otherwise
          
          [possible values: true, false]

      --xnuma-preemption
          Enable cross NUMA preemption

      --monitor-disable
          Disable monitor

  -e, --example <EXAMPLE>
          Write example layer specifications into the file and exit

      --layer-preempt-weight-disable <LAYER_PREEMPT_WEIGHT_DISABLE>
          ***DEPRECATED*** Disables preemption if the weighted load fraction of a layer
          (load_frac_adj) exceeds the threshold. The default is disabled (0.0)
          
          [default: 0.0]

      --layer-growth-weight-disable <LAYER_GROWTH_WEIGHT_DISABLE>
          ***DEPRECATED*** Disables layer growth if the weighted load fraction of a layer
          (load_frac_adj) exceeds the threshold. The default is disabled (0.0)
          
          [default: 0.0]

      --stats <STATS>
          Enable stats monitoring with the specified interval

      --monitor <MONITOR>
          Run in stats monitoring mode with the specified interval. Scheduler is not launched

      --run-example
          Run with example layer specifications (useful for e.g. CI pipelines)

      --local-llc-iteration
          ***DEPRECATED *** Enables iteration over local LLCs first for dispatch

      --lo-fb-wait-us <LO_FB_WAIT_US>
          Low priority fallback DSQs are used to execute tasks with custom CPU affinities. These
          DSQs are immediately executed iff a CPU is otherwise idle. However, after the specified
          wait, they are guranteed upto --lo-fb-share fraction of each CPU
          
          [default: 10000]

      --lo-fb-share <LO_FB_SHARE>
          The fraction of CPU time guaranteed to low priority fallback DSQs. See --lo-fb-wait-us
          
          [default: .05]

      --disable-antistall
          Disable antistall

      --enable-gpu-affinitize
          Enable numa topology based gpu task affinitization

      --gpu-affinitize-secs <GPU_AFFINITIZE_SECS>
          Interval at which to reaffinitize gpu tasks to numa nodes. Defaults to 900s
          
          [default: 900]

      --enable-match-debug
          Enable match debug This stores a mapping of task tid to layer id such that bpftool map
          dump can be used to debug layer matches

      --antistall-sec <ANTISTALL_SEC>
          Maximum task runnable_at delay (in seconds) before antistall turns on
          
          [default: 3]

      --enable-gpu-support
          Enable gpu support

      --gpu-kprobe-level <GPU_KPROBE_LEVEL>
          Gpu Kprobe Level The value set here determines how agressive the kprobes enabled on gpu
          driver functions are. Higher values are more aggressive, incurring more system overhead
          and more accurately identifying PIDs using GPUs in a more timely manner. Lower values
          incur less system overhead, at the cost of less accurately identifying GPU pids and taking
          longer to do so
          
          [default: 3]

      --netdev-irq-balance
          Enable netdev IRQ balancing. This is experimental and should be used with caution

      --disable-queued-wakeup
          Disable queued wakeup optimization

      --disable-percpu-kthread-preempt
          Per-cpu kthreads are preempting by default. Make it not so

      --percpu-kthread-preempt-all
          Only highpri (nice < 0) per-cpu kthreads are preempting by default. Make every per-cpu
          kthread preempting. Meaningful only if --disable-percpu-kthread-preempt is not set

  -V, --version
          Print scheduler version and exit

      --help-stats
          Show descriptions for statistics

      --layer-refresh-ms-avgruntime <LAYER_REFRESH_MS_AVGRUNTIME>
          Periodically force tasks in layers using the AvgRuntime match rule to reevaluate which
          layer they belong to. Default period of 2s. turns this off
          
          [default: 2000]

      --task-hint-map <TASK_HINT_MAP>
          Set the path for pinning the task hint map
          
          [default: ]

      --print-and-exit
          Print the config (after template expansion) and exit

  -h, --help
          Print help (see a summary with '-h')
```
