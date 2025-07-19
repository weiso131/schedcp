# scx_nest


### Overview

A scheduler based on the following Inria-Paris paper: [OS Scheduling with Nest:
Keeping Tasks Close Together on Warm
Cores](https://hal.inria.fr/hal-03612592/file/paper.pdf). The core idea of the
scheduler is to make scheduling decisions which encourage work to run on cores
that are expected to have high frequency. This scheduler currently will only
perform well on single CCX / single-socket hosts.

### Typical Use Case

`scx_nest` is designed to optimize workloads that CPU utilization somewhat low,
and which can benefit from running on a subset of cores on the host so as to
keep the frequencies high on those cores. Some workloads may perform better by
spreading work across many cores to avoid thrashing the cache, etc. Determining
whether a workload is well-suited to `scx_nest` will likely require
experimentation.

### Production Ready?

This scheduler could be used in a production environment, assuming the hardware
constraints enumerated above.

--------------------------------------------------------------------------------


### Overview

A sibling scheduler which ensures that tasks will only ever be co-located on a
physical core if they're in the same cgroup. It illustrates how a scheduling
policy could be implemented to mitigate CPU bugs, such as L1TF, and also shows
how some useful kfuncs such as `scx_bpf_kick_cpu()` can be utilized.

### Typical Use Case

While this scheduler is only meant to be used to illustrate certain sched_ext
features, with a bit more work (e.g. by adding some form of priority handling
inside and across cgroups), it could have been used as a way to quickly
mitigate L1TF before core scheduling was implemented and rolled out.

### Production Ready?

No

--------------------------------------------------------------------------------



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
