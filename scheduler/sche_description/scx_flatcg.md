# scx_flatcg


### Overview

A flattened cgroup hierarchy scheduler. This scheduler implements hierarchical
weight-based cgroup CPU control by flattening the cgroup hierarchy into a
single layer, by compounding the active weight share at each level. The effect
of this is a much more performant CPU controller, which does not need to
descend down cgroup trees in order to properly compute a cgroup's share.

### Typical Use Case

This scheduler could be useful for any typical workload requiring a CPU
controller, but which cannot tolerate the higher overheads of the fair CPU
controller.

### Production Ready?

Yes, though the scheduler (currently) does not adequately accommodate
thundering herds of cgroups. If, for example, many cgroups which are nested
behind a low-priority cgroup were to wake up around the same time, they may be
able to consume more CPU cycles than they are entitled to.

--------------------------------------------------------------------------------


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

## Command Line Options

```
/root/yunwei37/ai-os/scheduler/sche_bin/scx_flatcg: invalid option -- '-'
A flattened cgroup hierarchy sched_ext scheduler.

See the top-level comment in .bpf.c for more details.

Usage: scx_flatcg [-s SLICE_US] [-i INTERVAL] [-f] [-v]

  -s SLICE_US   Override slice duration
  -i INTERVAL   Report interval
  -f            Use FIFO scheduling instead of weighted vtime scheduling
  -v            Print libbpf debug messages
  -h            Display this help and exit
```
