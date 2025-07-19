# scx_central


### Overview

A "central" scheduler where scheduling decisions are made from a single CPU.
This scheduler illustrates how scheduling decisions can be dispatched from a
single CPU, allowing other cores to run with infinite slices, without timer
ticks, and without having to incur the overhead of making scheduling decisions.

### Typical Use Case

This scheduler could theoretically be useful for any workload that benefits
from minimizing scheduling overhead and timer ticks. An example of where this
could be particularly useful is running VMs, where running with infinite slices
and no timer ticks allows the VM to avoid unnecessary expensive vmexits.

### Production Ready?

Not yet. While tasks are run with an infinite slice (`SCX_SLICE_INF`), they're
preempted every 20ms in a timer callback. The scheduler also puts the core
scheduling logic inside of the central / scheduling CPU's `ops.dispatch()` path,
and does not yet have any kind of priority mechanism.

--------------------------------------------------------------------------------


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


## Command Line Options

```
/root/yunwei37/ai-os/scheduler/sche_bin/scx_central: invalid option -- '-'
A central FIFO sched_ext scheduler.

See the top-level comment in .bpf.c for more details.

Usage: scx_central [-s SLICE_US] [-c CPU]

  -s SLICE_US   Override slice duration
  -c CPU        Override the central CPU (default: 0)
  -v            Print libbpf debug messages
  -h            Display this help and exit
```
