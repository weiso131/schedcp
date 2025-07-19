# scx_qmap


### Overview

Another simple, yet slightly more complex scheduler that provides an example of
a basic weighted FIFO queuing policy. It also provides examples of some common
useful BPF features, such as sleepable per-task storage allocation in the
`ops.prep_enable()` callback, and using the `BPF_MAP_TYPE_QUEUE` map type to
enqueue tasks. It also illustrates how core-sched support could be implemented.

### Typical Use Case

Purely used to illustrate sched_ext features.

### Production Ready?

No

--------------------------------------------------------------------------------



### Overview

A simple scheduler that provides an example of a minimal sched_ext
scheduler. `scx_simple` can be run in either global weighted vtime mode, or
FIFO mode.

### Typical Use Case

Though very simple, this scheduler should perform reasonably well on
single-socket CPUs with a uniform L3 cache topology. Note that while running in
global FIFO mode may work well for some workloads, saturating threads can
easily starve inactive ones.

### Production Ready?

This scheduler could be used in a production environment, assuming the hardware
constraints enumerated above, and assuming the workload tolerates the simplicity
of the scheduling policy.

--------------------------------------------------------------------------------


### Overview

A variation on `scx_simple` with CPU selection that prioritizes an idle previous
CPU over finding a fully idle core (as is done in `scx_simple` and `scx_rusty`).


## Command Line Options

```
/root/yunwei37/ai-os/scheduler/sche_bin/scx_qmap: invalid option -- '-'
A simple five-level FIFO queue sched_ext scheduler.

See the top-level comment in .bpf.c for more details.

Usage: scx_qmap [-s SLICE_US] [-e COUNT] [-t COUNT] [-T COUNT] [-l COUNT] [-b COUNT]
       [-P] [-d PID] [-D LEN] [-p] [-v]

  -s SLICE_US   Override slice duration
  -e COUNT      Trigger scx_bpf_error() after COUNT enqueues
  -t COUNT      Stall every COUNT'th user thread
  -T COUNT      Stall every COUNT'th kernel thread
  -l COUNT      Trigger dispatch infinite looping after COUNT dispatches
  -b COUNT      Dispatch upto COUNT tasks together
  -P            Print out DSQ content to trace_pipe every second, use with -b
  -H            Boost nice -20 tasks in SHARED_DSQ, use with -b
  -d PID        Disallow a process from switching into SCHED_EXT (-1 for self)
  -D LEN        Set scx_exit_info.dump buffer length
  -S            Suppress qmap-specific debug dump
  -p            Switch only tasks on SCHED_EXT policy instead of all
  -v            Print libbpf debug messages
  -h            Display this help and exit
```
