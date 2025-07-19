# scx_pair


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


## Command Line Options

```
/root/yunwei37/ai-os/scheduler/sche_bin/scx_pair: invalid option -- '-'
A demo sched_ext core-scheduler which always makes every sibling CPU pair
execute from the same CPU cgroup.

See the top-level comment in .bpf.c for more details.

Usage: scx_pair [-S STRIDE]

  -S STRIDE     Override CPU pair stride (default: nr_cpus_ids / 2)
  -v            Print libbpf debug messages
  -h            Display this help and exit
```
