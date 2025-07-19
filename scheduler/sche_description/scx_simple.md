# scx_simple


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

### Typical Use Case

This scheduler outperforms the in-kernel fair class, `scx_simple`, and `scx_rusty`
on OLTP workloads run on systems with simple topology (i.e. non-NUMA, single
LLC).

### Production Ready?

`scx_prev` has not been tested in a production environment, but given its
similarity to `scx_simple`, it might be production ready for specific workloads
on hardware with simple topology.

--------------------------------------------------------------------------------


### Overview

A simple weighted vtime scheduler where all scheduling decisions take place in
user space. This is in contrast to Rusty, where load balancing lives in user
space, but scheduling decisions are still made in the kernel.

## Command Line Options

```
/root/yunwei37/ai-os/scheduler/sche_bin/scx_simple: invalid option -- '-'
A simple sched_ext scheduler.

See the top-level comment in .bpf.c for more details.

Usage: scx_simple [-f] [-v]

  -f            Use FIFO scheduling instead of weighted vtime scheduling
  -v            Print libbpf debug messages
  -h            Display this help and exit
```
