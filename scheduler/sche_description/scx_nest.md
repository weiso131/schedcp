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


