# scx_flatcg

## Overview

A flattened cgroup hierarchy scheduler. This scheduler implements hierarchical weight-based cgroup CPU control by flattening the cgroup hierarchy into a single layer, by compounding the active weight share at each level. The effect of this is a much more performant CPU controller, which does not need to descend down cgroup trees in order to properly compute a cgroup's share.

## Description

scx_flatcg is designed to provide efficient cgroup-based CPU scheduling by eliminating the performance overhead associated with traversing deep cgroup hierarchies. Traditional cgroup schedulers must walk the cgroup tree to calculate effective weights and make scheduling decisions, which can become expensive with complex hierarchies.

This scheduler solves the problem by:
1. **Hierarchy Flattening**: Converting the multi-level cgroup tree into a single flat structure
2. **Weight Compounding**: Pre-calculating effective weights by multiplying weights along the hierarchy path
3. **Direct Scheduling**: Making scheduling decisions based on the pre-calculated flat weights

The scheduler supports both weighted virtual time (vtime) scheduling and FIFO scheduling modes, allowing users to choose between fairness and simplicity based on their workload requirements.

## Features

- **Flattened Hierarchy**: Eliminates tree traversal overhead during scheduling decisions
- **Weight-based Scheduling**: Respects cgroup CPU weights while maintaining performance
- **Dual Scheduling Modes**: Supports both weighted vtime and FIFO scheduling
- **Low Overhead**: Significantly reduced scheduling overhead compared to traditional hierarchical schedulers
- **Configurable Time Slice**: Adjustable scheduling quantum
- **Performance Monitoring**: Built-in interval-based reporting of scheduling statistics

## Use Cases

This scheduler could be useful for any typical workload requiring a CPU controller, but which cannot tolerate the higher overheads of the fair CPU controller. It's particularly beneficial for:
- Container orchestration systems with deep cgroup hierarchies
- Multi-tenant environments requiring CPU isolation
- Workloads sensitive to scheduling overhead

## Production Readiness

Yes, though the scheduler (currently) does not adequately accommodate thundering herds of cgroups. If, for example, many cgroups which are nested behind a low-priority cgroup were to wake up around the same time, they may be able to consume more CPU cycles than they are entitled to.

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
