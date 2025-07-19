# scx_mitosis

## Overview

scx_mitosis is a dynamic affinity scheduler that implements a cell-based scheduling architecture. The scheduler dynamically assigns cgroups to "cells" which are then mapped to specific sets of CPUs. This approach allows for flexible CPU affinity management that can adapt to changing workload conditions.

The name "mitosis" reflects the scheduler's ability to dynamically split and merge cells based on system load and performance characteristics, similar to how biological cells undergo mitosis to divide.

## Description

scx_mitosis operates on a two-level scheduling architecture:

1. **Cell Management**: The scheduler divides the system into multiple cells (up to MAX_CELLS), where each cell is assigned a dynamic set of CPUs. Cells can be merged or split based on workload characteristics and system utilization.

2. **Cgroup Assignment**: Cgroups (and their associated tasks) are assigned to specific cells. Each cell maintains its own dispatch queue (DSQ) and performs virtual time (vtime) based scheduling for the cgroups within that cell.

The scheduler uses a hybrid BPF/userspace design where:
- The BPF component handles fast-path scheduling decisions, queue management, and vtime-based task selection
- The userspace component makes dynamic policy decisions about cell configuration, CPU assignment, and cgroup-to-cell mapping

Key scheduling decisions are made based on task locality preferences, with the scheduler tracking statistics for:
- Local queue decisions (task scheduled on preferred CPU)
- Default queue decisions
- High and low priority fallback queue decisions
- Affinity violations

## Features

- **Dynamic Cell Management**: Cells can be created, merged, or split based on system conditions
- **Adaptive CPU Assignment**: CPUs are dynamically reassigned between cells to balance load
- **Cgroup-aware Scheduling**: Full cgroup hierarchy support with per-cgroup scheduling contexts
- **Virtual Time Scheduling**: Fair scheduling within cells using vtime-based ordering
- **NUMA Awareness**: Can optimize cell-to-CPU mappings based on NUMA topology
- **Real-time Monitoring**: Detailed statistics on queue decisions and affinity violations
- **Configurable Intervals**: Separate intervals for reconfiguration, CPU rebalancing, and monitoring

## Production Readiness

No, scx_mitosis is an experimental scheduler designed to explore dynamic affinity management techniques. It serves as a research platform for cell-based scheduling architectures but is not recommended for production use.

## Command Line Options

```
scx_mitosis: A dynamic affinity scheduler

Cgroups are assigned to a dynamic number of Cells which are assigned to a dynamic set of CPUs. The
BPF part does simple vtime scheduling for each cell.

Userspace makes the dynamic decisions of which Cells should be merged or split and which CPUs they
should be assigned to.

Usage: scx_mitosis [OPTIONS]

Options:
  -v, --verbose...
          Enable verbose output, including libbpf details. Specify multiple times to increase
          verbosity

      --exit-dump-len <EXIT_DUMP_LEN>
          Exit debug dump buffer length. 0 indicates default
          
          [default: 0]

      --reconfiguration-interval-s <RECONFIGURATION_INTERVAL_S>
          Interval to consider reconfiguring the Cells (e.g. merge or split)
          
          [default: 10]

      --rebalance-cpus-interval-s <REBALANCE_CPUS_INTERVAL_S>
          Interval to consider rebalancing CPUs to Cells
          
          [default: 5]

      --monitor-interval-s <MONITOR_INTERVAL_S>
          Interval to report monitoring information
          
          [default: 1]

  -h, --help
          Print help (see a summary with '-h')
```
