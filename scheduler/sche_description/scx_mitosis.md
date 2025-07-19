# scx_mitosis

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
