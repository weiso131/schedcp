# Scheduler Components

This directory contains scheduler implementations and tools for the AI-OS project.

## SCX (Sched-Ext)

The `scx` subdirectory contains the sched-ext project, which provides extensible scheduling capabilities for Linux using BPF.

### Building

To build all SCX schedulers and components:

```bash
make build
```

To clean build artifacts:

```bash
make clean
```

### Available Schedulers

SCX includes several scheduler implementations:
- scx_simple: A simple scheduler example
- scx_central: Central scheduling algorithm
- scx_flatcg: Flattened cgroup scheduler
- scx_lavd: LAVD (Latency-Aware Virtual Deadline) scheduler
- scx_layered: Layered scheduler with priority tiers
- scx_nest: Nested scheduling implementation
- scx_pair: Pair-based scheduler
- scx_qmap: Queue-map based scheduler
- scx_rlfifo: Real-time FIFO scheduler
- scx_rusty: Rust-based scheduler implementation
- scx_userland: Userspace-driven scheduler

### Requirements

- Linux kernel with sched-ext support
- Rust toolchain
- Cargo
- Clang/LLVM for BPF compilation
- libbpf development headers

### Usage

After building, the schedulers can be found in `scx/scheds/rust/scx_*/target/release/`.

Example usage:
```bash
sudo ./scx/scheds/rust/scx_simple/target/release/scx_simple
```

For more information, see the [SCX project documentation](https://github.com/sched-ext/scx).