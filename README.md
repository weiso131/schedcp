# KernCP

KernCP is an AI-driven control-plane for the Linux kernel: a lightweight management daemon plus an eBPF-/MCP-based policy server that watches live workload signals (perf, DAMON, energy, sched_ext events) and learns how to tweak schedulers, sysctls and other kernel knobs in real timeso your laptop stays quiet, your CI box stays fast, and your cluster stays self-healing, all without hand-tuning conf files.

## Quick Start

```bash
# Build all components
make

# Run a scheduler with KernCP
./scheduler/sche_bin/scx_rusty --slice-us 20000

# Monitor performance
./scheduler/tools/scxtop
```

## Architecture

KernCP combines multiple technologies to create an intelligent kernel control system:

- **eBPF/sched_ext**: Runtime kernel programmability for schedulers and monitoring
- **MCP (Model Context Protocol)**: AI-driven policy decisions based on system state
- **Real-time telemetry**: Performance counters, DAMON memory stats, energy monitoring
- **Adaptive control loops**: Observe ’ Decide ’ Apply ’ Verify

## Key Features

- **Auto-tuning schedulers**: Dynamically adjust scheduler parameters based on workload
- **Sysctl optimization**: Automatic kernel parameter tuning without manual configuration
- **Energy awareness**: Balance performance with power consumption goals
- **Multi-workload support**: From interactive desktops to batch processing clusters
- **Safety mechanisms**: A/B testing, rollback, and verification for all changes

## Documentation

- [Ecosystem & Modules](kerncp-docs/related-and-modules.md) - Related projects and expansion modules
- [Developer Guide](scheduler/scx/DEVELOPER_GUIDE.md) - Contributing to KernCP
- [Scheduler Guide](CLAUDE.md) - Working with sched_ext schedulers

## Requirements

- Linux kernel 6.12+ with sched_ext support
- Clang/LLVM >= 16 (17 recommended)
- Rust toolchain >= 1.82
- libbpf >= 1.2.2

## License

KernCP is open source software. See LICENSE for details.