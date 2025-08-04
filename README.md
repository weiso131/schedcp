# schedcp

schedcp is  an MCP (Model Context Protocol) server that enables AI assistants to intelligently manage Linux kernel schedulers. It provides real-time workload analysis and automatic scheduler optimization through the sched_ext (scx) framework.

## Features

- **AI-Powered Scheduler Selection**: Natural language workload descriptions for intelligent scheduler recommendations
- **Workload Profiling**: Create and manage workload profiles with execution history tracking
- **Real-time Management**: Start, stop, and monitor sched_ext schedulers dynamically
- **Performance Tracking**: Capture and analyze scheduler performance metrics across different workloads
- **MCP Integration**: Seamless integration with AI assistants like Claude for automated optimization

## Quick Start

### Building schedcp

```bash
# Clone the repository
git clone https://github.com/yunwei37/ai-os
cd ai-os

# Build the MCP server and CLI
cd mcp
cargo build --release
# The binaries will be at:
# - mcp/target/release/schedcp (MCP server)
# - mcp/target/release/schedcp-cli (CLI tool)
```

### Installing the MCP Server

For Claude Code:

```bash
# Add to Claude Code configuration
claude mcp add schedcp /path/to/ai-os/mcp/target/release/schedcp
```

### Using schedcp CLI

```bash
# List available schedulers
./mcp/target/release/schedcp-cli list

# Run a scheduler
./mcp/target/release/schedcp-cli run scx_rusty --slice-us 20000

# Stop the running scheduler
./mcp/target/release/schedcp-cli stop

# Check scheduler status
./mcp/target/release/schedcp-cli status
```

### Building Kernel Schedulers

```bash
# Build all schedulers (outputs to scheduler/sche_bin/)
make

# Build only C schedulers
make build-c

# Build only Rust schedulers
make build-rust

# Build tools (scxtop, scxctl)
make build-tools
```

## Architecture

schedcp combines multiple technologies to create an intelligent kernel control system:

- **eBPF/sched_ext**: Runtime kernel programmability for schedulers and monitoring
- **MCP (Model Context Protocol)**: AI-driven policy decisions based on system state
- **Real-time telemetry**: Performance counters, DAMON memory stats, energy monitoring
- **Adaptive control loops**: Observe � Decide � Apply � Verify

## Key Features

- **Auto-tuning schedulers**: Dynamically adjust scheduler parameters based on workload
- **Sysctl optimization**: Automatic kernel parameter tuning without manual configuration
- **Energy awareness**: Balance performance with power consumption goals
- **Multi-workload support**: From interactive desktops to batch processing clusters
- **Safety mechanisms**: A/B testing, rollback, and verification for all changes

## Documentation

- [Ecosystem & Modules](kerncp-docs/related-and-modules.md) - Related projects and expansion modules
- [Developer Guide](scheduler/scx/DEVELOPER_GUIDE.md) - Contributing to schedcp
- [Scheduler Guide](CLAUDE.md) - Working with sched_ext schedulers

## Requirements

- Linux kernel 6.12+ with sched_ext support
- Clang/LLVM >= 16 (17 recommended)
- Rust toolchain >= 1.82
- libbpf >= 1.2.2

## License

schedcp is open source software. See LICENSE for details.