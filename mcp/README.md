# SchedCP - Sched-ext Scheduler Management Tools

A Model Context Protocol (MCP) server and CLI tool for managing Linux sched-ext schedulers. This is part of the [ai-os](https://github.com/yunwei37/ai-os) project.

## Overview

SchedCP provides two interfaces for managing Linux kernel schedulers:
- **schedcp-cli** - Command-line tool for direct scheduler management
- **schedcp** - MCP server enabling AI assistants to manage schedulers

All scheduler binaries and configuration are embedded in the executables, making deployment simple and self-contained.

## Features

- **Embedded Schedulers**: All sched-ext schedulers and configuration embedded in binaries
- **Rich Information**: Detailed scheduler information including algorithms, use cases, and tuning parameters
- **Smart Filtering**: Filter schedulers by name or production readiness
- **AI Integration**: MCP server enables AI assistants to help with scheduler selection and tuning
- **Production Ready**: Clear indication of which schedulers are ready for production use

## Quick Start

### Prerequisites

1. Linux kernel 6.12+ with sched-ext support
2. Rust toolchain:
```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

### Build

```bash
cargo build --release
```

## schedcp-cli Usage

### List Schedulers

```bash
# List all schedulers with full details
./target/release/schedcp-cli list

# Filter by name (partial match)
./target/release/schedcp-cli list --name rusty

# List only production-ready schedulers
./target/release/schedcp-cli list --production

# Combine filters
./target/release/schedcp-cli list --production --name simple
```

### Run a Scheduler

```bash
# Set sudo password in environment variable
export SCHEDCP_SUDO_PASSWORD="your_password"

# Run with default parameters
./target/release/schedcp-cli run scx_rusty --sudo

# Run with custom parameters
./target/release/schedcp-cli run scx_rusty --sudo -- --slice-us-underutil 30000 --fifo-sched
```

## schedcp MCP Server Usage

### Available Tools

- **list_schedulers** - List schedulers with detailed information
  - Parameters: `name` (optional), `production_ready` (optional)
- **run_scheduler** - Start a scheduler with specified parameters
  - Parameters: `name` (required), `args` (optional array)
- **stop_scheduler** - Stop a running scheduler
  - Parameters: `execution_id` (required)
- **get_execution_status** - Get status of a scheduler execution
  - Parameters: `execution_id` (required)

### Running the Server

```bash
# Set sudo password in environment variable
export SCHEDCP_SUDO_PASSWORD="your_password"

# Run the MCP server
./target/release/schedcp
```

### Claude Desktop Configuration

Add to your Claude Desktop configuration:

```json
{
  "mcpServers": {
    "schedcp": {
      "command": "/path/to/schedcp",
      "env": {
        "SCHEDCP_SUDO_PASSWORD": "your_password"
      }
    }
  }
}
```

## Available Schedulers

### Production-Ready Schedulers
- **scx_rusty** - Multi-domain scheduler with intelligent load balancing
- **scx_simple** - Simple scheduler for single-socket systems
- **scx_lavd** - Latency-aware scheduler for gaming and interactive workloads
- **scx_bpfland** - Interactive workload prioritization
- **scx_layered** - Highly configurable multi-layer scheduler
- **scx_flatcg** - High-performance cgroup-aware scheduler
- **scx_nest** - Frequency-optimized scheduler for low CPU utilization
- **scx_flash** - EDF scheduler for predictable latency
- **scx_p2dq** - Versatile scheduler with pick-two load balancing

### Experimental Schedulers
- Various experimental and educational schedulers for testing and development

## Security Configuration

The tools require sudo access to load kernel schedulers. Two options:

1. **Environment Variable** (development):
   ```bash
   export SCHEDCP_SUDO_PASSWORD="your_password"
   ```

2. **Passwordless sudo** (recommended for production):
   ```bash
   sudo visudo
   # Add: your_username ALL=(ALL) NOPASSWD: /path/to/scheduler/binaries
   ```

## Architecture

- **Rust** with embedded resources using `rust-embed`
- **Async runtime** using Tokio for concurrent operations
- **MCP protocol** implementation using `rmcp` crate
- **Self-contained** with all schedulers and configuration embedded

## Development

For more information:
- [ai-os Documentation](https://github.com/yunwei37/ai-os)
- [sched-ext Documentation](https://github.com/sched-ext/scx)
- Source code in `src/`

## License

Part of the ai-os project. See the main project repository for license information.