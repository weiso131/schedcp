# SchedCP - MCP Server for Sched-ext Scheduler Management

A Model Context Protocol (MCP) server and CLI tool for managing Linux sched-ext schedulers. This is part of the [ai-os](https://github.com/yunwei37/ai-os) project that enables AI assistants to optimize Linux kernel scheduling for specific workloads.

## Overview

SchedCP provides intelligent scheduler management through two interfaces:
- **schedcp** - MCP server enabling AI assistants (like Claude) to manage schedulers programmatically
- **schedcp-cli** - Command-line tool for direct scheduler management and testing

The MCP server allows AI assistants to:
- Analyze workload characteristics and select optimal schedulers
- Create and manage workload profiles with natural language descriptions
- Track scheduler performance across different workloads
- Make data-driven scheduler recommendations based on execution history

## Features

- **Embedded Schedulers**: All sched-ext schedulers and configuration embedded in binaries
- **Rich Information**: Detailed scheduler information including algorithms, use cases, and tuning parameters
- **Smart Filtering**: Filter schedulers by name or production readiness
- **AI Integration**: MCP server enables AI assistants to help with scheduler selection and tuning
- **Production Ready**: Clear indication of which schedulers are ready for production use
- **Custom Schedulers**: Create, compile, and verify custom BPF schedulers from source code
- **Workload Profiles**: Create and manage workload profiles with natural language descriptions
- **Execution History**: Track scheduler performance history for different workloads
- **System Monitoring**: Real-time CPU, memory, and scheduler metrics collection
- **Persistent Storage**: All workload profiles and history are saved to disk automatically
- **Enhanced Logging**: Comprehensive logging to file (schedcp.log) for debugging and monitoring

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

### Create and Run a Custom Scheduler

```bash
# Create and run a custom BPF scheduler from source
./target/release/schedcp-cli create-and-run /path/to/my_scheduler.bpf.c
```

### Monitor System Metrics

```bash
# Monitor system metrics for default 10 seconds
./target/release/schedcp-cli monitor

# Monitor for custom duration
./target/release/schedcp-cli monitor --duration 30

# Short flag
./target/release/schedcp-cli monitor -d 5
```

Output includes:
- CPU utilization (average and maximum)
- Memory usage (percentage and MB)
- Scheduler statistics (timeslices and run time)

## schedcp MCP Server Usage

### Available Tools

#### Scheduler Management
- **list_schedulers** - List schedulers with detailed information
  - Parameters: `name` (optional), `production_ready` (optional)
- **run_scheduler** - Start a scheduler with specified parameters
  - Parameters: `name` (required), `args` (optional array)
- **stop_scheduler** - Stop a running scheduler
  - Parameters: `execution_id` (required)
- **get_execution_status** - Get status of a scheduler execution (includes command and args)
  - Parameters: `execution_id` (required)

#### Custom Scheduler Creation
- **create_and_verify_scheduler** - Create, compile, and verify a custom BPF scheduler
  - Parameters: `name` (required), `source_code` (required), `description`, `algorithm`, `use_cases`, `characteristics`, `limitations`, `performance_profile`

#### Workload Profile Management
- **workload** - Unified workload profile management command
  - Commands: `create`, `list`, `get_history`, `add_history`
  - Parameters vary by command

#### System Monitoring
- **system_monitor** - Collect CPU, memory, and scheduler metrics
  - Commands: `start` (begin monitoring), `stop` (end and get summary)
  - Collects samples every second including:
    - CPU utilization percentages
    - Memory usage (percent and MB)
    - Scheduler statistics from /proc/schedstat

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

### Workflow Examples

#### Workload Profile Workflow

```bash
# 1. Create a workload profile
# AI assistant creates a profile for a web server workload

# 2. Run a scheduler
# AI assistant runs scx_bpfland for the workload

# 3. Add execution history
# AI assistant adds the execution results to the workload profile

# 4. View workload history
# AI assistant can retrieve all past scheduler executions for the workload
```

The workload profiles and history are persisted in `schedcp_workloads.json` in the current directory.

#### System Monitoring Workflow

```bash
# 1. Start monitoring
# AI assistant starts monitoring with: {"command": "start"}

# 2. Run workload/scheduler
# Execute the workload or benchmark while monitoring is active

# 3. Stop and get summary
# AI assistant stops monitoring with: {"command": "stop"}
# Receives comprehensive summary with CPU, memory, and scheduler metrics
```

#### Custom Scheduler Workflow

```bash
# 1. Create custom scheduler
# AI assistant creates and verifies a custom BPF scheduler from source code

# 2. Run custom scheduler
# The custom scheduler is now available in the scheduler list

# 3. Monitor performance
# Use system_monitor to track performance of the custom scheduler
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