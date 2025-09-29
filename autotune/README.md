# Autotune

A command-line tool for AI-assisted command optimization and scheduler tuning, integrated with Claude AI and the schedcp MCP server.

## Overview

Autotune helps optimize command execution by:
- Running commands and collecting performance metrics
- Analyzing execution results with Claude AI
- Providing optimization suggestions
- Offering scheduler optimization for compute-intensive workloads

## Installation

Build from source:
```bash
cargo build --release
```

The binary will be available at `target/release/autotune`.

## Usage

### Basic Command Optimization

Run a command and get optimization suggestions:
```bash
autotune run <command>
```

Run with interactive Claude session:
```bash
autotune run --interactive <command>
```

### Scheduler Optimization (cc command)

For compute-intensive workloads, use the `cc` subcommand for scheduler optimization:
```bash
autotune cc <command>
```

This will:
- Analyze the workload characteristics
- Create a workload profile
- Test multiple schedulers
- Provide recommendations for optimal scheduler configuration

### Daemon Mode

Start the autotune daemon:
```bash
autotune daemon
```

Submit commands to the running daemon:
```bash
autotune submit <command>
autotune submit --interactive <command>
```

## Commands

- `run` - Execute a command locally and get optimization suggestions
- `cc` - Run scheduler optimization for compute workloads
- `submit` - Submit a command to the running daemon
- `daemon` - Start the autotune daemon service

## Dependencies

- Claude CLI must be installed and available in PATH
- For scheduler optimization: schedcp MCP server and sched-ext framework
- Rust 1.82+ for building

## Architecture

The project consists of:
- **CLI interface** (`src/bin/cli.rs`) - Command-line interface and argument parsing
- **Daemon module** (`src/daemon.rs`) - Command execution and Claude integration
- **Prompt module** (`src/prompt.rs`) - AI prompt generation for different optimization scenarios
- **Library interface** (`src/lib.rs`) - Public API for the core modules

## Integration

Autotune integrates with:
- **Claude AI** - For optimization analysis and suggestions
- **schedcp MCP server** - For advanced scheduler optimization
- **sched-ext framework** - For kernel scheduler management

## Examples

Basic command optimization:
```bash
autotune run "make -j8"
```

Interactive optimization session:
```bash
autotune run --interactive "python train.py"
```

Scheduler optimization for compilation:
```bash
autotune cc "make -j$(nproc)"
```