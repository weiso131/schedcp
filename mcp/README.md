# bpftrace MCP Server

A Model Context Protocol (MCP) server that provides AI assistants with access to bpftrace kernel tracing capabilities. This is part of the [ai-os](https://github.com/yunwei37/ai-os) project.

## Overview

This MCP server acts as a secure bridge between AI assistants and the bpftrace tool, enabling kernel-level system observation and debugging through natural language interactions.

## Features

- **AI-Powered Kernel Debugging**: Enable AI assistants to help debug complex Linux kernel issues through natural language
- **Discover System Trace Points**: Browse and search through kernel probes to monitor system behavior
- **Rich Context and Examples**: Access production-ready bpftrace scripts for common debugging scenarios
- **Secure Execution Model**: Run kernel traces safely without giving AI direct root access
- **Asynchronous Operation**: Start long-running traces and retrieve results later
- **System Capability Detection**: Automatically discover kernel tracing features and capabilities

## Quick Start

### Prerequisites

1. Install Rust:
```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

2. Install bpftrace:
```bash
sudo apt-get install bpftrace  # Ubuntu/Debian
# or
sudo dnf install bpftrace      # Fedora
```

3. Build the server:
```bash
cargo build --release
```

### Setup

Use the automated setup scripts:

- **Claude Desktop**: `./setup_claude.sh`
- **Claude Code**: `./setup_claude_code.sh`

### Running

```bash
# Direct execution
./target/release/bpftrace-mcp-server

# Through cargo
cargo run --release
```

## Available Tools

1. **list_probes** - Lists available bpftrace probes with optional filtering
2. **bpf_info** - Shows bpftrace system information and capabilities
3. **exec_program** - Executes bpftrace programs asynchronously
4. **get_result** - Retrieves execution results

## Security Configuration

The server requires sudo access for bpftrace. Two options:

1. **Password File** (development):
   ```bash
   echo "BPFTRACE_PASSWD=your_sudo_password" > .env
   ```

2. **Passwordless sudo** (recommended):
   ```bash
   sudo visudo
   # Add: your_username ALL=(ALL) NOPASSWD: /usr/bin/bpftrace
   ```

## Architecture

Built with:
- Rust and the `rmcp` crate for MCP protocol implementation
- Tokio async runtime for concurrent operations
- Thread-safe in-memory execution buffering
- Automatic cleanup of old execution buffers

## Development

For development guidance and implementation details, see:
- [ai-os Documentation](https://github.com/yunwei37/ai-os)
- Source code in `src/main.rs`
- Example bpftrace scripts in `src/tools/`

## License

Part of the ai-os project. See the main project repository for license information.