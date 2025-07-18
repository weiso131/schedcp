# AI-OS MCP Server

A minimal Model Context Protocol (MCP) server implementation in Rust for AI-OS.

## Overview

This is a simple demo MCP server that provides a counter service with basic operations:
- Increment counter
- Decrement counter
- Get current value
- Reset counter

## Building

```bash
cd mcp
cargo build --release
```

The binary will be available at `target/release/ai-os-mcp`.

## Running

The MCP server communicates via stdio:

```bash
./target/release/ai-os-mcp
```

## Available Tools

1. **increment** - Increment the counter by a specified amount (default: 1)
2. **decrement** - Decrement the counter by a specified amount (default: 1)
3. **get_value** - Get the current counter value
4. **reset** - Reset the counter to zero

## Integration with Claude

To use this MCP server with Claude Desktop or Claude Code, add it to your MCP configuration.

### Claude Desktop Configuration

Add to `~/Library/Application Support/Claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "ai-os": {
      "command": "/root/yunwei37/ai-os/mcp/target/release/ai-os-mcp"
    }
  }
}
```

### Claude Code Configuration

Add to `~/.claude_code/settings.json`:

```json
{
  "mcpServers": {
    "ai-os": {
      "command": "/root/yunwei37/ai-os/mcp/target/release/ai-os-mcp"
    }
  }
}
```

## Development

This is a minimal implementation for demonstration purposes. Future enhancements could include:
- Integration with AI-OS scheduler management
- Real-time scheduler monitoring
- Benchmark execution tools
- Performance metrics collection