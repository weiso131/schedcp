#!/bin/bash

# Setup script for Claude Code integration

set -e

echo "Setting up SchedCP MCP server for Claude Code..."

# Build the MCP server
echo "Building MCP server..."
cargo build --release

# Get the binary path
BINARY_PATH="$(pwd)/target/release/schedcp-mcp"

# Create Claude Code config directory if it doesn't exist
CONFIG_DIR="$HOME/.claude_code"
mkdir -p "$CONFIG_DIR"

CONFIG_FILE="$CONFIG_DIR/settings.json"

# Check if config file exists
if [ -f "$CONFIG_FILE" ]; then
    echo "Updating existing Claude Code configuration..."
    # Backup existing config
    cp "$CONFIG_FILE" "$CONFIG_FILE.bak"
    
    # Update config using a temporary file
    cat "$CONFIG_FILE" | python3 -c "
import sys
import json

config = json.load(sys.stdin)
if 'mcpServers' not in config:
    config['mcpServers'] = {}

config['mcpServers']['schedcp'] = {
    'command': '$BINARY_PATH'
}

print(json.dumps(config, indent=2))
" > "$CONFIG_FILE.tmp"
    
    mv "$CONFIG_FILE.tmp" "$CONFIG_FILE"
else
    echo "Creating new Claude Code configuration..."
    cat > "$CONFIG_FILE" << EOF
{
  "mcpServers": {
    "schedcp": {
      "command": "$BINARY_PATH"
    }
  }
}
EOF
fi

echo "Claude Code configuration updated successfully!"
echo "MCP server binary: $BINARY_PATH"
echo "Configuration file: $CONFIG_FILE"
echo ""
echo "The MCP server will be available when you start Claude Code."