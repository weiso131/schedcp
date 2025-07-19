#!/bin/bash

# Setup script for Claude Desktop integration

set -e

echo "Setting up SchedCP MCP server for Claude Desktop..."

# Build the MCP server
echo "Building MCP server..."
cargo build --release

# Get the binary path
BINARY_PATH="$(pwd)/target/release/schedcp-mcp"

# Create Claude Desktop config directory if it doesn't exist
CONFIG_DIR="$HOME/Library/Application Support/Claude"
mkdir -p "$CONFIG_DIR"

CONFIG_FILE="$CONFIG_DIR/claude_desktop_config.json"

# Check if config file exists
if [ -f "$CONFIG_FILE" ]; then
    echo "Updating existing Claude Desktop configuration..."
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
    echo "Creating new Claude Desktop configuration..."
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

echo "Claude Desktop configuration updated successfully!"
echo "MCP server binary: $BINARY_PATH"
echo "Configuration file: $CONFIG_FILE"
echo ""
echo "Please restart Claude Desktop to load the new MCP server."