#!/bin/bash

# Setup script for adding schedcp MCP server to Claude Code

echo "🚀 Setting up schedcp MCP server for Claude Code"
echo "=============================================="

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Get the absolute path of the project
PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
SERVER_BINARY="$PROJECT_DIR/target/release/schedcp"

echo -e "${GREEN}📁 Project directory: $PROJECT_DIR${NC}"

# Check if claude command exists
if ! command -v claude &> /dev/null; then
    echo -e "${RED}❌ Error: 'claude' command not found. Please install Claude Code CLI first.${NC}"
    exit 1
fi

# Check if Rust is installed
if ! command -v cargo &> /dev/null; then
    echo -e "${RED}❌ Error: Rust/Cargo not found. Please install Rust from https://rustup.rs/${NC}"
    exit 1
fi

echo -e "${GREEN}🦀 Rust/Cargo found${NC}"

# Check if scheduler binaries exist
SCHEDULER_BIN_DIR="$(cd "$PROJECT_DIR/.." && pwd)/scheduler/sche_bin"
if [ ! -d "$SCHEDULER_BIN_DIR" ] || [ -z "$(ls -A $SCHEDULER_BIN_DIR 2>/dev/null)" ]; then
    echo -e "${YELLOW}⚠️  Warning: Scheduler binaries not found!${NC}"
    echo "Please build the schedulers first:"
    echo "  cd $(dirname "$PROJECT_DIR")"
    echo "  make"
    echo ""
fi

# Ask for scope
echo ""
echo "Choose installation scope:"
echo "1) Local (current project only)"
echo "2) User (all your projects)"
echo "3) Project (shared with team via .mcp.json)"
echo ""
read -p "Enter choice [1-3] (default: 1): " choice

case $choice in
    2)
        SCOPE="--scope user"
        SCOPE_NAME="user"
        ;;
    3)
        SCOPE="--scope project"
        SCOPE_NAME="project"
        ;;
    *)
        SCOPE=""
        SCOPE_NAME="local"
        ;;
esac

echo -e "${GREEN}📦 Installing with $SCOPE_NAME scope${NC}"

# Build the Rust server
echo ""
echo -e "${GREEN}Building schedcp MCP server...${NC}"
cd "$PROJECT_DIR"
if cargo build --release; then
    echo -e "${GREEN}✅ Successfully built schedcp MCP server!${NC}"
else
    echo -e "${RED}❌ Failed to build server${NC}"
    exit 1
fi

# Check if the binary exists
if [ ! -f "$SERVER_BINARY" ]; then
    echo -e "${RED}❌ Error: Server binary not found at $SERVER_BINARY${NC}"
    exit 1
fi

# Add the server to Claude Code
echo ""
echo -e "${GREEN}Adding schedcp server to Claude Code...${NC}"

if claude mcp add $SCOPE schedcp "$SERVER_BINARY"; then
    echo -e "${GREEN}✅ Successfully added schedcp MCP server!${NC}"
else
    echo -e "${RED}❌ Failed to add server to Claude Code${NC}"
    exit 1
fi

# Show status
echo ""
echo -e "${GREEN}Current MCP servers:${NC}"
claude mcp list

echo ""
echo -e "${GREEN}✅ Setup complete!${NC}"
echo ""
echo "You can now use schedcp in Claude Code. Try asking:"
echo "  - 'List available sched-ext schedulers'"
echo "  - 'Analyze my workload and recommend a scheduler'"
echo "  - 'Run scx_rusty scheduler with custom parameters'"
echo "  - 'Create a workload profile for my application'"
echo ""
echo -e "${YELLOW}⚠️  Security Note:${NC}"
echo "The schedcp server requires root privileges to manage kernel schedulers."
echo "Make sure you have appropriate permissions to run schedulers."
echo ""
echo "To use the CLI tool directly:"
echo "  $PROJECT_DIR/target/release/schedcp-cli --help"