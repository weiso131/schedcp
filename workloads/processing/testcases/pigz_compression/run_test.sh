#!/bin/bash
# Pigz Compression Test Runner
# This script runs the pigz compression test and analyzes the results

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== Pigz Compression Long-tail Test ==="
echo "Real-world scenario: Backup systems compressing mixed-size files"
echo ""

# Check if pigz is available
if ! command -v pigz &> /dev/null; then
    echo "Error: pigz is not installed. Please install it first:"
    echo "  Ubuntu/Debian: sudo apt-get install pigz"
    echo "  CentOS/RHEL: sudo yum install pigz"
    exit 1
fi

# Check if python3 is available
if ! command -v python3 &> /dev/null; then
    echo "Error: python3 is not installed."
    exit 1
fi

# Install required Python packages if not available
python3 -c "import psutil" 2>/dev/null || {
    echo "Installing required Python package: psutil"
    pip3 install psutil --user
}

echo "1. Generating test data..."
make generate-data

echo ""
echo "2. Running compression test with process monitoring..."
echo "   This will compress 99 small files (1MB each) and 1 large file (2GB)"
echo "   Expected behavior: Large file compression takes much longer"

# Record start time
start_time=$(date +%s)

make run-test

# Record end time
end_time=$(date +%s)
total_time=$((end_time - start_time))

echo ""
echo "3. Test completed in ${total_time} seconds"
echo ""
echo "4. Analyzing results..."
make analyze

echo ""
echo "=== Test Summary ==="
echo "This test demonstrates the long-tail problem where:"
echo "- 99 small files compress quickly (should finish in seconds)"
echo "- 1 large file takes much longer (should be the bottleneck)"
echo "- With proper scheduling, the large file can be isolated to one CPU"
echo "  while small files utilize the other CPU efficiently"
echo ""
echo "Check results/ directory for detailed analysis."