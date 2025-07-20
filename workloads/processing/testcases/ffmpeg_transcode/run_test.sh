#!/bin/bash
# FFmpeg Transcode Test Runner

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== FFmpeg Transcode Long-tail Test ==="
echo "Real-world scenario: Video platform processing mixed-length uploads"
echo ""

# Check if ffmpeg is available
if ! command -v ffmpeg &> /dev/null; then
    echo "Error: ffmpeg is not installed. Please install it first:"
    echo "  Ubuntu/Debian: sudo apt-get install ffmpeg"
    echo "  CentOS/RHEL: sudo yum install ffmpeg"
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

echo "1. Generating test video data..."
make generate-data

echo ""
echo "2. Running transcode test with process monitoring..."
echo "   This will transcode 10 short videos (1s each) and 1 long video (30s)"
echo "   Expected behavior: Long video transcoding takes much longer"

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
echo "- 10 short videos transcode quickly (should finish in seconds)"
echo "- 1 long video takes much longer (should be the bottleneck)"
echo "- With proper scheduling, the long video can be isolated to one CPU"
echo "  while short videos utilize the other CPU efficiently"
echo ""
echo "Check results/ directory for detailed analysis."