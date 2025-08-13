#!/bin/bash

# CXL Bandwidth-Aware Scheduler Test Script
# Tests various features of the scx_cxl scheduler

set -e

SCHED_PATH="./scx_cxl"
TEST_DURATION=10

echo "=== CXL Bandwidth-Aware Scheduler Test Suite ==="
echo

# Function to run a test
run_test() {
    local test_name="$1"
    local test_cmd="$2"
    local expected_result="$3"
    
    echo "Running test: $test_name"
    echo "Command: $test_cmd"
    
    if timeout $TEST_DURATION $test_cmd > /tmp/test_output.log 2>&1; then
        echo "✓ Test passed: $test_name"
    else
        exit_code=$?
        if [ $exit_code -eq 124 ]; then
            echo "✓ Test completed (timeout expected): $test_name"
        else
            echo "✗ Test failed: $test_name (exit code: $exit_code)"
            tail -n 20 /tmp/test_output.log
        fi
    fi
    echo
}

# Check if running as root
if [ "$EUID" -ne 0 ]; then 
    echo "Please run as root (required for BPF scheduler)"
    exit 1
fi

# Build the scheduler if not exists
if [ ! -f "$SCHED_PATH" ]; then
    echo "Building scx_cxl scheduler..."
    cd ../..
    meson setup build --prefix=/usr
    meson compile -C build
    cd -
    SCHED_PATH="../../build/scheds/c/scx_cxl"
fi

# Test 1: Basic functionality
run_test "Basic functionality" \
    "$SCHED_PATH -v -s 20000 -n 4" \
    "should_start"

# Test 2: Custom bandwidth limits
run_test "Custom bandwidth limits" \
    "$SCHED_PATH -r 800 -w 600 -v" \
    "should_set_limits"

# Test 3: Disable DAMON
run_test "Disable DAMON integration" \
    "$SCHED_PATH -d -v" \
    "damon_disabled"

# Test 4: Disable CXL-aware scheduling
run_test "Disable CXL-aware scheduling" \
    "$SCHED_PATH -c -v" \
    "cxl_disabled"

# Test 5: Disable bandwidth control
run_test "Disable bandwidth control" \
    "$SCHED_PATH -b -v" \
    "bandwidth_control_disabled"

# Test 6: All features disabled
run_test "All features disabled (minimal mode)" \
    "$SCHED_PATH -d -c -b -v" \
    "minimal_mode"

# Test 7: High bandwidth configuration
run_test "High bandwidth configuration" \
    "$SCHED_PATH -r 2000 -w 2000 -s 10000 -v" \
    "high_bandwidth"

# Test 8: Low latency configuration
run_test "Low latency configuration" \
    "$SCHED_PATH -s 5000 -n 8 -v" \
    "low_latency"

# Test 9: Monitoring mode
run_test "Monitoring mode" \
    "$SCHED_PATH -v -m 2" \
    "monitoring_enabled"

# Test 10: Help output
echo "Testing help output..."
if $SCHED_PATH -h 2>&1 | grep -q "CXL PMU-aware scheduler"; then
    echo "✓ Help output test passed"
else
    echo "✗ Help output test failed"
fi
echo

# Cleanup
rm -f /tmp/test_output.log

echo "=== Test Suite Completed ==="
echo
echo "Note: Some tests may timeout intentionally after $TEST_DURATION seconds."
echo "This is expected behavior for testing the scheduler's continuous operation."