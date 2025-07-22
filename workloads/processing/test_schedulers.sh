#!/bin/bash

# Simple scheduler testing script
# Tests three scenarios: ctest, fifo, and default schedulers

set -e

LOADER="./schedulers/loader"
WORKLOAD_SCRIPT="./evaluate_workloads_parallel.py"

# Check prerequisites
check_prereqs() {
    if [ ! -f "$LOADER" ]; then
        echo "Error: schedulers/loader not found. Please run from the processing directory."
        exit 1
    fi
    
    if [ "$EUID" -ne 0 ]; then
        echo "This script needs to be run with sudo for loading BPF schedulers."
        echo "Please run: sudo ./test_schedulers.sh"
        exit 1
    fi
}

# Load scheduler
load_scheduler() {
    local scheduler_path=$1
    local name=$2
    
    echo "Loading $name scheduler..."
    sudo $LOADER ./schedulers/$scheduler_path &
    SCHEDULER_PID=$!
    sleep 3
    
    if ! kill -0 $SCHEDULER_PID 2>/dev/null; then
        echo "Failed to load $name scheduler"
        return 1
    fi
    
    echo "Scheduler loaded successfully (PID: $SCHEDULER_PID)"
    return 0
}

# Unload scheduler
unload_scheduler() {
    if [ -n "$SCHEDULER_PID" ]; then
        echo "Unloading scheduler..."
        kill $SCHEDULER_PID 2>/dev/null || true
        wait $SCHEDULER_PID 2>/dev/null || true
        SCHEDULER_PID=""
        sleep 2
    fi
}

# Run workload test
run_test() {
    local name=$1
    local output_file="results_$name.json"
    
    echo "Running workload test for $name..."
    python3 $WORKLOAD_SCRIPT --save $output_file
    
    if [ $? -eq 0 ]; then
        echo "Test completed, results saved to $output_file"
    else
        echo "Test failed for $name"
    fi
}

# Test a single scheduler
test_scheduler() {
    local name=$1
    local scheduler_file=$2
    
    echo "==============================="
    echo "Testing $name scheduler"
    echo "==============================="
    
    if [ -n "$scheduler_file" ]; then
        if [ ! -f "schedulers/$scheduler_file" ]; then
            echo "Error: schedulers/$scheduler_file not found. Please run 'make' in schedulers directory first."
            return 1
        fi
        
        load_scheduler $scheduler_file $name
        if [ $? -ne 0 ]; then
            return 1
        fi
    else
        echo "Testing with default Linux CFS scheduler"
    fi
    
    run_test $name
    
    unload_scheduler
}

# Main function
main() {
    check_prereqs
    
    # Trap to ensure cleanup
    trap unload_scheduler EXIT
    
    # Test ctest scheduler
    test_scheduler "ctest" "ctest.bpf.o"
    
    # Test fifo scheduler
    test_scheduler "fifo" "fifo.bpf.o"
    
    # Test default scheduler
    test_scheduler "default" ""
    
    echo "==============================="
    echo "All tests completed!"
    echo "Check results_*.json files for detailed results"
    echo "==============================="
}

main "$@"