#!/bin/bash
# Comprehensive evaluation script for scheduler paper

# Check if running as root (required for scheduler operations)
if [ "$EUID" -ne 0 ]; then 
    echo "Please run as root (required for scheduler operations)"
    exit 1
fi

# Create results directory with timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR="evaluation_results_${TIMESTAMP}"
mkdir -p $RESULTS_DIR

echo "Starting comprehensive scheduler evaluation..."
echo "Results will be saved to: $RESULTS_DIR"

# Build schedulers if needed
echo "Building schedulers..."
cd /root/yunwei37/ai-os/scheduler
make clean && make

# Function to run experiment with proper logging
run_experiment() {
    local scheduler=$1
    local workload=$2
    local config=$3
    local output_file=$4
    
    echo "Running: $scheduler with $workload ($config)"
    
    # Ensure no other scheduler is running
    pkill -f scx_ || true
    sleep 2
    
    # Run the experiment
    timeout 400 python3 /root/yunwei37/ai-os/workloads/basic/evaluation_framework.py \
        --scheduler "$scheduler" \
        --workload "$workload" \
        --config "$config" \
        --output "$output_file" 2>&1 | tee "${RESULTS_DIR}/${output_file}.log"
    
    # Cool down period
    sleep 10
}

# 1. Baseline experiments with different schedulers
echo "Phase 1: Baseline Performance Evaluation"
for scheduler in scx_lavd scx_rusty scx_bpfland scx_simple; do
    # CPU-intensive workload
    run_experiment "$scheduler" \
        "schbench -m 2 -t 16 -r 30" \
        "baseline" \
        "baseline_${scheduler}_cpu"
    
    # Memory-intensive workload  
    run_experiment "$scheduler" \
        "stress-ng --vm 4 --vm-bytes 1G --timeout 30" \
        "baseline" \
        "baseline_${scheduler}_memory"
done

# 2. Varying load experiments
echo "Phase 2: Load Sensitivity Analysis"
for threads in 1 2 4 8 16 32; do
    run_experiment "scx_lavd" \
        "schbench -m 2 -t $threads -r 30" \
        "threads_$threads" \
        "load_sensitivity_threads_$threads"
done

# 3. Phase-changing workload
echo "Phase 3: Dynamic Workload Adaptation"
cat > ${RESULTS_DIR}/phase_workload.sh << 'EOF'
#!/bin/bash
# Phase 1: CPU intensive (30s)
stress-ng --cpu 8 --timeout 30 &
PID1=$!

# Phase 2: Memory intensive (30s)
sleep 30
kill $PID1
stress-ng --vm 4 --vm-bytes 1G --timeout 30 &
PID2=$!

# Phase 3: Mixed (30s)
sleep 30
kill $PID2
stress-ng --cpu 4 --vm 2 --vm-bytes 512M --timeout 30
EOF

chmod +x ${RESULTS_DIR}/phase_workload.sh
run_experiment "scx_lavd" \
    "${RESULTS_DIR}/phase_workload.sh" \
    "phase_changing" \
    "dynamic_phase_changing"

# 4. Collect detailed metrics
echo "Phase 4: Detailed Metrics Collection"

# Run with detailed tracing
echo "Collecting BPF trace data..."
bpftrace -e '
tracepoint:sched:sched_switch {
    @switches = count();
    @avg_runtime[comm] = avg(nsecs - arg1);
}
interval:s:1 {
    print(@switches);
    print(@avg_runtime);
    clear(@switches);
}
' > ${RESULTS_DIR}/bpf_trace.log &
BPF_PID=$!

# Run experiment with tracing
run_experiment "scx_rusty" \
    "schbench -m 2 -t 16 -r 30" \
    "with_tracing" \
    "detailed_metrics"

kill $BPF_PID

# 5. Generate performance report
echo "Phase 5: Generating Performance Report"

cat > ${RESULTS_DIR}/generate_report.py << 'EOF'
import json
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load all result files
results = []
for file in glob.glob("baseline_*.json"):
    with open(file, 'r') as f:
        results.append(json.load(f))

# Generate summary statistics
print("=== Scheduler Performance Summary ===")
for result in results:
    scheduler = result['scheduler']
    metrics = result['metrics']
    
    cpu_usage = [100 - (m['cpu_idle'] / m['cpu_total'] * 100) for m in metrics]
    avg_cpu = np.mean(cpu_usage)
    std_cpu = np.std(cpu_usage)
    
    print(f"{scheduler}:")
    print(f"  Average CPU Usage: {avg_cpu:.2f}% ± {std_cpu:.2f}%")
    print(f"  Duration: {result['duration']}s")
    print()

# Generate comparison plot
plt.figure(figsize=(12, 8))
# Add plotting code here
plt.savefig("performance_summary.pdf")
EOF

cd $RESULTS_DIR
python3 generate_report.py

# 6. Statistical analysis
echo "Phase 6: Statistical Analysis"

cat > ${RESULTS_DIR}/statistical_analysis.R << 'EOF'
# R script for statistical analysis
library(tidyverse)
library(ggplot2)

# Load data
data <- read.csv("combined_results.csv")

# Perform Kruskal-Wallis test
kruskal.test(throughput ~ scheduler, data = data)

# Pairwise comparisons
pairwise.wilcox.test(data$throughput, data$scheduler, p.adjust.method = "bonferroni")

# Effect size calculation
library(effsize)
cliff.delta(data$throughput[data$scheduler == "scx_lavd"],
            data$throughput[data$scheduler == "scx_rusty"])
EOF

echo "Evaluation complete! Results saved to: $RESULTS_DIR"
echo "To generate figures, run: python3 ${RESULTS_DIR}/generate_report.py"