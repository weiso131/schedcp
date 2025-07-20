# Spark Shuffle (Simulated) Test Case Makefile

.PHONY: all clean generate-data run-test analyze help

# Configuration
DATA_DIR = /tmp/spark_shuffle_data
OUTPUT_DIR = /tmp/spark_shuffle_output
RESULTS_DIR = /tmp/spark_shuffle_results
SCRIPT_NAME = skewed_workload.py

all: generate-data run-test analyze

help:
	@echo "Spark Shuffle (Simulated) Test Case"
	@echo ""
	@echo "Available targets:"
	@echo "  generate-data  - Create Python test script"
	@echo "  run-test      - Run shuffle simulation with analysis"
	@echo "  analyze       - Show analysis results"
	@echo "  clean         - Remove generated files"
	@echo "  help          - Show this help"

generate-data:
	@echo "Generating skewed workload simulation script..."
	@mkdir -p $(DATA_DIR)
	@cat > $(DATA_DIR)/$(SCRIPT_NAME) << 'EOF'
#!/usr/bin/env python3
"""
Simulated Spark shuffle with data skew.
Creates workload similar to analytics with hot keys.
"""

import multiprocessing as mp
import time
import random

def process_partition(partition_data):
    """Simulate processing a data partition."""
    partition_id, record_count = partition_data
    
    # Simulate processing time proportional to data size
    # Hot key (partition 999) has 10x more data
    processing_time = record_count / 10000  # Scale factor for reasonable test time
    
    print(f"Worker {mp.current_process().pid}: Processing partition {partition_id} with {record_count} records")
    
    # Simulate actual work
    start_time = time.time()
    result = 0
    while time.time() - start_time < processing_time:
        result += sum(range(1000))  # CPU-intensive work
    
    print(f"Worker {mp.current_process().pid}: Completed partition {partition_id} in {time.time() - start_time:.2f}s")
    return (partition_id, result)

if __name__ == '__main__':
    print("Starting skewed data processing simulation...")
    
    # Create skewed data: 99 small partitions + 1 huge partition (hot key)
    partitions = []
    
    # 9 small partitions with 1K records each
    for i in range(9):
        partitions.append((i, 1000))
    
    # 1 hot partition with 10K records (10x larger)
    partitions.append((999, 10000))
    
    print(f"Created {len(partitions)} partitions")
    print("Small partitions: 9 x 1K records")
    print("Hot partition: 1 x 10K records (10x skew)")
    
    # Process with 2 workers to simulate 2-CPU system
    start_time = time.time()
    
    with mp.Pool(2) as pool:
        results = pool.map(process_partition, partitions)
    
    end_time = time.time()
    
    print(f"\nProcessing completed in {end_time - start_time:.2f} seconds")
    print(f"Processed {len(results)} partitions")
	
	# Show results summary
    small_partitions = [r for r in results if r[0] != 999]
    hot_partition = [r for r in results if r[0] == 999]
    
    print(f"Small partitions processed: {len(small_partitions)}")
    print(f"Hot partition processed: {len(hot_partition)}")
EOF
	@chmod +x $(DATA_DIR)/$(SCRIPT_NAME)
	@echo "Simulation script created: $(DATA_DIR)/$(SCRIPT_NAME)"

run-test:
	@echo "Running Spark shuffle simulation test..."
	@mkdir -p $(OUTPUT_DIR) $(RESULTS_DIR)
	@echo "Starting analysis and workload simulation..."
	@cd $(DATA_DIR) && \
	python3 ../../common/analyze.py "python3 $(SCRIPT_NAME)" \
		> ../$(RESULTS_DIR)/shuffle_log.txt 2>&1
	@mv $(DATA_DIR)/process_analysis.json $(RESULTS_DIR)/ 2>/dev/null || true
	@echo "Test completed. Results saved to $(RESULTS_DIR)/"

analyze:
	@echo "=== Spark Shuffle Simulation Test Analysis ==="
	@echo ""
	@if [ -f $(RESULTS_DIR)/process_analysis.json ]; then \
		python3 ../common/analyze_results.py $(RESULTS_DIR)/process_analysis.json; \
	else \
		echo "No results found. Run 'make run-test' first."; \
	fi

clean:
	@echo "Cleaning up generated files..."
	@rm -rf $(DATA_DIR) $(OUTPUT_DIR) $(RESULTS_DIR)
	@echo "Cleanup complete."

debug-data:
	@echo "Data directory contents:"
	@if [ -d $(DATA_DIR) ]; then ls -lh $(DATA_DIR)/; else echo "No data directory found."; fi

debug-results:
	@echo "Results directory contents:"
	@if [ -d $(RESULTS_DIR) ]; then ls -lh $(RESULTS_DIR)/; else echo "No results directory found."; fi