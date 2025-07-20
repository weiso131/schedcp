# Flink Join Test Case Makefile

.PHONY: all clean generate-data run-test analyze help

# Configuration
DATA_DIR = data
OUTPUT_DIR = output
RESULTS_DIR = results
SCRIPT_NAME = flink_join_sim.py

all: generate-data run-test analyze

help:
	@echo "Flink Join Test Case"
	@echo ""
	@echo "Available targets:"
	@echo "  generate-data  - Create join simulation script"
	@echo "  run-test      - Run join simulation with analysis"
	@echo "  analyze       - Show analysis results"
	@echo "  clean         - Remove generated files"
	@echo "  help          - Show this help"

generate-data:
	@echo "Generating Flink join simulation script..."
	@mkdir -p $(DATA_DIR)
	@cat > $(DATA_DIR)/$(SCRIPT_NAME) << 'EOF'
#!/usr/bin/env python3
"""
Flink-style join simulation with data skew.
Simulates joining sales with product data where popular items create hot keys.
"""

import multiprocessing as mp
import time
import random

def process_join_partition(args):
    """Process a join partition (simulates Flink TaskManager slot)."""
    partition_data, slot_id = args
    key, left_records, right_records = partition_data
    
    worker_pid = mp.current_process().pid
    total_combinations = len(left_records) * len(right_records)
    
    print(f"Slot {slot_id} (PID {worker_pid}): Processing key {key} - {len(left_records)} x {len(right_records)} = {total_combinations} joins")
    
    start_time = time.time()
    
    # Simulate join processing - CPU work proportional to combinations
    result = []
    for left in left_records:
        for right in right_records:
            # Simulate join operation with some CPU work
            join_result = {
                'key': key,
                'left_id': left,
                'right_id': right,
                'score': hash((left, right)) % 1000
            }
            result.append(join_result)
            
            # Add some processing delay
            if len(result) % 10000 == 0:
                time.sleep(0.001)  # Small delay every 10K joins
    
    end_time = time.time()
    print(f"Slot {slot_id} (PID {worker_pid}): Completed key {key} in {end_time - start_time:.2f}s ({len(result)} join results)")
    
    return (key, len(result))

if __name__ == '__main__':
    print("Starting Flink-style join simulation with data skew...")
    
    # Create skewed join data
    print("Creating synthetic join data...")
    
    # Simulate sales data (left side) and product data (right side)
    partitions = []
    
    # 99 normal products with few sales each
    for product_id in range(99):
        left_records = list(range(product_id * 10, (product_id + 1) * 10))  # 10 sales per product
        right_records = [product_id]  # 1 product record
        partitions.append(((product_id, left_records, right_records), 0))  # Assign to slot 0
    
    # 1 hot product (iPhone, bestseller) with many sales
    hot_product_id = 999
    left_records = list(range(100000))  # 100K sales for hot product
    right_records = [hot_product_id]    # 1 product record
    partitions.append(((hot_product_id, left_records, right_records), 1))  # Assign to slot 1
    
    print(f"Created {len(partitions)} partitions")
    print("Normal products: 99 x ~10 sales each")
    print("Hot product: 1 x 100K sales (10,000x skew)")
    
    # Redistribute partitions to simulate Flink's task assignment
    # In reality, hot key would end up dominating one slot
    slot_0_partitions = [p for p in partitions if p[1] == 0]
    slot_1_partitions = [p for p in partitions if p[1] == 1]
    
    print(f"Slot 0 gets: {len(slot_0_partitions)} partitions")
    print(f"Slot 1 gets: {len(slot_1_partitions)} partitions")
    
    # Process partitions in parallel (simulating 2 TaskManager slots)
    start_time = time.time()
    
    with mp.Pool(2) as pool:
        results = pool.map(process_join_partition, partitions)
    
    end_time = time.time()
    
    print(f"\nJoin completed in {end_time - start_time:.2f} seconds")
    
    # Analyze results
    total_joins = sum(count for _, count in results)
    hot_key_result = next((count for key, count in results if key == 999), 0)
    normal_key_joins = total_joins - hot_key_result
    
    print(f"Total join results: {total_joins}")
    print(f"Hot key joins: {hot_key_result}")
    print(f"Normal key joins: {normal_key_joins}")
    print(f"Skew ratio: {hot_key_result / (normal_key_joins / 99):.0f}x" if normal_key_joins > 0 else "N/A")
EOF
	@chmod +x $(DATA_DIR)/$(SCRIPT_NAME)
	@echo "Join simulation script created: $(DATA_DIR)/$(SCRIPT_NAME)"

run-test:
	@echo "Running Flink join simulation test..."
	@mkdir -p $(OUTPUT_DIR) $(RESULTS_DIR)
	@echo "Starting analysis and join simulation..."
	@cd $(DATA_DIR) && \
	python3 ../../common/analyze.py "python3 $(SCRIPT_NAME)" \
		> ../$(RESULTS_DIR)/join_log.txt 2>&1
	@mv $(DATA_DIR)/process_analysis.json $(RESULTS_DIR)/ 2>/dev/null || true
	@echo "Test completed. Results saved to $(RESULTS_DIR)/"

analyze:
	@echo "=== Flink Join Test Analysis ==="
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