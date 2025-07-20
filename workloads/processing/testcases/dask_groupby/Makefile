# Dask GroupBy Test Case Makefile

.PHONY: all clean generate-data run-test analyze help

# Configuration
DATA_DIR = data
OUTPUT_DIR = output
RESULTS_DIR = results
SCRIPT_NAME = dask_groupby.py

all: generate-data run-test analyze

help:
	@echo "Dask GroupBy Test Case"
	@echo ""
	@echo "Available targets:"
	@echo "  generate-data  - Create Python script for dask groupby"
	@echo "  run-test      - Run groupby simulation with analysis"
	@echo "  analyze       - Show analysis results"
	@echo "  clean         - Remove generated files"
	@echo "  help          - Show this help"

generate-data:
	@echo "Generating Dask groupby simulation script..."
	@mkdir -p $(DATA_DIR)
	@cat > $(DATA_DIR)/$(SCRIPT_NAME) << 'EOF'
#!/usr/bin/env python3
"""
Simulated Dask DataFrame groupby with data skew.
Creates workload similar to customer analytics with hot keys.
"""

import pandas as pd
import numpy as np
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
import time

def process_group(args):
    """Process a chunk of data."""
    chunk_id, data_chunk = args
    worker_pid = mp.current_process().pid
    
    print(f"Worker {worker_pid}: Processing chunk {chunk_id} with {len(data_chunk)} records")
    
    start_time = time.time()
    
    # Simulate groupby operation
    result = data_chunk.groupby('k')['v'].sum()
    
    # Add some CPU work proportional to data size
    work_iterations = len(data_chunk) // 1000
    dummy_result = 0
    for _ in range(work_iterations):
        dummy_result += sum(range(100))
    
    end_time = time.time()
    print(f"Worker {worker_pid}: Completed chunk {chunk_id} in {end_time - start_time:.2f}s")
    
    return result

if __name__ == '__main__':
    print("Starting Dask-style groupby simulation with data skew...")
    
    # Create skewed data: most keys have few records, one key has many
    print("Creating synthetic data...")
    
    # 99 normal keys with 1K records each
    normal_data = []
    for k in range(99):
        chunk = pd.DataFrame({
            'k': [k] * 1000,
            'v': np.random.randint(1, 100, 1000)
        })
        normal_data.append(chunk)
    
    # 1 hot key with 500K records (500x larger)
    hot_data = pd.DataFrame({
        'k': [999] * 500000,
        'v': np.random.randint(1, 100, 500000)
    })
    
    # Combine and split into chunks for parallel processing
    all_data = pd.concat(normal_data + [hot_data], ignore_index=True)
    
    # Split into 2 chunks (simulating 2 workers)
    # One chunk will get most of the hot key data
    chunk_size = len(all_data) // 2
    chunks = [
        (0, all_data[:chunk_size]),
        (1, all_data[chunk_size:])
    ]
    
    print(f"Data created: {len(all_data)} total records")
    print(f"Chunk 0: {len(chunks[0][1])} records")
    print(f"Chunk 1: {len(chunks[1][1])} records")
    print("Processing with 2 workers...")
    
    start_time = time.time()
    
    # Process chunks in parallel
    with ProcessPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(process_group, chunks))
    
    end_time = time.time()
    
    print(f"\nGroupby completed in {end_time - start_time:.2f} seconds")
    print("Results summary:")
    for i, result in enumerate(results):
        print(f"  Chunk {i}: {len(result)} unique keys")
EOF
	@chmod +x $(DATA_DIR)/$(SCRIPT_NAME)
	@echo "Simulation script created: $(DATA_DIR)/$(SCRIPT_NAME)"

run-test:
	@echo "Running Dask groupby simulation test..."
	@mkdir -p $(OUTPUT_DIR) $(RESULTS_DIR)
	@echo "Starting analysis and groupby simulation..."
	@cd $(DATA_DIR) && \
	python3 ../../common/analyze.py "python3 $(SCRIPT_NAME)" \
		> ../$(RESULTS_DIR)/groupby_log.txt 2>&1
	@mv $(DATA_DIR)/process_analysis.json $(RESULTS_DIR)/ 2>/dev/null || true
	@echo "Test completed. Results saved to $(RESULTS_DIR)/"

analyze:
	@echo "=== Dask GroupBy Test Analysis ==="
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