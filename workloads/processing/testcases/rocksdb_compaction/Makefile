# RocksDB Compaction Test Case Makefile

.PHONY: all clean generate-data run-test analyze help

# Configuration
DATA_DIR = data
OUTPUT_DIR = output
RESULTS_DIR = results
SCRIPT_NAME = rocksdb_sim.py

all: generate-data run-test analyze

help:
	@echo "RocksDB Compaction Test Case"
	@echo ""
	@echo "Available targets:"
	@echo "  generate-data  - Create RocksDB simulation script"
	@echo "  run-test      - Run compaction simulation with analysis"
	@echo "  analyze       - Show analysis results"
	@echo "  clean         - Remove generated files"
	@echo "  help          - Show this help"

generate-data:
	@echo "Generating RocksDB compaction simulation script..."
	@mkdir -p $(DATA_DIR)
	@cat > $(DATA_DIR)/$(SCRIPT_NAME) << 'EOF'
#!/usr/bin/env python3
"""
RocksDB compaction simulation.
Simulates database compaction with mixed file sizes.
"""

import multiprocessing as mp
import time
import random
import os

def simulate_compaction(args):
    """Simulate compacting a set of SST files."""
    file_group, worker_id = args
    worker_pid = mp.current_process().pid
    
    file_sizes = [f['size'] for f in file_group]
    file_count = len(file_group)
    total_size = sum(file_sizes)
    
    print(f"Worker {worker_pid}: Compacting {file_count} files, total size: {total_size/1024/1024:.1f}MB")
    
    start_time = time.time()
    
    # Simulate compaction work - CPU intensive operation
    # Processing time roughly proportional to data size
    work_units = total_size // 1024  # Work units based on KB
    
    result_data = []
    for i in range(work_units):
        # Simulate reading, merging, and writing data
        dummy_work = 0
        for _ in range(100):  # CPU work per unit
            dummy_work += hash(i) % 1000
        
        result_data.append(dummy_work)
        
        # Add small delay every 1000 units to simulate I/O
        if i % 1000 == 0 and i > 0:
            time.sleep(0.001)
    
    # Simulate writing output file
    output_size = int(total_size * 0.8)  # Compaction typically reduces size
    
    end_time = time.time()
    print(f"Worker {worker_pid}: Completed compaction in {end_time - start_time:.2f}s (output: {output_size/1024/1024:.1f}MB)")
    
    return {
        'input_files': file_count,
        'input_size': total_size,
        'output_size': output_size,
        'duration': end_time - start_time
    }

if __name__ == '__main__':
    print("Starting RocksDB compaction simulation...")
    
    # Create simulated SST files with mixed sizes
    print("Creating synthetic SST file metadata...")
    
    # 99 small SST files (1-10 MB each)
    small_files = []
    for i in range(99):
        file_size = random.randint(1, 10) * 1024 * 1024  # 1-10 MB
        small_files.append({
            'name': f'small_{i:03d}.sst',
            'size': file_size,
            'level': 1
        })
    
    # 1 large SST file (1 GB)
    large_file = {
        'name': 'large_000.sst',
        'size': 1024 * 1024 * 1024,  # 1 GB
        'level': 0
    }
    
    # Group files for compaction
    # In RocksDB, files are typically grouped by level and size
    compaction_groups = [
        (small_files, 0),  # Small files go to worker 0
        ([large_file], 1)  # Large file goes to worker 1
    ]
    
    total_input_size = sum(f['size'] for f in small_files) + large_file['size']
    print(f"Total input data: {total_input_size/1024/1024/1024:.2f} GB")
    print(f"Small files: {len(small_files)} files, {sum(f['size'] for f in small_files)/1024/1024:.1f} MB")
    print(f"Large file: 1 file, {large_file['size']/1024/1024:.1f} MB")
    
    print("Starting compaction with 2 threads...")
    start_time = time.time()
    
    # Run compaction in parallel
    with mp.Pool(2) as pool:
        results = pool.map(simulate_compaction, compaction_groups)
    
    end_time = time.time()
    
    print(f"\nCompaction completed in {end_time - start_time:.2f} seconds")
    
    # Show results
    total_input = sum(r['input_size'] for r in results)
    total_output = sum(r['output_size'] for r in results)
    compression_ratio = total_output / total_input if total_input > 0 else 0
    
    print(f"Input data: {total_input/1024/1024/1024:.2f} GB")
    print(f"Output data: {total_output/1024/1024/1024:.2f} GB")
    print(f"Compression ratio: {compression_ratio:.2f}")
    
    for i, result in enumerate(results):
        print(f"Worker {i}: {result['input_files']} files, {result['duration']:.2f}s")
EOF
	@chmod +x $(DATA_DIR)/$(SCRIPT_NAME)
	@echo "RocksDB simulation script created: $(DATA_DIR)/$(SCRIPT_NAME)"

run-test:
	@echo "Running RocksDB compaction simulation test..."
	@mkdir -p $(OUTPUT_DIR) $(RESULTS_DIR)
	@echo "Starting analysis and compaction simulation..."
	@cd $(DATA_DIR) && \
	python3 ../../common/analyze.py "python3 $(SCRIPT_NAME)" \
		> ../$(RESULTS_DIR)/compaction_log.txt 2>&1
	@mv $(DATA_DIR)/process_analysis.json $(RESULTS_DIR)/ 2>/dev/null || true
	@echo "Test completed. Results saved to $(RESULTS_DIR)/"

analyze:
	@echo "=== RocksDB Compaction Test Analysis ==="
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