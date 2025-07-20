# Pandas ETL Test Case Makefile

.PHONY: all clean generate-data run-test analyze help

# Configuration
DATA_DIR = /tmp/pandas_etl_data
OUTPUT_DIR = /tmp/pandas_etl_output
RESULTS_DIR = /tmp/pandas_etl_results
SCRIPT_NAME = pandas_etl.py
SMALL_FILE_COUNT = 3

all: generate-data run-test analyze

help:
	@echo "Pandas ETL Test Case"
	@echo ""
	@echo "Available targets:"
	@echo "  generate-data  - Create log files and ETL script"
	@echo "  run-test      - Run ETL simulation with analysis"
	@echo "  analyze       - Show analysis results"
	@echo "  clean         - Remove generated files"
	@echo "  help          - Show this help"

generate-data:
	@echo "Generating log files and ETL script..."
	@mkdir -p $(DATA_DIR)/logs
	@echo "Creating $(SMALL_FILE_COUNT) small log files..."
	@for i in $$(seq 1 $(SMALL_FILE_COUNT)); do \
		seq 1 500 | awk '{print strftime("%Y-%m-%d %H:%M:%S", systime()), "INFO", "Message", $$1}' | \
		gzip > $(DATA_DIR)/logs/log$$i.gz; \
	done
	@echo "Creating 1 large log file..."
	@seq 1 5000 | awk '{print strftime("%Y-%m-%d %H:%M:%S", systime()), "INFO", "Message", $$1}' | \
	gzip > $(DATA_DIR)/logs/log_large.gz
	@cat > $(DATA_DIR)/$(SCRIPT_NAME) << 'EOF'
#!/usr/bin/env python3
"""
Pandas multiprocessing ETL simulation.
Processes gzipped log files with data skew.
"""

import multiprocessing as mp
import pandas as pd
import gzip
import glob
import time
import os

def parse_log_file(args):
    """Parse a gzipped log file."""
    filename, worker_id = args
    worker_pid = mp.current_process().pid
    
    print(f"Worker {worker_pid}: Processing {filename}")
    start_time = time.time()
    
    try:
        # Read and parse the gzipped log file
        with gzip.open(filename, 'rt') as f:
            lines = f.readlines()
        
        # Simulate parsing work
        data = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 4:
                data.append({
                    'timestamp': ' '.join(parts[:2]),
                    'level': parts[2],
                    'message': ' '.join(parts[3:])
                })
        
        # Create DataFrame and do some processing
        df = pd.DataFrame(data)
        
        # Simulate ETL transformations
        df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
        summary = df.groupby(['level', 'hour']).size().reset_index(name='count')
        
        # Add some CPU-intensive work proportional to file size
        work_iterations = len(data) // 50
        dummy_result = 0
        for _ in range(work_iterations):
            dummy_result += sum(range(25))
        
        end_time = time.time()
        print(f"Worker {worker_pid}: Completed {os.path.basename(filename)} in {end_time - start_time:.2f}s ({len(data)} records)")
        
        return summary
        
    except Exception as e:
        print(f"Worker {worker_pid}: Error processing {filename}: {e}")
        return pd.DataFrame()

if __name__ == '__main__':
    print("Starting Pandas ETL simulation...")
    
    # Find all log files
    log_files = glob.glob('logs/*.gz')
    print(f"Found {len(log_files)} log files to process")
    
    # Add worker IDs for tracking
    file_tasks = [(f, i) for i, f in enumerate(log_files)]
    
    print("Processing with 2 workers...")
    start_time = time.time()
    
    # Process files with multiprocessing
    with mp.Pool(2) as pool:
        results = pool.map(parse_log_file, file_tasks)
    
    end_time = time.time()
    
    # Combine results
    all_results = pd.concat([r for r in results if not r.empty], ignore_index=True)
    final_summary = all_results.groupby(['level', 'hour'])['count'].sum().reset_index()
    
    print(f"\nETL completed in {end_time - start_time:.2f} seconds")
    print(f"Processed {len(log_files)} files")
    print(f"Final summary shape: {final_summary.shape}")
EOF
	@chmod +x $(DATA_DIR)/$(SCRIPT_NAME)
	@echo "ETL script created: $(DATA_DIR)/$(SCRIPT_NAME)"
	@echo "Log files created:"
	@ls -lh $(DATA_DIR)/logs/

run-test:
	@echo "Running Pandas ETL simulation test..."
	@mkdir -p $(OUTPUT_DIR) $(RESULTS_DIR)
	@echo "Starting analysis and ETL simulation..."
	@cd $(DATA_DIR) && \
	python3 ../../common/analyze.py "python3 $(SCRIPT_NAME)" \
		> ../$(RESULTS_DIR)/etl_log.txt 2>&1
	@mv $(DATA_DIR)/process_analysis.json $(RESULTS_DIR)/ 2>/dev/null || true
	@echo "Test completed. Results saved to $(RESULTS_DIR)/"

analyze:
	@echo "=== Pandas ETL Test Analysis ==="
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