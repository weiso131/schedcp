# DuckDB Import Test Case Makefile

.PHONY: all clean generate-data run-test analyze help

# Configuration
DATA_DIR = data
OUTPUT_DIR = output
RESULTS_DIR = results
SMALL_FILE_COUNT = 10
LARGE_FILE_ROWS = 1000000

all: generate-data run-test analyze

help:
	@echo "DuckDB Import Test Case"
	@echo ""
	@echo "Available targets:"
	@echo "  generate-data  - Create CSV files for import testing"
	@echo "  run-test      - Run DuckDB import with analysis"
	@echo "  analyze       - Show analysis results"
	@echo "  clean         - Remove generated files"
	@echo "  help          - Show this help"

generate-data:
	@echo "Generating CSV files for DuckDB import testing..."
	@mkdir -p $(DATA_DIR)
	@echo "Creating $(SMALL_FILE_COUNT) small CSV files (10K rows each)..."
	@for i in $$(seq 1 $(SMALL_FILE_COUNT)); do \
		echo "id,value,category" > $(DATA_DIR)/file$$i.csv; \
		seq 1 10000 | awk '{print $$1","rand()","int(rand()*10)}' >> $(DATA_DIR)/file$$i.csv; \
	done
	@echo "Creating 1 large CSV file ($(LARGE_FILE_ROWS) rows)..."
	@echo "id,value,category" > $(DATA_DIR)/file_large.csv
	@seq 1 $(LARGE_FILE_ROWS) | awk '{print $$1","rand()","int(rand()*10)}' >> $(DATA_DIR)/file_large.csv
	@echo "CSV generation complete."
	@echo "Files created:"
	@ls -lh $(DATA_DIR)/

run-test:
	@echo "Running DuckDB import test..."
	@mkdir -p $(OUTPUT_DIR) $(RESULTS_DIR)
	@echo "Starting analysis and import operations..."
	@# Use SQLite as DuckDB alternative for easier testing
	@cd $(DATA_DIR) && \
	python3 ../../common/analyze.py \
		"find . -name '*.csv' -exec sh -c 'sqlite3 ../$(OUTPUT_DIR)/$$(basename \"{}\" .csv).db \".mode csv\" \".import \\\"{}\\\" data\" &' \; ; wait" \
		> ../$(RESULTS_DIR)/import_log.txt 2>&1
	@mv $(DATA_DIR)/process_analysis.json $(RESULTS_DIR)/ 2>/dev/null || true
	@echo "Test completed. Results saved to $(RESULTS_DIR)/"

analyze:
	@echo "=== DuckDB Import Test Analysis ==="
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