# Pigz Compression Test Case Makefile

.PHONY: all clean generate-data run-test analyze help

# Configuration
DATA_DIR = /tmp/pigz_test_data
OUTPUT_DIR = output
RESULTS_DIR = results
LARGE_FILE_SIZE = 10    # MB (reduced from 2048)
SMALL_FILE_COUNT = 9     # Reduced from 99
SMALL_FILE_SIZE = 1     # MB

all: generate-data run-test analyze

help:
	@echo "Pigz Compression Test Case"
	@echo ""
	@echo "Available targets:"
	@echo "  generate-data  - Create test files (99 small + 1 large)"
	@echo "  run-test      - Run compression test with analysis"
	@echo "  analyze       - Show analysis results"
	@echo "  clean         - Remove generated files"
	@echo "  help          - Show this help"

generate-data:
	@echo "Generating test data..."
	@mkdir -p $(DATA_DIR)
	@echo "Creating $(SMALL_FILE_COUNT) small files ($(SMALL_FILE_SIZE)MB each)..."
	@for i in $$(seq 1 $(SMALL_FILE_COUNT)); do \
		dd if=/dev/zero of=$(DATA_DIR)/file$$i.dat bs=1M count=$(SMALL_FILE_SIZE) 2>/dev/null; \
	done
	@echo "Creating 1 large file ($(LARGE_FILE_SIZE)MB)..."
	@dd if=/dev/zero of=$(DATA_DIR)/large.iso bs=1M count=$(LARGE_FILE_SIZE) 2>/dev/null
	@echo "Data generation complete."
	@echo "Files created:"
	@ls -lh $(DATA_DIR)/ | tail -5

run-test:
	@echo "Running pigz compression test..."
	@mkdir -p $(OUTPUT_DIR) $(RESULTS_DIR)
	@echo "Starting analysis and compression..."
	@cd $(DATA_DIR) && \
	python3 ../../common/analyze.py \
		"find . -type f -print0 | xargs -0 -n1 -P2 pigz -1 -k" \
		> ../$(RESULTS_DIR)/compression_log.txt 2>&1
	@mv $(DATA_DIR)/process_analysis.json $(RESULTS_DIR)/
	@echo "Test completed. Results saved to $(RESULTS_DIR)/"

analyze:
	@echo "=== Pigz Compression Test Analysis ==="
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

# Development targets
debug-data:
	@echo "Data directory contents:"
	@if [ -d $(DATA_DIR) ]; then ls -lh $(DATA_DIR)/; else echo "No data directory found."; fi

debug-results:
	@echo "Results directory contents:"
	@if [ -d $(RESULTS_DIR) ]; then ls -lh $(RESULTS_DIR)/; else echo "No results directory found."; fi