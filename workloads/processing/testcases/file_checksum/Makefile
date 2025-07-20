# File Checksum Test Case Makefile

.PHONY: all clean generate-data run-test analyze help

# Configuration
DATA_DIR = data
OUTPUT_DIR = output
RESULTS_DIR = results
SMALL_FILE_COUNT = 99
SMALL_FILE_SIZE = 10     # MB
LARGE_FILE_SIZE = 1024   # MB (1GB)

all: generate-data run-test analyze

help:
	@echo "File Checksum Test Case"
	@echo ""
	@echo "Available targets:"
	@echo "  generate-data  - Create files for checksum testing"
	@echo "  run-test      - Run parallel checksum with analysis"
	@echo "  analyze       - Show analysis results"
	@echo "  clean         - Remove generated files"
	@echo "  help          - Show this help"

generate-data:
	@echo "Generating files for checksum testing..."
	@mkdir -p $(DATA_DIR)
	@echo "Creating $(SMALL_FILE_COUNT) small files ($(SMALL_FILE_SIZE)MB each)..."
	@for i in $$(seq 1 $(SMALL_FILE_COUNT)); do \
		dd if=/dev/urandom of=$(DATA_DIR)/file$$i.dat bs=1M count=$(SMALL_FILE_SIZE) 2>/dev/null; \
	done
	@echo "Creating 1 large file ($(LARGE_FILE_SIZE)MB)..."
	@dd if=/dev/urandom of=$(DATA_DIR)/largefile.dat bs=1M count=$(LARGE_FILE_SIZE) 2>/dev/null
	@echo "File generation complete."
	@echo "Files created:"
	@ls -lh $(DATA_DIR)/ | tail -5

run-test:
	@echo "Running parallel file checksum test..."
	@mkdir -p $(OUTPUT_DIR) $(RESULTS_DIR)
	@echo "Starting analysis and checksum calculation..."
	@cd $(DATA_DIR) && \
	python3 ../../common/analyze.py \
		"find . -type f -print0 | xargs -0 -n1 -P2 sha256sum" \
		> ../$(RESULTS_DIR)/checksum_log.txt 2>&1
	@mv $(DATA_DIR)/process_analysis.json $(RESULTS_DIR)/ 2>/dev/null || true
	@echo "Test completed. Results saved to $(RESULTS_DIR)/"

analyze:
	@echo "=== File Checksum Test Analysis ==="
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