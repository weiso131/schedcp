# Sort and Compress Test Case Makefile

.PHONY: all clean generate-data run-test analyze help

# Configuration
DATA_DIR = /tmp/sort_test_data
OUTPUT_DIR = output
RESULTS_DIR = results
SMALL_FILE_COUNT = 5   # Reduced from 10
LARGE_FILE_LINES = 100000  # 100K lines (reduced from 1M)

all: generate-data run-test analyze

help:
	@echo "Sort and Compress Test Case"
	@echo ""
	@echo "Available targets:"
	@echo "  generate-data  - Create data files for sorting"
	@echo "  run-test      - Run parallel sort/compress with analysis"
	@echo "  analyze       - Show analysis results"
	@echo "  clean         - Remove generated files"
	@echo "  help          - Show this help"

generate-data:
	@echo "Generating data files for sort/compress testing..."
	@mkdir -p $(DATA_DIR)
	@echo "Creating $(SMALL_FILE_COUNT) small files (~10K lines each)..."
	@for i in $$(seq 1 $(SMALL_FILE_COUNT)); do \
		seq 1 10000 | shuf > $(DATA_DIR)/part_$$i.tsv; \
	done
	@echo "Creating 1 large file ($(LARGE_FILE_LINES) lines)..."
	@seq 1 $(LARGE_FILE_LINES) | shuf > $(DATA_DIR)/part_large.tsv
	@echo "Data generation complete."
	@echo "Files created:"
	@ls -lh $(DATA_DIR)/

run-test:
	@echo "Running parallel sort and compress test..."
	@mkdir -p $(OUTPUT_DIR) $(RESULTS_DIR)
	@echo "Starting analysis and sort/compress operations..."
	@cd $(DATA_DIR) && \
	python3 ../../common/analyze.py \
		"find . -name '*.tsv' -exec sh -c 'sort \"{}\" | zstd -q -o \"../$(OUTPUT_DIR)/$$(basename \"{}\").zst\" &' \; ; wait" \
		> ../$(RESULTS_DIR)/sort_log.txt 2>&1
	@mv $(DATA_DIR)/process_analysis.json $(RESULTS_DIR)/ 2>/dev/null || true
	@echo "Test completed. Results saved to $(RESULTS_DIR)/"

analyze:
	@echo "=== Sort and Compress Test Analysis ==="
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