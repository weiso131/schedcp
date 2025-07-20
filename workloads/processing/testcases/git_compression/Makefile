# Git Compression Test Case Makefile

.PHONY: all clean generate-data run-test analyze help

# Configuration
DATA_DIR = /tmp/git_compression_data
OUTPUT_DIR = /tmp/git_compression_output
RESULTS_DIR = /tmp/git_compression_results
REPO_NAME = test-repo

all: generate-data run-test analyze

help:
	@echo "Git Compression Test Case"
	@echo ""
	@echo "Available targets:"
	@echo "  generate-data  - Create git repo with mixed object sizes"
	@echo "  run-test      - Run git gc with analysis"
	@echo "  analyze       - Show analysis results"
	@echo "  clean         - Remove generated files"
	@echo "  help          - Show this help"

generate-data:
	@echo "Generating git repository with mixed object sizes..."
	@mkdir -p $(DATA_DIR)
	@cd $(DATA_DIR) && \
	git init $(REPO_NAME) && \
	cd $(REPO_NAME) && \
	echo "Creating 9 small commits..." && \
	for i in $$(seq 1 9); do \
		echo "Small change $$i" > file$$i.txt; \
		git add file$$i.txt; \
		git commit -m "Small commit $$i" --quiet; \
	done && \
	echo "Creating 1 large binary blob (10MB)..." && \
	dd if=/dev/zero of=large.bin bs=1M count=10 2>/dev/null && \
	git add large.bin && \
	git commit -m "Add large binary" --quiet
	@echo "Repository created with mixed object sizes."
	@echo "Repository info:"
	@cd $(DATA_DIR)/$(REPO_NAME) && \
	echo "Commits: $$(git rev-list --count HEAD)" && \
	echo "Objects: $$(git count-objects)" && \
	echo "Size: $$(du -sh .git | cut -f1)"

run-test:
	@echo "Running git gc test..."
	@mkdir -p $(OUTPUT_DIR) $(RESULTS_DIR)
	@echo "Configuring git for parallel operation..."
	@cd $(DATA_DIR)/$(REPO_NAME) && \
	git config pack.threads 2 && \
	git config gc.auto 0
	@echo "Starting analysis and git gc..."
	@cd $(DATA_DIR)/$(REPO_NAME) && \
	python3 ../../../common/analyze.py "git gc --aggressive" \
		> ../../$(RESULTS_DIR)/gc_log.txt 2>&1
	@mv $(DATA_DIR)/$(REPO_NAME)/process_analysis.json $(RESULTS_DIR)/ 2>/dev/null || true
	@echo "Test completed. Results saved to $(RESULTS_DIR)/"

analyze:
	@echo "=== Git Compression Test Analysis ==="
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