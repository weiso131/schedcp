# Pytest Test Suite Test Case Makefile

.PHONY: all clean generate-data run-test analyze help

# Configuration
DATA_DIR = /tmp/pytest_suite_data
OUTPUT_DIR = /tmp/pytest_suite_output
RESULTS_DIR = /tmp/pytest_suite_results
TEST_FILE = test_suite.py

all: generate-data run-test analyze

help:
	@echo "Pytest Test Suite Test Case"
	@echo ""
	@echo "Available targets:"
	@echo "  generate-data  - Create test suite with mixed durations"
	@echo "  run-test      - Run pytest with analysis"
	@echo "  analyze       - Show analysis results"
	@echo "  clean         - Remove generated files"
	@echo "  help          - Show this help"

generate-data:
	@echo "Generating pytest test suite..."
	@mkdir -p $(DATA_DIR)
	@cat > $(DATA_DIR)/$(TEST_FILE) << 'EOF'
#!/usr/bin/env python3
"""
Test suite with mixed execution times to demonstrate long-tail scheduling.
99 fast tests + 1 slow integration test.
"""

import time
import pytest

# 9 fast unit tests
def test_fast_1(): time.sleep(0.01); assert True
def test_fast_2(): time.sleep(0.01); assert True
def test_fast_3(): time.sleep(0.01); assert True
def test_fast_4(): time.sleep(0.01); assert True
def test_fast_5(): time.sleep(0.01); assert True
def test_fast_6(): time.sleep(0.01); assert True
def test_fast_7(): time.sleep(0.01); assert True
def test_fast_8(): time.sleep(0.01); assert True
def test_fast_9(): time.sleep(0.01); assert True

# 1 slow integration test (simulates database setup/teardown)
def test_slow_integration():
    """Simulate slow integration test with database setup."""
    print("Starting slow integration test (simulating database setup)...")
    time.sleep(2)  # 2 seconds for faster testing
    assert True
EOF
	@echo "Test suite created: $(DATA_DIR)/$(TEST_FILE)"

run-test:
	@echo "Running pytest test suite..."
	@mkdir -p $(OUTPUT_DIR) $(RESULTS_DIR)
	@echo "Starting analysis and test execution..."
	@cd $(DATA_DIR) && \
	python3 ../../common/analyze.py "pytest -v -x --tb=short $(TEST_FILE)" \
		> ../$(RESULTS_DIR)/pytest_log.txt 2>&1
	@mv $(DATA_DIR)/process_analysis.json $(RESULTS_DIR)/ 2>/dev/null || true
	@echo "Test completed. Results saved to $(RESULTS_DIR)/"

analyze:
	@echo "=== Pytest Test Suite Analysis ==="
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