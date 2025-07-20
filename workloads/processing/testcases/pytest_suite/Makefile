# Pytest Test Suite Test Case Makefile

.PHONY: all clean generate-data run-test analyze help

# Configuration
DATA_DIR = data
OUTPUT_DIR = output
RESULTS_DIR = results
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

# 99 fast unit tests
def test_fast_1(): time.sleep(0.01); assert True
def test_fast_2(): time.sleep(0.01); assert True
def test_fast_3(): time.sleep(0.01); assert True
def test_fast_4(): time.sleep(0.01); assert True
def test_fast_5(): time.sleep(0.01); assert True
def test_fast_6(): time.sleep(0.01); assert True
def test_fast_7(): time.sleep(0.01); assert True
def test_fast_8(): time.sleep(0.01); assert True
def test_fast_9(): time.sleep(0.01); assert True
def test_fast_10(): time.sleep(0.01); assert True
def test_fast_11(): time.sleep(0.01); assert True
def test_fast_12(): time.sleep(0.01); assert True
def test_fast_13(): time.sleep(0.01); assert True
def test_fast_14(): time.sleep(0.01); assert True
def test_fast_15(): time.sleep(0.01); assert True
def test_fast_16(): time.sleep(0.01); assert True
def test_fast_17(): time.sleep(0.01); assert True
def test_fast_18(): time.sleep(0.01); assert True
def test_fast_19(): time.sleep(0.01); assert True
def test_fast_20(): time.sleep(0.01); assert True
def test_fast_21(): time.sleep(0.01); assert True
def test_fast_22(): time.sleep(0.01); assert True
def test_fast_23(): time.sleep(0.01); assert True
def test_fast_24(): time.sleep(0.01); assert True
def test_fast_25(): time.sleep(0.01); assert True
def test_fast_26(): time.sleep(0.01); assert True
def test_fast_27(): time.sleep(0.01); assert True
def test_fast_28(): time.sleep(0.01); assert True
def test_fast_29(): time.sleep(0.01); assert True
def test_fast_30(): time.sleep(0.01); assert True
def test_fast_31(): time.sleep(0.01); assert True
def test_fast_32(): time.sleep(0.01); assert True
def test_fast_33(): time.sleep(0.01); assert True
def test_fast_34(): time.sleep(0.01); assert True
def test_fast_35(): time.sleep(0.01); assert True
def test_fast_36(): time.sleep(0.01); assert True
def test_fast_37(): time.sleep(0.01); assert True
def test_fast_38(): time.sleep(0.01); assert True
def test_fast_39(): time.sleep(0.01); assert True
def test_fast_40(): time.sleep(0.01); assert True
def test_fast_41(): time.sleep(0.01); assert True
def test_fast_42(): time.sleep(0.01); assert True
def test_fast_43(): time.sleep(0.01); assert True
def test_fast_44(): time.sleep(0.01); assert True
def test_fast_45(): time.sleep(0.01); assert True
def test_fast_46(): time.sleep(0.01); assert True
def test_fast_47(): time.sleep(0.01); assert True
def test_fast_48(): time.sleep(0.01); assert True
def test_fast_49(): time.sleep(0.01); assert True
def test_fast_50(): time.sleep(0.01); assert True
def test_fast_51(): time.sleep(0.01); assert True
def test_fast_52(): time.sleep(0.01); assert True
def test_fast_53(): time.sleep(0.01); assert True
def test_fast_54(): time.sleep(0.01); assert True
def test_fast_55(): time.sleep(0.01); assert True
def test_fast_56(): time.sleep(0.01); assert True
def test_fast_57(): time.sleep(0.01); assert True
def test_fast_58(): time.sleep(0.01); assert True
def test_fast_59(): time.sleep(0.01); assert True
def test_fast_60(): time.sleep(0.01); assert True
def test_fast_61(): time.sleep(0.01); assert True
def test_fast_62(): time.sleep(0.01); assert True
def test_fast_63(): time.sleep(0.01); assert True
def test_fast_64(): time.sleep(0.01); assert True
def test_fast_65(): time.sleep(0.01); assert True
def test_fast_66(): time.sleep(0.01); assert True
def test_fast_67(): time.sleep(0.01); assert True
def test_fast_68(): time.sleep(0.01); assert True
def test_fast_69(): time.sleep(0.01); assert True
def test_fast_70(): time.sleep(0.01); assert True
def test_fast_71(): time.sleep(0.01); assert True
def test_fast_72(): time.sleep(0.01); assert True
def test_fast_73(): time.sleep(0.01); assert True
def test_fast_74(): time.sleep(0.01); assert True
def test_fast_75(): time.sleep(0.01); assert True
def test_fast_76(): time.sleep(0.01); assert True
def test_fast_77(): time.sleep(0.01); assert True
def test_fast_78(): time.sleep(0.01); assert True
def test_fast_79(): time.sleep(0.01); assert True
def test_fast_80(): time.sleep(0.01); assert True
def test_fast_81(): time.sleep(0.01); assert True
def test_fast_82(): time.sleep(0.01); assert True
def test_fast_83(): time.sleep(0.01); assert True
def test_fast_84(): time.sleep(0.01); assert True
def test_fast_85(): time.sleep(0.01); assert True
def test_fast_86(): time.sleep(0.01); assert True
def test_fast_87(): time.sleep(0.01); assert True
def test_fast_88(): time.sleep(0.01); assert True
def test_fast_89(): time.sleep(0.01); assert True
def test_fast_90(): time.sleep(0.01); assert True
def test_fast_91(): time.sleep(0.01); assert True
def test_fast_92(): time.sleep(0.01); assert True
def test_fast_93(): time.sleep(0.01); assert True
def test_fast_94(): time.sleep(0.01); assert True
def test_fast_95(): time.sleep(0.01); assert True
def test_fast_96(): time.sleep(0.01); assert True
def test_fast_97(): time.sleep(0.01); assert True
def test_fast_98(): time.sleep(0.01); assert True
def test_fast_99(): time.sleep(0.01); assert True

# 1 slow integration test (simulates database setup/teardown)
def test_slow_integration():
    """Simulate slow integration test with database setup."""
    print("Starting slow integration test (simulating database setup)...")
    time.sleep(10)  # 10 seconds for faster testing
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