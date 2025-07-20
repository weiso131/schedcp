# Master Makefile for Long-tail Scheduler Test Cases

.PHONY: all clean help list-tests run-all analyze-all

# Available test cases
TEST_CASES = pigz_compression ffmpeg_transcode pytest_suite git_compression rocksdb_compaction file_checksum spark_shuffle sort_compress dask_groupby duckdb_import pandas_etl flink_join

all: help

help:
	@echo "Long-tail Scheduler Test Cases"
	@echo "=============================="
	@echo ""
	@echo "Available commands:"
	@echo "  list-tests    - List all available test cases"
	@echo "  run-all       - Run all test cases"
	@echo "  analyze-all   - Show analysis for all completed tests"
	@echo "  clean         - Clean all test cases"
	@echo "  help          - Show this help"
	@echo ""
	@echo "Individual test case commands:"
	@echo "  run-<testcase>     - Run specific test case"
	@echo "  analyze-<testcase> - Analyze specific test case"
	@echo "  clean-<testcase>   - Clean specific test case"
	@echo ""
	@echo "Available test cases: $(TEST_CASES)"
	@echo ""
	@echo "Example usage:"
	@echo "  make run-pigz_compression     # Run pigz test"
	@echo "  make analyze-ffmpeg_transcode # Analyze ffmpeg results"

list-tests:
	@echo "Available test cases:"
	@for test in $(TEST_CASES); do \
		if [ -d $$test ]; then \
			echo "  $$test - "; \
			if [ -f $$test/README.md ]; then \
				head -1 $$test/README.md | sed 's/^# //'; \
			else \
				echo "Long-tail workload test"; \
			fi; \
		fi; \
	done

run-all:
	@echo "Running all test cases..."
	@for test in $(TEST_CASES); do \
		if [ -d $$test ]; then \
			echo ""; \
			echo "=== Running $$test ==="; \
			cd $$test && make run-test && cd ..; \
		fi; \
	done

analyze-all:
	@echo "Analyzing all test cases..."
	@for test in $(TEST_CASES); do \
		if [ -d $$test ] && [ -f $$test/results/process_analysis.json ]; then \
			echo ""; \
			echo "=== Analysis for $$test ==="; \
			cd $$test && make analyze && cd ..; \
		fi; \
	done

clean:
	@echo "Cleaning all test cases..."
	@for test in $(TEST_CASES); do \
		if [ -d $$test ]; then \
			echo "Cleaning $$test..."; \
			cd $$test && make clean && cd ..; \
		fi; \
	done

# Individual test case targets
$(addprefix run-,$(TEST_CASES)):
	@test_name=$(patsubst run-%,%,$@); \
	if [ -d $$test_name ]; then \
		echo "Running $$test_name test case..."; \
		cd $$test_name && make run-test; \
	else \
		echo "Test case $$test_name not found."; \
	fi

$(addprefix analyze-,$(TEST_CASES)):
	@test_name=$(patsubst analyze-%,%,$@); \
	if [ -d $$test_name ]; then \
		echo "Analyzing $$test_name test case..."; \
		cd $$test_name && make analyze; \
	else \
		echo "Test case $$test_name not found."; \
	fi

$(addprefix clean-,$(TEST_CASES)):
	@test_name=$(patsubst clean-%,%,$@); \
	if [ -d $$test_name ]; then \
		echo "Cleaning $$test_name test case..."; \
		cd $$test_name && make clean; \
	else \
		echo "Test case $$test_name not found."; \
	fi