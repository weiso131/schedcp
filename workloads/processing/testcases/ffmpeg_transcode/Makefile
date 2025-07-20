# FFmpeg Transcode Test Case Makefile

.PHONY: all clean generate-data run-test analyze help

# Configuration
DATA_DIR = /tmp/ffmpeg_test_data
OUTPUT_DIR = output
RESULTS_DIR = results
SHORT_VIDEO_COUNT = 5   # Reduced from 10
LONG_VIDEO_DURATION = 10  # seconds (reduced from 30)

all: generate-data run-test analyze

help:
	@echo "FFmpeg Transcode Test Case"
	@echo ""
	@echo "Available targets:"
	@echo "  generate-data  - Create test videos (10 short + 1 long)"
	@echo "  run-test      - Run transcode test with analysis"
	@echo "  analyze       - Show analysis results"
	@echo "  clean         - Remove generated files"
	@echo "  help          - Show this help"

generate-data:
	@echo "Generating test video data..."
	@mkdir -p $(DATA_DIR) $(OUTPUT_DIR)
	@echo "Creating $(SHORT_VIDEO_COUNT) short video clips (1 second each)..."
	@for i in $$(seq 1 $(SHORT_VIDEO_COUNT)); do \
		ffmpeg -f lavfi -i testsrc=duration=1:size=320x240:rate=30 \
			-y $(DATA_DIR)/clip$$i.mp4 -loglevel quiet; \
	done
	@echo "Creating 1 long video clip ($(LONG_VIDEO_DURATION) seconds)..."
	@ffmpeg -f lavfi -i testsrc=duration=$(LONG_VIDEO_DURATION):size=1920x1080:rate=30 \
		-y $(DATA_DIR)/long_clip.mp4 -loglevel quiet
	@echo "Video generation complete."
	@echo "Files created:"
	@ls -lh $(DATA_DIR)/

run-test:
	@echo "Running FFmpeg transcode test..."
	@mkdir -p $(OUTPUT_DIR) $(RESULTS_DIR)
	@echo "Starting analysis and transcoding..."
	@python3 ../common/analyze.py \
		"find $(DATA_DIR) -name '*.mp4' -exec sh -c 'ffmpeg -loglevel quiet -i \"{}\" -vf scale=640:-1 -c:v libx264 -preset veryfast \"$(OUTPUT_DIR)/$$(basename \"{}\")\" &' \; ; wait" \
		> $(RESULTS_DIR)/transcode_log.txt 2>&1
	@mv process_analysis.json $(RESULTS_DIR)/ 2>/dev/null || true
	@echo "Test completed. Results saved to $(RESULTS_DIR)/"

analyze:
	@echo "=== FFmpeg Transcode Test Analysis ==="
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