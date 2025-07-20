.PHONY: all build clean update help download-models test-quick test-full test-sharegpt bench-schedulers

BUILD_DIR := build
LLAMA_DIR := llama.cpp
BINARY := $(BUILD_DIR)/llama-cli

all: build

$(BUILD_DIR):
	@mkdir -p $(BUILD_DIR)

build: $(BUILD_DIR)
	@echo "Building llama.cpp..."
	@cd $(LLAMA_DIR) && cmake -B ../$(BUILD_DIR) . \
		-DCMAKE_BUILD_TYPE=Release \
		-DGGML_CUDA=OFF \
		-DGGML_METAL=OFF \
		-DGGML_VULKAN=OFF \
		-DGGML_SYCL=OFF
	@cd $(BUILD_DIR) && make -j$(shell nproc)
	@echo "Build complete. Binary available at: $(BINARY)"

build-cuda: $(BUILD_DIR)
	@echo "Building llama.cpp with CUDA support..."
	@cd $(LLAMA_DIR) && cmake -B ../$(BUILD_DIR) . \
		-DCMAKE_BUILD_TYPE=Release \
		-DGGML_CUDA=ON
	@cd $(BUILD_DIR) && make -j$(shell nproc)
	@echo "Build complete with CUDA. Binary available at: $(BINARY)"

clean:
	@echo "Cleaning build artifacts..."
	@rm -rf $(BUILD_DIR)
	@echo "Clean complete."

update:
	@echo "Updating llama.cpp submodule..."
	@git submodule update --init --recursive
	@cd $(LLAMA_DIR) && git pull origin master
	@echo "Update complete."

download-models: $(MODEL_DIR)
	@echo "Checking models..."
	@if [ ! -f "$(MODEL_1B)" ]; then \
		echo "Downloading TinyLlama 1.1B model..."; \
		python download_test_model.py; \
	fi
	@if [ ! -f "$(MODEL_3B)" ]; then \
		echo "Downloading Llama 3.2 3B model..."; \
		cd $(MODEL_DIR) && huggingface-cli download bartowski/Llama-3.2-3B-Instruct-GGUF --include "Llama-3.2-3B-Instruct-Q4_K_M.gguf" --local-dir .; \
	fi
	@echo "Models ready."

$(MODEL_DIR):
	@mkdir -p $(MODEL_DIR)

$(DATASET_DIR):
	@mkdir -p $(DATASET_DIR)

$(RESULTS_DIR):
	@mkdir -p $(RESULTS_DIR)

test-quick: build download-models $(DATASET_DIR) $(RESULTS_DIR)
	@echo "Running quick ShareGPT test ($(SAMPLES_QUICK) samples, $(CONCURRENT_QUICK) concurrent)..."
	@if [ ! -f "$(SHAREGPT_VICUNA)" ] && [ ! -f "$(SHAREGPT_BENCH)" ]; then \
		echo "Downloading ShareGPT dataset..."; \
		python download_sharegpt.py --dataset vicuna --num-samples 100; \
	fi
	@python sharegpt_llama_server_eval.py \
		--num-samples $(SAMPLES_QUICK) \
		--max-concurrent $(CONCURRENT_QUICK) \
		--server-logs

test-full: build download-models $(DATASET_DIR) $(RESULTS_DIR)
	@echo "Running full ShareGPT test ($(SAMPLES_FULL) samples, $(CONCURRENT_FULL) concurrent)..."
	@if [ ! -f "$(SHAREGPT_VICUNA)" ] && [ ! -f "$(SHAREGPT_BENCH)" ]; then \
		echo "Downloading ShareGPT dataset..."; \
		python download_sharegpt.py --dataset vicuna --num-samples 500; \
	fi
	@python sharegpt_llama_server_eval.py \
		--num-samples $(SAMPLES_FULL) \
		--max-concurrent $(CONCURRENT_FULL) \
		--n-threads 32 \
		--n-parallel 32 \
		--server-logs

test-sharegpt: test-quick
	@echo "ShareGPT quick test completed. Check results directory."

bench-schedulers: build download-models $(DATASET_DIR) $(RESULTS_DIR)
	@echo "Benchmarking all production schedulers..."
	@if [ ! -f "$(SHAREGPT_VICUNA)" ] && [ ! -f "$(SHAREGPT_BENCH)" ]; then \
		echo "Downloading ShareGPT dataset..."; \
		python download_sharegpt.py --dataset vicuna --num-samples 100; \
	fi
	@python sharegpt_llama_server_eval.py \
		--num-samples 20 \
		--max-concurrent 4 \
		--production-only \
		--server-logs

# Custom benchmark with specific parameters
bench-custom: build download-models $(DATASET_DIR) $(RESULTS_DIR)
	@echo "Custom benchmark (set SAMPLES and CONCURRENT env vars)..."
	@python sharegpt_llama_server_eval.py \
		--num-samples $${SAMPLES:-10} \
		--max-concurrent $${CONCURRENT:-2} \
		--server-logs

# Model and dataset paths
MODEL_DIR := models
DATASET_DIR := datasets
RESULTS_DIR := results
LLAMA_SERVER := $(BUILD_DIR)/bin/llama-server
MODEL_3B := $(MODEL_DIR)/Llama-3.2-3B-Instruct-Q4_K_M.gguf
MODEL_1B := $(MODEL_DIR)/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf
SHAREGPT_VICUNA := $(DATASET_DIR)/sharegpt_vicuna.json
SHAREGPT_BENCH := $(DATASET_DIR)/sharegpt_benchmark.json

# Benchmark parameters
SAMPLES_QUICK := 5
SAMPLES_FULL := 1000
CONCURRENT_QUICK := 2
CONCURRENT_FULL := 64

help:
	@echo "Available targets:"
	@echo "  Build targets:"
	@echo "    make build        - Build llama.cpp (CPU only)"
	@echo "    make build-cuda   - Build llama.cpp with CUDA support"
	@echo "  "
	@echo "  Setup targets:"
	@echo "    make download-models - Download all required models"
	@echo "  "
	@echo "  Benchmark targets:"
	@echo "    make test-quick   - Quick test with 5 samples, 2 concurrent"
	@echo "    make test-full    - Full test with 100 samples, 10 concurrent"
	@echo "    make test-sharegpt - Test ShareGPT server benchmark"
	@echo "    make bench-schedulers - Benchmark all production schedulers"
	@echo "  "
	@echo "  Maintenance targets:"
	@echo "    make clean        - Remove build artifacts"
	@echo "    make update       - Update llama.cpp submodule"
	@echo "    make help         - Show this help message"