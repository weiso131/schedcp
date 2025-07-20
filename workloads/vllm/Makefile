# Makefile for building and managing vLLM

# Variables
PYTHON ?= python3
PIP ?= pip3
VLLM_DIR = vllm
CUDA_HOME ?= /usr/local/cuda
BUILD_DIR = build
VENV_DIR = venv

# Installation variables
USE_SYSTEM_INSTALL ?= 0
INSTALL_TYPE ?= pip  # Options: pip, source

# CUDA architecture flags
CUDA_ARCH ?= 75,80,86,89,90  # Default architectures

# Number of build jobs
MAX_JOBS ?= $(shell nproc)

# Colors for output
GREEN = \033[0;32m
RED = \033[0;31m
NC = \033[0m # No Color

.PHONY: all build install install-pip install-source test benchmark benchmark-latency benchmark-throughput clean help setup-venv

all: install

# Check prerequisites
check-prereqs:
	@echo "$(GREEN)Checking prerequisites...$(NC)"
	@which python3 >/dev/null 2>&1 || { echo "$(RED)Python3 is required but not installed.$(NC)" >&2; exit 1; }
	@which pip3 >/dev/null 2>&1 || { echo "$(RED)pip3 is required but not installed.$(NC)" >&2; exit 1; }
	@if [ -z "$$(which nvcc 2>/dev/null)" ]; then \
		echo "$(RED)Warning: nvcc not found. CUDA installation may be required for GPU support.$(NC)"; \
	fi

# Setup virtual environment
setup-venv:
	@if [ ! -d "$(VENV_DIR)" ]; then \
		echo "$(GREEN)Creating virtual environment...$(NC)"; \
		$(PYTHON) -m venv $(VENV_DIR); \
		. $(VENV_DIR)/bin/activate && pip install --upgrade pip setuptools wheel; \
	fi
	@echo "$(GREEN)Virtual environment ready. Activate with: source $(VENV_DIR)/bin/activate$(NC)"

# Install vLLM using pip (recommended for most users)
install-pip: check-prereqs
	@echo "$(GREEN)Installing vLLM via pip...$(NC)"
	$(PIP) install vllm
	@echo "$(GREEN)vLLM installed successfully via pip!$(NC)"

# Build and install vLLM from source
install-source: check-prereqs
	@echo "$(GREEN)Building vLLM from source...$(NC)"
	@cd $(VLLM_DIR) && \
	export MAX_JOBS=$(MAX_JOBS) && \
	export TORCH_CUDA_ARCH_LIST="$(CUDA_ARCH)" && \
	$(PIP) install -e . --verbose
	@echo "$(GREEN)vLLM built and installed from source!$(NC)"

# Build only (without installing)
build: check-prereqs
	@echo "$(GREEN)Building vLLM...$(NC)"
	@cd $(VLLM_DIR) && \
	export MAX_JOBS=$(MAX_JOBS) && \
	export TORCH_CUDA_ARCH_LIST="$(CUDA_ARCH)" && \
	$(PYTHON) setup.py build_ext --inplace
	@echo "$(GREEN)Build complete!$(NC)"

# Install dependencies only
install-deps: check-prereqs
	@echo "$(GREEN)Installing dependencies...$(NC)"
	$(PIP) install cmake>=3.26 ninja packaging setuptools>=61 setuptools-scm>=8 wheel jinja2
	$(PIP) install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
	@echo "$(GREEN)Dependencies installed!$(NC)"

# Default install based on INSTALL_TYPE
install: 
ifeq ($(INSTALL_TYPE),source)
	@$(MAKE) install-source
else
	@$(MAKE) install-pip
endif

# Run basic tests
test:
	@echo "$(GREEN)Running basic vLLM tests...$(NC)"
	@cd $(VLLM_DIR) && $(PYTHON) -c "import vllm; print('vLLM version:', vllm.__version__)"
	@echo "$(GREEN)Basic tests passed!$(NC)"

# Benchmark commands
benchmark-latency:
	@echo "$(GREEN)Running latency benchmark...$(NC)"
	@echo "Usage: make benchmark-latency MODEL=<model-name> BATCH_SIZE=<size> INPUT_LEN=<len> OUTPUT_LEN=<len>"
	@echo "Example: make benchmark-latency MODEL=meta-llama/Llama-2-7b-hf BATCH_SIZE=1 INPUT_LEN=32 OUTPUT_LEN=128"
	@if [ -n "$(MODEL)" ]; then \
		cd $(VLLM_DIR) && $(PYTHON) benchmarks/benchmark_latency.py \
			--model $(MODEL) \
			--batch-size $${BATCH_SIZE:-1} \
			--input-len $${INPUT_LEN:-32} \
			--output-len $${OUTPUT_LEN:-128} \
			--num-iters-warmup $${WARMUP:-10} \
			--num-iters $${ITERS:-30}; \
	else \
		echo "$(RED)Please specify MODEL parameter$(NC)"; \
		exit 1; \
	fi

benchmark-throughput:
	@echo "$(GREEN)Running throughput benchmark...$(NC)"
	@echo "Usage: make benchmark-throughput MODEL=<model-name> NUM_PROMPTS=<num> DATASET=<dataset>"
	@echo "Example: make benchmark-throughput MODEL=meta-llama/Llama-2-7b-hf NUM_PROMPTS=1000 DATASET=sharegpt"
	@if [ -n "$(MODEL)" ]; then \
		cd $(VLLM_DIR) && $(PYTHON) benchmarks/benchmark_throughput.py \
			--model $(MODEL) \
			--num-prompts $${NUM_PROMPTS:-1000} \
			--dataset-name $${DATASET:-sharegpt} \
			$${DATASET_PATH:+--dataset-path $$DATASET_PATH} \
			$${INPUT_LEN:+--input-len $$INPUT_LEN} \
			$${OUTPUT_LEN:+--output-len $$OUTPUT_LEN}; \
	else \
		echo "$(RED)Please specify MODEL parameter$(NC)"; \
		exit 1; \
	fi

benchmark-serving:
	@echo "$(GREEN)Running serving benchmark...$(NC)"
	@echo "Usage: make benchmark-serving MODEL=<model-name> PORT=<port> NUM_PROMPTS=<num>"
	@echo "Example: make benchmark-serving MODEL=meta-llama/Llama-2-7b-hf PORT=8000 NUM_PROMPTS=100"
	@echo "Note: Make sure vLLM server is running first with: vllm serve <model-name>"
	@if [ -n "$(MODEL)" ]; then \
		cd $(VLLM_DIR) && $(PYTHON) benchmarks/benchmark_serving.py \
			--backend vllm \
			--model $(MODEL) \
			--endpoint /v1/completions \
			--port $${PORT:-8000} \
			--num-prompts $${NUM_PROMPTS:-100} \
			--dataset-name $${DATASET:-sharegpt} \
			$${DATASET_PATH:+--dataset-path $$DATASET_PATH}; \
	else \
		echo "$(RED)Please specify MODEL parameter$(NC)"; \
		exit 1; \
	fi

# Download common benchmark datasets
download-datasets:
	@echo "$(GREEN)Downloading benchmark datasets...$(NC)"
	@mkdir -p datasets
	@if [ ! -f "datasets/ShareGPT_V3_unfiltered_cleaned_split.json" ]; then \
		echo "Downloading ShareGPT dataset..."; \
		wget -P datasets https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json; \
	fi
	@if [ ! -f "datasets/sonnet.txt" ]; then \
		echo "Copying Sonnet dataset..."; \
		cp $(VLLM_DIR)/benchmarks/sonnet.txt datasets/; \
	fi
	@echo "$(GREEN)Datasets ready in ./datasets/$(NC)"

# Run a quick benchmark suite
benchmark-quick: install
	@echo "$(GREEN)Running quick benchmark suite...$(NC)"
	@$(MAKE) benchmark-latency MODEL=facebook/opt-125m BATCH_SIZE=1 INPUT_LEN=32 OUTPUT_LEN=32 WARMUP=5 ITERS=10
	@$(MAKE) benchmark-throughput MODEL=facebook/opt-125m NUM_PROMPTS=100 DATASET=random INPUT_LEN=32 OUTPUT_LEN=32

# Clean build artifacts
clean:
	@echo "$(GREEN)Cleaning build artifacts...$(NC)"
	@cd $(VLLM_DIR) && \
	rm -rf build dist *.egg-info _C*.so vllm/*.so vllm/**/*.so .eggs
	@rm -rf $(BUILD_DIR)
	@echo "$(GREEN)Clean complete!$(NC)"

# Clean everything including virtual environment
clean-all: clean
	@echo "$(GREEN)Removing virtual environment...$(NC)"
	@rm -rf $(VENV_DIR)
	@echo "$(GREEN)Full clean complete!$(NC)"

# Help command
help:
	@echo "vLLM Build and Benchmark Makefile"
	@echo ""
	@echo "Installation targets:"
	@echo "  make install              - Install vLLM (default: pip, set INSTALL_TYPE=source for source build)"
	@echo "  make install-pip          - Install vLLM using pip (recommended)"
	@echo "  make install-source       - Build and install vLLM from source"
	@echo "  make install-deps         - Install dependencies only"
	@echo "  make build                - Build vLLM without installing"
	@echo "  make setup-venv           - Create a virtual environment"
	@echo ""
	@echo "Benchmark targets:"
	@echo "  make benchmark-latency    - Run latency benchmark"
	@echo "  make benchmark-throughput - Run throughput benchmark"
	@echo "  make benchmark-serving    - Run serving benchmark (requires running server)"
	@echo "  make benchmark-quick      - Run a quick benchmark suite"
	@echo "  make download-datasets    - Download common benchmark datasets"
	@echo ""
	@echo "Other targets:"
	@echo "  make test                 - Run basic tests"
	@echo "  make clean                - Clean build artifacts"
	@echo "  make clean-all            - Clean everything including virtual environment"
	@echo "  make help                 - Show this help message"
	@echo ""
	@echo "Environment variables:"
	@echo "  PYTHON                    - Python executable (default: python3)"
	@echo "  PIP                       - Pip executable (default: pip3)"
	@echo "  MAX_JOBS                  - Number of parallel build jobs (default: nproc)"
	@echo "  CUDA_ARCH                 - CUDA architectures to build for (default: 75,80,86,89,90)"
	@echo "  INSTALL_TYPE              - Installation type: pip or source (default: pip)"
	@echo ""
	@echo "Benchmark parameters:"
	@echo "  MODEL                     - Model name/path (required for benchmarks)"
	@echo "  BATCH_SIZE                - Batch size for latency benchmark"
	@echo "  INPUT_LEN                 - Input length"
	@echo "  OUTPUT_LEN                - Output length"
	@echo "  NUM_PROMPTS               - Number of prompts for throughput benchmark"
	@echo "  DATASET                   - Dataset name (sharegpt, sonnet, random, etc.)"
	@echo "  DATASET_PATH              - Path to dataset file"
	@echo "  PORT                      - Server port for serving benchmark"
	@echo ""
	@echo "Examples:"
	@echo "  make install"
	@echo "  make install-source MAX_JOBS=8"
	@echo "  make benchmark-latency MODEL=meta-llama/Llama-2-7b-hf"
	@echo "  make benchmark-throughput MODEL=facebook/opt-125m NUM_PROMPTS=1000"