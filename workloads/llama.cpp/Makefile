.PHONY: all build clean update help

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

help:
	@echo "Available targets:"
	@echo "  make build      - Build llama.cpp (CPU only)"
	@echo "  make build-cuda - Build llama.cpp with CUDA support"
	@echo "  make clean      - Remove build artifacts"
	@echo "  make update     - Update llama.cpp submodule"
	@echo "  make help       - Show this help message"