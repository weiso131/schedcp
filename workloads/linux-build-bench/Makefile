# Makefile for Linux build benchmark
LINUX_DIR := linux
CONFIG := defconfig
JOBS ?= $(shell nproc)

.PHONY: all config clean distclean help

all: config
	@echo "Linux kernel configured. Ready for benchmarking."

config:
	@echo "Configuring Linux kernel with $(CONFIG)..."
	@cd $(LINUX_DIR) && make $(CONFIG)
	@echo "Configuration complete."

# Configure with minimal config for faster builds
tinyconfig:
	@echo "Configuring Linux kernel with tinyconfig..."
	@cd $(LINUX_DIR) && make tinyconfig
	@echo "Tiny configuration complete."

# Configure with allnoconfig for minimal build
allnoconfig:
	@echo "Configuring Linux kernel with allnoconfig..."
	@cd $(LINUX_DIR) && make allnoconfig
	@echo "Minimal configuration complete."

# Clean build artifacts
clean:
	@echo "Cleaning Linux build artifacts..."
	@cd $(LINUX_DIR) && make clean
	@echo "Clean complete."

# Deep clean including config
distclean:
	@echo "Deep cleaning Linux build..."
	@cd $(LINUX_DIR) && make distclean
	@echo "Distclean complete."

# Show kernel version
version:
	@cd $(LINUX_DIR) && make kernelversion

help:
	@echo "Linux Build Benchmark Makefile"
	@echo "Usage: make [target]"
	@echo ""
	@echo "Targets:"
	@echo "  all         - Configure kernel with default config"
	@echo "  config      - Configure kernel with defconfig"
	@echo "  tinyconfig  - Configure with minimal tinyconfig"
	@echo "  allnoconfig - Configure with allnoconfig (smallest)"
	@echo "  clean       - Clean build artifacts"
	@echo "  distclean   - Clean everything including config"
	@echo "  version     - Show kernel version"
	@echo "  help        - Show this help message"
	@echo ""
	@echo "Variables:"
	@echo "  CONFIG=<config> - Kernel config to use (default: defconfig)"
	@echo "  JOBS=<n>        - Number of parallel jobs (default: nproc)"