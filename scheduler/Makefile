# Makefile for building SCX schedulers

.PHONY: all build build-c build-rust clean help update

# Default target
all: build

# Build all SCX schedulers (C and Rust)
build: build-c build-rust
	@echo "All schedulers built successfully!"
	@echo "Listing all compiled schedulers:"
	@ls -la sche_bin/ 2>/dev/null || echo "No schedulers found in sche_bin/"

# Build C schedulers only
build-c:
	@echo "Building C SCX schedulers..."
	@mkdir -p sche_bin
	@cd scx && \
	echo "Setting up meson build..."; \
	if [ ! -d build ]; then \
		meson setup build --prefix=~ -Denable_rust=false || exit 1; \
	fi; \
	echo "Compiling all C schedulers..."; \
	meson compile -C build 2>&1 | grep -E "Building|Compiling|Linking|FAILED" || true; \
	echo "Copying C schedulers to sche_bin..."; \
	find build/scheds/c -name "scx_*" -type f -executable ! -name "*.o" ! -name "*.a" ! -name "*_test" -exec cp {} ../sche_bin/ \; 2>/dev/null || true
	@echo "C schedulers built successfully!"

# Build Rust schedulers
build-rust:
	@echo "Building Rust SCX schedulers..."
	@mkdir -p sche_bin
	@cd scx && \
	echo "Setting up full meson build for Rust schedulers..."; \
	if [ ! -d build ]; then \
		meson setup build --prefix=~ || exit 1; \
	fi; \
	echo "Building Rust schedulers via meson..."; \
	meson compile -C build rust_all 2>&1 | grep -E "Compiling|Building|Finished|error|FAILED" || true; \
	echo "Copying Rust schedulers from target/release to sche_bin..."; \
	if [ -d target/release ]; then \
		for binary in target/release/scx_*; do \
			if [ -f "$$binary" ] && [ -x "$$binary" ] && ! echo "$$binary" | grep -E "\.d$$|\.rlib$$"; then \
				cp "$$binary" ../sche_bin/ 2>/dev/null && \
				echo "  ✓ $$(basename $$binary) copied to sche_bin"; \
			fi; \
		done; \
	else \
		echo "  No target/release directory found"; \
	fi
	@echo "Rust schedulers build complete!"

# Clean build artifacts
clean:
	@echo "Cleaning SCX build artifacts..."
	@cd scx && rm -rf build target
	@rm -rf sche_bin
	@echo "Clean complete!"

# Update submodule
update:
	@echo "Updating SCX submodule..."
	@git submodule update --init --recursive scx
	@cd scx && git pull origin main
	@echo "Update complete!"

# Help target
help:
	@echo "Available targets:"
	@echo "  make build      - Build all SCX schedulers (C and Rust) and copy to sche_bin/"
	@echo "  make build-c    - Build only C schedulers"
	@echo "  make build-rust - Build only Rust schedulers"
	@echo "  make clean      - Clean all build artifacts"
	@echo "  make update     - Update SCX submodule to latest version"
	@echo "  make help       - Show this help message"