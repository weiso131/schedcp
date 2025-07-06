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
	find build/scheds/c -name "scx_*" -type f -executable ! -name "*.o" ! -name "*.a" -exec cp {} ../sche_bin/ \; 2>/dev/null || true
	@echo "C schedulers built successfully!"

# Build Rust schedulers
build-rust:
	@echo "Building Rust SCX schedulers..."
	@mkdir -p sche_bin
	@cd scx/scheds/rust && \
	echo "Found Rust schedulers:"; \
	ls -d scx_*/ 2>/dev/null || echo "No Rust schedulers found"; \
	for sched in scx_*/; do \
		if [ -d "$$sched" ] && [ -f "$$sched/Cargo.toml" ]; then \
			sched_name=$$(basename $$sched); \
			echo "Building $$sched_name..."; \
			cd "$$sched" && \
			cargo build --release 2>&1 | grep -E "Compiling|Building|Finished|error" || true; \
			if [ -f "target/release/$$sched_name" ]; then \
				cp "target/release/$$sched_name" ../../../../sche_bin/ && \
				echo "  ✓ $$sched_name copied to sche_bin"; \
			else \
				echo "  ✗ $$sched_name build failed or binary not found"; \
			fi; \
			cd ..; \
		fi; \
	done
	@echo "Rust schedulers build complete!"

# Clean build artifacts
clean:
	@echo "Cleaning SCX build artifacts..."
	@cd scx && rm -rf build
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