# Makefile for building SCX schedulers

.PHONY: all build build-c build-rust build-tools clean help update doc

# Default target
all: build build-tools

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
	rm -rf build; \
	meson setup build --prefix=~ || exit 1; \
	echo "Building all C schedulers..."; \
	meson compile -C build 2>/dev/null || meson compile -C build || exit 1; \
	echo "Copying C schedulers to sche_bin..."; \
	find build -name "scx_*" -type f -executable ! -name "*.o" ! -name "*.a" ! -name "*_test" ! -name "*.d" -exec cp {} ../sche_bin/ \; 2>/dev/null || true
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
	meson compile -C build || exit 1; \
	echo "Copying Rust schedulers from build to sche_bin..."; \
	find build -name "scx_*" -type f -executable ! -name "*.o" ! -name "*.a" ! -name "*_test" ! -name "*.d" -exec cp {} ../sche_bin/ \; 2>/dev/null || true; \
	echo "Also copying from target/release if available..."; \
	if [ -d target/release ]; then \
		for binary in target/release/scx_*; do \
			if [ -f "$$binary" ] && [ -x "$$binary" ] && ! echo "$$binary" | grep -E "\.d$$|\.rlib$$"; then \
				cp "$$binary" ../sche_bin/ 2>/dev/null && \
				echo "  ✓ $$(basename $$binary) copied to sche_bin"; \
			fi; \
		done; \
	fi
	@echo "Rust schedulers build complete!"

# Build tools (scx_loader, scxctl, scxtop)
build-tools:
	@echo "Building SCX tools..."
	@mkdir -p tools
	@cd scx && \
	echo "Building tools via meson..."; \
	if [ ! -d build ]; then \
		meson setup build --prefix=~ || exit 1; \
	fi; \
	echo "Building scx_loader..."; \
	cd rust/scx_loader && cargo build --release 2>&1 | grep -E "Compiling|Building|Finished|error" || true; \
	cd ../.. && \
	echo "Building scxctl..."; \
	cd tools/scxctl && cargo build --release 2>&1 | grep -E "Compiling|Building|Finished|error" || true; \
	cd ../.. && \
	echo "Building scxtop..."; \
	cd tools/scxtop && cargo build --release 2>&1 | grep -E "Compiling|Building|Finished|error" || true; \
	cd ../.. && \
	echo "Copying tools to tools/ directory..."; \
	if [ -d target/release ]; then \
		for tool in scx_loader scxctl scxtop; do \
			if [ -f "target/release/$$tool" ]; then \
				cp "target/release/$$tool" ../tools/ && \
				echo "  ✓ $$tool copied to tools/"; \
			else \
				echo "  ✗ $$tool not found"; \
			fi; \
		done; \
	fi; \
	echo "Looking for other tools in meson build..."; \
	if [ -d build ]; then \
		find build -name "dsq_lat" -o -name "runq_lat" -o -name "veristat*" -type f -executable | while read tool; do \
			cp "$$tool" ../tools/ 2>/dev/null && \
			echo "  ✓ $$(basename $$tool) copied to tools/"; \
		done; \
	fi
	@echo "Tools build complete!"
	@echo "Available tools:"
	@ls -la tools/ 2>/dev/null || echo "No tools found"

# Clean build artifacts
clean:
	@echo "Cleaning SCX build artifacts..."
	@cd scx && rm -rf build target
	@cd scx/rust/scx_loader && rm -rf target 2>/dev/null || true
	@cd scx/tools/scxctl && rm -rf target 2>/dev/null || true
	@cd scx/tools/scxtop && rm -rf target 2>/dev/null || true
	@rm -rf sche_bin tools
	@echo "Clean complete!"

# Update submodule
update:
	@echo "Updating SCX submodule..."
	@git submodule update --init --recursive scx
	@cd scx && git pull origin main
	@echo "Update complete!"

# Generate documentation for all schedulers
doc:
	@echo "Generating scheduler documentation..."
	@mkdir -p sche_description
	@echo "Extracting C scheduler descriptions..."
	@cd scx && \
	for sched in simple central flatcg nest pair qmap userland prev sdt; do \
		echo "Processing scx_$$sched..."; \
		if [ -f "scheds/c/scx_$$sched.bpf.c" ]; then \
			echo "# scx_$$sched" > ../sche_description/scx_$$sched.md; \
			echo "" >> ../sche_description/scx_$$sched.md; \
			awk '/^\/\*/{p=1} p{print} /\*\//{if(p) exit}' scheds/c/scx_$$sched.bpf.c | \
			sed 's/^\/\*\s*//' | sed 's/^\s*\*\s*//' | sed 's/\s*\*\/$//' | \
			sed '/^$$/{N; /^\n$$/d}' >> ../sche_description/scx_$$sched.md; \
			grep -A50 "^## scx_$$sched" scheds/c/README.md 2>/dev/null | \
			awk '/^## scx_/{next} /^## scx_/{exit} {print}' >> ../sche_description/scx_$$sched.md 2>/dev/null || true; \
		fi; \
	done
	@echo "Copying Rust scheduler READMEs..."
	@cd scx/scheds/rust && \
	for sched_dir in scx_*/; do \
		sched=$$(basename $$sched_dir); \
		if [ -f "$$sched_dir/README.md" ] && [ -s "$$sched_dir/README.md" ]; then \
			cp "$$sched_dir/README.md" ../../../sche_description/$$sched.md; \
			echo "  ✓ Copied $$sched.md"; \
		else \
			echo "  ✗ No README for $$sched, creating from source..."; \
			echo "# $$sched" > ../../../sche_description/$$sched.md; \
			echo "" >> ../../../sche_description/$$sched.md; \
			if [ -f "$$sched_dir/src/bpf/main.bpf.c" ]; then \
				awk '/^\/\*/{p=1} p{print} /\*\//{if(p) exit}' "$$sched_dir/src/bpf/main.bpf.c" | \
				sed 's/^\/\*\s*//' | sed 's/^\s*\*\s*//' | sed 's/\s*\*\/$//' | \
				sed '/^$$/{N; /^\n$$/d}' >> ../../../sche_description/$$sched.md 2>/dev/null || \
				echo "No documentation found in source." >> ../../../sche_description/$$sched.md; \
			else \
				echo "No documentation found." >> ../../../sche_description/$$sched.md; \
			fi; \
		fi; \
	done
	@echo "Documentation generation complete!"
	@echo "Generated documentation files:"
	@ls -1 sche_description/*.md 2>/dev/null | wc -l
	@echo "files in sche_description/"

# Help target
help:
	@echo "Available targets:"
	@echo "  make            - Build all schedulers and tools"
	@echo "  make build      - Build all SCX schedulers (C and Rust) and copy to sche_bin/"
	@echo "  make build-c    - Build only C schedulers"
	@echo "  make build-rust - Build only Rust schedulers"
	@echo "  make build-tools- Build tools (scx_loader, scxctl, scxtop) and copy to tools/"
	@echo "  make doc        - Generate documentation for all schedulers in sche_description/"
	@echo "  make clean      - Clean all build artifacts"
	@echo "  make update     - Update SCX submodule to latest version"
	@echo "  make help       - Show this help message"