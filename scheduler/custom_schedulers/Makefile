# Standalone Makefile for custom schedulers - out of tree build
# This makefile compiles multiple schedulers independently from the main scx tree

# Scheduler names - just add scheduler names here to build them
SCHED_NAMES = scx_cxl scx_cxl_rl scx_simple_cxl_pmu scx_pmu

# Compiler and tools
CLANG = clang
CC = gcc
BPFTOOL = bpftool

# Include paths - matching the main scx build system
PROJECT_ROOT = $(shell git rev-parse --show-toplevel)
SCX_INCLUDES = \
	-I$(PROJECT_ROOT)/scheduler/scx/scheds/include \
	-I$(PROJECT_ROOT)/scheduler/scx/scheds/include/scx \
	-I$(PROJECT_ROOT)/scheduler/scx/scheds/include/arch/x86 \
	-I$(PROJECT_ROOT)/scheduler/scx/scheds/include/bpf-compat \
	-I$(PROJECT_ROOT)/scheduler/scx/scheds/include/lib

# System includes (like the main build system does)
SYS_INCLUDES = \
	-idirafter /usr/lib/llvm-19/lib/clang/19/include \
	-idirafter /usr/local/include \
	-idirafter /usr/include/x86_64-linux-gnu \
	-idirafter /usr/include

# BPF compilation flags (matching main scx build)
BPF_CFLAGS = -g -O2 -Wall -Wno-compare-distinct-pointer-types \
	-D__TARGET_ARCH_x86 -mcpu=v3 -mlittle-endian \
	$(SYS_INCLUDES) \
	$(SCX_INCLUDES)

# User space compilation flags
USER_CFLAGS = -O2 -g -Wall -Werror \
	$(shell pkg-config --cflags libbpf) \
	-I. \
	$(SCX_INCLUDES)

USER_LDFLAGS = $(shell pkg-config --libs libbpf) -lelf -lz -lzstd

# Generate file lists for all schedulers
BPF_SRCS = $(addsuffix .bpf.c,$(SCHED_NAMES))
USER_SRCS = $(addsuffix .c,$(SCHED_NAMES))
BPF_OBJS = $(addsuffix .bpf.o,$(SCHED_NAMES))
SKEL_HS = $(addsuffix .bpf.skel.h,$(SCHED_NAMES))
TARGETS = $(SCHED_NAMES)

# Default target - build all schedulers
all: $(TARGETS)

# Pattern rules for BPF compilation
%.bpf.o: %.bpf.c
	@echo "Compiling BPF program: $<"
	$(CLANG) $(BPF_CFLAGS) -target bpf -c $< -o $@

# Generate BPF skeleton directly from BPF object
%.bpf.skel.h: %.bpf.o
	@echo "Generating BPF skeleton: $@"
	$(BPFTOOL) gen skeleton $< name $(patsubst %.bpf.skel.h,%,$@) > $@

# Compile user space program
# Each scheduler depends on its skeleton header and C source file
define SCHEDULER_TARGET
$(1): $(1).bpf.skel.h $(1).c
	@echo "Compiling user space program: $(1).c"
	$$(CC) $$(USER_CFLAGS) $(1).c -o $(1) $$(USER_LDFLAGS)
endef

# Generate rules for each scheduler
$(foreach sched,$(SCHED_NAMES),$(eval $(call SCHEDULER_TARGET,$(sched))))

# Install target
install: all
	@echo "Installing to ../sche_bin/"
	mkdir -p ../sche_bin
	cp $(TARGETS) ../sche_bin/

# Clean build artifacts
clean:
	@echo "Cleaning build artifacts..."
	rm -f $(BPF_OBJS) $(SKEL_HS) $(TARGETS)

# Help target
help:
	@echo "Available targets:"
	@echo "  make         - Build all schedulers: $(SCHED_NAMES)"
	@echo "  make clean   - Clean build artifacts"
	@echo "  make install - Install to ../sche_bin/"
	@echo "  make help    - Show this help"
	@echo ""
	@echo "To add more schedulers, edit SCHED_NAMES variable in the Makefile"
	@echo "Current schedulers: $(SCHED_NAMES)"

.PHONY: all clean install help