# Standalone Makefile for scx_simple scheduler - out of tree build
# This makefile compiles the scheduler independently from the main scx tree

# Scheduler name
SCHED_NAME = scx_simple

# Source files
BPF_SRC = $(SCHED_NAME).bpf.c
USER_SRC = $(SCHED_NAME).c
BPF_OBJ = $(SCHED_NAME).bpf.o
BPF_L1O = $(SCHED_NAME).bpf.l1o
BPF_L2O = $(SCHED_NAME).bpf.l2o
BPF_L3O = $(SCHED_NAME).bpf.l3o
SKEL_H = $(SCHED_NAME).bpf.skel.h
TARGET = $(SCHED_NAME)

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

# Default target
all: $(TARGET)

# Compile BPF program
$(BPF_OBJ): $(BPF_SRC)
	@echo "Compiling BPF program: $<"
	$(CLANG) $(BPF_CFLAGS) -target bpf -c $< -o $@

# Multi-stage BPF object processing (like main scx build)
$(BPF_L1O): $(BPF_OBJ)
	@echo "Processing BPF object L1: $@"
	$(BPFTOOL) gen object $@ $<

$(BPF_L2O): $(BPF_L1O)
	@echo "Processing BPF object L2: $@"
	$(BPFTOOL) gen object $@ $<

$(BPF_L3O): $(BPF_L2O)
	@echo "Processing BPF object L3: $@"
	$(BPFTOOL) gen object $@ $<
	@echo "Verifying L2 and L3 objects are identical"
	cmp $(BPF_L2O) $(BPF_L3O)

# Generate BPF skeleton from final processed object
$(SKEL_H): $(BPF_L3O)
	@echo "Generating BPF skeleton: $@"
	$(BPFTOOL) gen skeleton $< name $(SCHED_NAME) > $@

# Compile user space program
$(TARGET): $(SKEL_H) $(USER_SRC)
	@echo "Compiling user space program: $(USER_SRC)"
	$(CC) $(USER_CFLAGS) $(USER_SRC) -o $@ $(USER_LDFLAGS)

# Clean build artifacts
clean:
	rm -f $(BPF_OBJ) $(BPF_L1O) $(BPF_L2O) $(BPF_L3O) $(SKEL_H) $(TARGET)

# Help target
help:
	@echo "Available targets:"
	@echo "  make         - Build the scheduler"
	@echo "  make clean   - Clean build artifacts"
	@echo "  make install - Install to ../sche_bin/"
	@echo "  make help    - Show this help"

.PHONY: all clean install help