# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AI-OS is a Linux scheduler optimization project that uses the sched-ext (scx) framework to implement and test various BPF-based kernel schedulers. The project includes multiple scheduler implementations in both C and Rust, along with benchmarking tools and workloads.

## Build Commands

### Building Schedulers
```bash
# Build all schedulers (C and Rust) - outputs to sche_bin/
make

# Build only C schedulers
make build-c

# Build only Rust schedulers
make build-rust

# Build tools (scx_loader, scxctl, scxtop) - outputs to tools/
make build-tools

# Generate scheduler documentation
make doc

# Clean build artifacts
make clean

# Update scx submodule
make update
```

### Meson Build System (Advanced)
```bash
cd scheduler/scx
meson setup build --prefix=~
meson compile -C build
meson test -v -C build
```

## Testing

### Unit Tests
```bash
cd scheduler/scx
meson setup build
meson compile -C build
meson test -v -C build
```

### Running Benchmarks
```bash
# Schbench workload
python workloads/basic/scheduler_test/schbench_bench_start.py

# LLAMA.cpp benchmark
python workloads/llama.cpp/llamacpp_bench_start.py

# CXL micro benchmark
python workloads/cxl-micro/cxl_micro_bench_start.py
```

## Architecture

### Directory Structure
- **scheduler/**: Main scheduler code and build system
  - **scx/**: Submodule containing the sched-ext framework
  - **sche_bin/**: Compiled scheduler binaries (after build)
  - **sche_description/**: Auto-generated scheduler documentation
  - **scheduler_runner.py**: Python module for running and testing schedulers
  - **schedulers.json**: Scheduler metadata and configuration
  
- **workloads/**: Benchmark and test workloads
  - **basic/**: Basic scheduler tests (schbench)
  - **llama.cpp/**: LLM inference benchmarks
  - **cxl-micro/**: Memory system benchmarks

### Key Components

1. **Scheduler Implementations**
   - C schedulers: Located in `scheduler/scx/scheds/c/`
   - Rust schedulers: Located in `scheduler/scx/scheds/rust/`
   - Each scheduler consists of BPF code (.bpf.c) and userspace component

2. **Tools**
   - **scxtop**: Real-time scheduler statistics viewer
   - **scxctl**: Command-line scheduler control
   - **scx_loader**: Scheduler loading daemon

3. **SchedulerRunner Module** (`scheduler/scheduler_runner.py`)
   - Unified interface for managing schedulers
   - Provides methods to start/stop schedulers
   - Run benchmarks with different schedulers
   - Parse scheduler configuration from schedulers.json

### Scheduler Types

The project includes various scheduler types optimized for different workloads:
- **scx_rusty**: General-purpose multi-domain scheduler (production-ready)
- **scx_lavd**: Latency-aware scheduler for interactive workloads
- **scx_layered**: Highly configurable multi-layer scheduler
- **scx_bpfland**: Interactive workload prioritization
- **scx_flash**: EDF-based scheduler for predictable latency
- **scx_flatcg**: Cgroup-aware scheduler for containers

## Development Tips

### Adding New Features
1. Check existing scheduler implementations for patterns
2. Follow the BPF coding conventions in the scx submodule
3. Use the SchedulerRunner class for testing new schedulers
4. Update schedulers.json with new scheduler metadata

### Common Tasks
```bash
# List available schedulers
python scheduler/scheduler_runner.py

# Run a specific scheduler
./scheduler/sche_bin/scx_rusty --slice-us 20000

# Monitor scheduler performance
./scheduler/tools/scxtop

# Control schedulers
./scheduler/tools/scxctl status
```

### Debugging
- Use `bpftool` to inspect loaded BPF programs
- Check kernel logs with `dmesg` for scheduler messages
- Use `scxtop` for real-time performance monitoring
- Enable stats collection in schedulers for detailed metrics

## Requirements

- Linux kernel 6.12+ with sched-ext support
- Clang/LLVM >= 16 (17 recommended)
- Rust toolchain >= 1.82
- Meson >= 1.2.0
- libbpf >= 1.2.2 (1.3 recommended)
- Dependencies: libelf, libz, libzstd, bpftool

## Important Notes

- The project uses meson as the primary build system
- Schedulers are loaded as kernel modules and require root privileges
- Production-ready schedulers are marked in schedulers.json
- Always test schedulers in a safe environment before production use