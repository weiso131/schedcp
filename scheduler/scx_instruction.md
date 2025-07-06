# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

sched_ext (SCX) is a Linux kernel feature that enables implementing kernel thread schedulers in BPF and dynamically loading them. This repository contains various scheduler implementations and support utilities. The project is production-ready with deployments at Meta and Google.

## Build Commands

```bash
# Initial setup
meson setup build --prefix ~

# Build all schedulers
meson compile -C build

# Build specific scheduler
meson compile -C build scx_simple

# Build only C schedulers
cd scheds/c && meson setup build && meson compile -C build

# Build only Rust schedulers
cd scheds/rust && meson setup build && meson compile -C build

# Install schedulers
meson install -C build
```

## Testing Commands

```bash
# Run all tests
meson test -v -C build

# Run specific test
meson test -v -C build scx_p2dq_test

# Run Rust tests
cargo test
cargo nextest run

# Test specific scheduler (requires virtme-ng)
meson compile -C build test_sched_scx_simple

# Run stress tests
meson compile -C build stress_tests_scx_simple

# Run BPF verification statistics
meson compile -C build veristat_scx_simple
```

## Linting and Formatting

```bash
# Format Rust code
cargo fmt

# Check Rust formatting
cargo fmt -- --check

# Run Clippy
cargo clippy -- -Dwarnings

# Format Python files
black .github/include/**/*.py
isort .github/include/**/*.py

# Run full CI check locally
python3 ./.github/include/ci.py all
```

## Architecture

### Repository Structure
- `scheds/include/`: Shared BPF and C include files for schedulers
- `scheds/c/`: C-based example schedulers
- `scheds/rust/`: Production Rust schedulers
- `rust/scx_utils/`: Common Rust utility library
- `.github/`: CI/CD workflows and scripts

### Key Components
1. **BPF Schedulers**: Core scheduling logic in BPF programs
2. **User-space Components**: Configuration and runtime control
3. **Dispatch Queues (DSQs)**: Interface between kernel and BPF scheduler
4. **struct_ops**: BPF feature for exporting callbacks

### Scheduler Types
- **Production Rust schedulers**: scx_bpfland, scx_lavd, scx_layered, scx_rusty
- **C example schedulers**: scx_simple, scx_qmap, scx_central, scx_flatcg

## Development Workflow

1. **Making Changes to Schedulers**:
   - BPF logic: Edit `*.bpf.c` files
   - User-space logic: Edit corresponding `.c` or `main.rs` files
   - Shared headers: Edit files in `scheds/include/`

2. **Adding New Schedulers**:
   - Create new directory under `scheds/c/` or `scheds/rust/`
   - Follow existing scheduler structure
   - Add to `meson.build` file
   - Include BPF program and user-space component

3. **Testing Changes**:
   - Build: `meson compile -C build <scheduler_name>`
   - Run: `sudo ./build/scheds/c/<scheduler_name>` (requires root)
   - Monitor: Add `--monitor 1` flag for statistics
   - Debug: Use `--verbose` flag for detailed logs

## Important Dependencies

- Linux kernel >= 6.12 with CONFIG_SCHED_CLASS_EXT=y
- clang >= 16
- libbpf >= 1.2.2
- Rust toolchain >= 1.82
- meson >= 1.2

## Key Development Patterns

### BPF Scheduler Structure
```c
// In *.bpf.c files
struct sched_ext_ops ops = {
    .select_cpu = (void *)select_cpu,
    .enqueue = (void *)enqueue,
    .dispatch = (void *)dispatch,
    .name = "scheduler_name",
};
```

### User-space Pattern
```c
// In *.c files
1. Parse arguments
2. Load and attach BPF program
3. Monitor statistics
4. Handle signals for cleanup
```

### Rust Scheduler Pattern
```rust
// In main.rs files
1. Define scheduler struct
2. Implement BpfScheduler trait
3. Set up stats collection
4. Run main loop with signal handling
```

## Debugging Tools

- `scxtop`: Real-time scheduler statistics
- `bpftool prog`: List loaded BPF programs
- `retsnoop`: Trace kernel functions
- `bpftrace`: Dynamic tracing
- `dmesg`: Kernel logs for scheduler messages

## Common Issues and Solutions

1. **Permission denied**: Run schedulers with sudo
2. **Kernel not supported**: Check CONFIG_SCHED_CLASS_EXT=y
3. **Build failures**: Ensure all dependencies are installed
4. **Scheduler not starting**: Check dmesg for BPF verification errors

## Code Style Guidelines

- Follow existing code patterns in each scheduler
- Use consistent indentation (tabs for C, 4 spaces for Rust)
- Keep BPF programs simple and verifiable
- Document complex scheduling decisions
- Add meaningful statistics for monitoring