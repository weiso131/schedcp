# ML-Based Scheduler

This is a standalone implementation of an ML-based scheduler inspired by the scx_rusty modifications and the LWN article on machine learning schedulers.

## Overview

This scheduler uses a neural network model to make task migration decisions based on system metrics such as:
- CPU utilization
- Domain load balancing
- Idle/active CPU counts

## Architecture

The implementation consists of:
- **TensorFlow Integration**: Uses TensorFlow for ML inference
- **BPF Interface**: Structures for communicating with kernel-side BPF programs
- **Migration Decision Engine**: ML-based logic for task migration
- **Model Training**: Python script for training and exporting models

## Building

```bash
# Install dependencies
make install-deps

# Generate a sample model
make model

# Build the scheduler
make build

# Run the scheduler
make run
```

## Key Components

1. **lib.rs**: Core ML scheduler implementation with TensorFlow model loading and prediction
2. **main.rs**: Standalone executable for testing the ML scheduler
3. **bpf_interface.rs**: Data structures for BPF communication
4. **scheduler_integration.rs**: Integration layer for processing migration requests
5. **model_export.py**: Python script for training and exporting ML models

## Integration with sched-ext

In a full implementation, this would integrate with the sched-ext framework through:
- BPF programs that collect system metrics
- User-space daemon that runs the ML model
- BPF maps for sharing migration decisions

## Performance

Based on the research, ML-based schedulers can achieve:
- 10% improvement in kernel compilation time
- 77% reduction in task migrations
- Dynamic adaptation to different workload patterns