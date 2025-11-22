# PyTorch Benchmark Workloads

This directory contains PyTorch-based benchmark workloads for testing and evaluating scheduler performance.

## Overview

PyTorch workloads are used to evaluate scheduler behavior under various deep learning scenarios including:
- Model training (batch processing)
- Model inference (latency-sensitive)
- Data loading and preprocessing
- GPU-CPU coordination

## Workloads

(Add specific PyTorch benchmark workloads here)

## Requirements

- Python 3.8+
- PyTorch
- CUDA toolkit (for GPU workloads)

## Installation

```bash
pip install torch torchvision torchaudio
```

## Usage

(Add usage instructions for running PyTorch benchmarks)

## Metrics

Key performance metrics to collect:
- Training throughput (samples/sec)
- Inference latency (ms)
- GPU utilization
- CPU utilization
- Memory usage
- Scheduling delays

## Integration with schedcp

These workloads can be profiled and optimized using the schedcp MCP server for intelligent scheduler selection.
