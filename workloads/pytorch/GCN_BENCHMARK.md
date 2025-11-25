# GCN Benchmark

No overscription (

```

======================================================================
Results Summary
======================================================================
Avg epoch time: 15.719s
Median epoch time: 15.719s
Total training time: 15.72s

Accuracy:
  train: 0.0998

Memory Usage:
  GPU allocated: 6.62 GB
  CPU used: 1.61 GB
  Total: 8.24 GB

UVM Statistics:
  Peak allocated: 27.07 GB
  Allocations: 1015
  Frees: 856
======================================================================
```

baseline

```

======================================================================
Results Summary
======================================================================
Avg epoch time: 71.000s
Median epoch time: 71.000s
Total training time: 71.00s

Accuracy:
  train: 0.1001

Memory Usage:
  GPU allocated: 11.03 GB
  CPU used: 1.61 GB
  Total: 12.64 GB

UVM Statistics:
  Peak allocated: 45.11 GB
  Allocations: 1611
  Frees: 1368
======================================================================
```

uvm

```
======================================================================
Results Summary
======================================================================
Avg epoch time: 27.429s
Median epoch time: 27.429s
Total training time: 27.43s

Accuracy:
  train: 0.1001

Memory Usage:
  GPU allocated: 11.03 GB
  CPU used: 1.62 GB
  Total: 12.64 GB

UVM Statistics:
  Peak allocated: 45.11 GB
  Allocations: 1611
  Frees: 1368
======================================================================
```

## Overview

This benchmark implements a standard GCN (Graph Convolutional Network) with symmetric normalization following the canonical formulation: **D^{-1/2}(A + I)D^{-1/2}**.

Key features:
- ✅ Normalized GCN aggregation (symmetric + self-loops)
- ✅ Two propagation modes: SpMM (standard) and Chunked (memory-efficient)
- ✅ Standard dataset support (ogbn-arxiv, ogbn-products via OGB)
- ✅ UVM support for extreme-scale experiments
- ✅ Reproducible evaluation with warmup, synchronization, and JSON output

```

# Test 1: Small graph, SpMM mode
echo ""
echo "[Test 1/4] Small graph (100K nodes) - SpMM mode"
echo "Expected: Fast execution, ~0.005s per epoch"
python benchmark_gnn_uvm.py --dataset random --nodes 100000 \
    --edges_per_node 10 --features 64 --hidden 128 \
    --epochs 2 --warmup 1 --prop spmm \
    --report_json test_results_spmm_small.json

# Test 2: Small graph, Chunked mode
echo ""
echo "[Test 2/4] Small graph (100K nodes) - Chunked mode"
echo "Expected: Similar speed to SpMM, less GPU memory"
python benchmark_gnn_uvm.py --dataset random --nodes 100000 \
    --edges_per_node 10 --features 64 --hidden 128 \
    --epochs 2 --warmup 1 --prop chunked \
    --report_json test_results_chunked_small.json

# Test 3: Medium graph without UVM
echo ""
echo "[Test 3/4] Medium graph (1M nodes) - No UVM"
echo "Expected: Success, ~0.2s per epoch"
python benchmark_gnn_uvm.py --dataset random --nodes 1000000 \
    --edges_per_node 10 --features 128 --hidden 256 \
    --epochs 2 --warmup 1 --prop chunked \
    --report_json test_results_medium_no_uvm.json

# Test 4: Large graph with UVM (oversubscription)
echo ""
echo "[Test 4/4] Large graph (10M nodes) - UVM Oversubscription"
echo "Expected: Success but slow (~70s per epoch), peak memory > GPU capacity"
export CUDA_MANAGED_FORCE_DEVICE_ALLOC=1
python benchmark_gnn_uvm.py --dataset random --nodes 10000000 \
    --edges_per_node 10 --features 128 --hidden 256 \
    --epochs 1 --warmup 0 --prop chunked --use_uvm \
    --report_json test_results_large_uvm.json 2>&1 | grep -v "^\[UVM\] Alloc"
```

## Test Environment

- **GPU**: NVIDIA GeForce RTX 5090 (31.36 GB usable)
- **PyTorch**: 2.9.0+cu128
- **CUDA**: 12.8

---

## Experiment 1: Standard Scale (1M nodes)

### Configuration
- **Nodes**: 1,000,000
- **Edges**: 10,000,000 (avg 10 per node)
- **Features**: 128 dimensions
- **Hidden**: 256 dimensions
- **Epochs**: 3 (with 1 warmup)

### Results: SpMM Mode (Standard Sparse Matrix Multiplication)

| Metric | Value |
|--------|-------|
| Avg Epoch Time | **0.141s** |
| Median Epoch Time | 0.141s |
| Total Training Time | 0.424s |
| GPU Memory | **1.54 GB** |
| CPU Memory | 1.66 GB |
| Total Memory | 3.20 GB |
| Train Accuracy | 0.1059 |

### Results: Chunked Mode (Memory-Efficient)

| Metric | Value |
|--------|-------|
| Avg Epoch Time | **0.218s** |
| Median Epoch Time | 0.218s |
| Total Training Time | 0.655s |
| GPU Memory | **1.12 GB** |
| CPU Memory | 1.64 GB |
| Total Memory | 2.76 GB |
| Train Accuracy | 0.1060 |

### Key Observations

1. **Memory Efficiency**: Chunked mode uses **27% less GPU memory** (1.12 GB vs 1.54 GB)
   - Avoids creating [E, F] intermediate tensors
   - Processes edges in chunks of 2M

2. **Performance Trade-off**: Chunked mode is **54% slower** (0.218s vs 0.141s per epoch)
   - More kernel launches due to chunked processing
   - But enables training graphs that exceed GPU memory

3. **Accuracy**: Both methods produce identical results
   - Same GCN formulation with symmetric normalization

---

## Experiment 2: Oversubscription (10M nodes)

### Configuration
- **Nodes**: 10,000,000
- **Edges**: 100,000,000 (avg 10 per node)
- **Features**: 128 dimensions
- **Hidden**: 256 dimensions
- **Epochs**: 2 (with 1 warmup)
- **Mode**: Chunked (required for this scale)

### Without UVM: **OUT OF MEMORY** ❌

```
torch.OutOfMemoryError: CUDA out of memory.
Tried to allocate 1.91 GiB. GPU 0 has a total capacity of 31.36 GiB
of which 1.40 GiB is free. Including non-PyTorch memory, this process
has 29.94 GiB memory in use.
```

- Training **fails during warmup epoch**
- Peak memory exceeds GPU capacity (~30 GB allocated)

### With UVM: **SUCCESS** ✅

| Metric | Value |
|--------|-------|
| Avg Epoch Time | **69.87s** |
| Median Epoch Time | 69.91s |
| Total Training Time | 139.74s |
| GPU Memory (reported) | 11.03 GB |
| CPU Memory | 1.61 GB |
| **UVM Peak Allocated** | **45.11 GB** |
| UVM Allocations | 3,857 |
| UVM Frees | 3,186 |
| Train Accuracy | 0.1007 |

### Critical Insights

1. **Oversubscription Achieved**: UVM peak allocation (45.11 GB) **exceeds GPU capacity** (31.36 GB)
   - Successfully trains a graph that would otherwise OOM
   - Data seamlessly migrates between GPU and system memory

2. **Performance Impact**:
   - **495× slower** than 1M node baseline (69.87s vs 0.141s per epoch)
   - But training **completes successfully** vs complete failure without UVM

3. **Memory Thrashing**:
   - 3,857 allocations during 3 epochs (warmup + 2 measured)
   - ~1,286 allocations per epoch suggests frequent memory management overhead
   - Large 10.24 GB allocations indicate PyTorch's internal tensor operations

4. **Trade-off Analysis**:
   - **Without UVM**: Fast but limited by GPU memory → OOM at 10M nodes
   - **With UVM**: 495× slower but enables extreme-scale training

---

## Performance Summary Table

| Configuration | Nodes | Mode | UVM | Epoch Time | GPU Mem | Peak Mem | Result |
|---------------|-------|------|-----|------------|---------|----------|--------|
| Small (baseline) | 1M | SpMM | ❌ | 0.141s | 1.54 GB | 1.54 GB | ✅ Fast |
| Small (efficient) | 1M | Chunked | ❌ | 0.218s | 1.12 GB | 1.12 GB | ✅ Memory-efficient |
| Large (no UVM) | 10M | Chunked | ❌ | N/A | N/A | >31 GB | ❌ **OOM** |
| Large (UVM) | 10M | Chunked | ✅ | 69.87s | 11.03 GB | **45.11 GB** | ✅ **Oversubscription** |

---

## Recommendations for OSDI/NSDI Submissions

### When to Use SpMM Mode
- Standard accuracy baselines on OGB datasets (ogbn-arxiv, ogbn-products)
- When full graph fits in GPU memory
- Maximum performance is critical

### When to Use Chunked Mode
- Memory-constrained scenarios
- Very large graphs with many features
- Demonstrating memory-efficient algorithms

### When to Use UVM
- **Oversubscription experiments** (dataset > GPU memory)
- Demonstrating scalability beyond hardware limits
- Studying memory management trade-offs
- **Note**: Expect 100-500× slowdown due to PCIe transfer overhead

### Reporting Guidelines

For reproducibility, report:
1. **Hardware**: GPU model, memory capacity, PCIe generation
2. **Configuration**: Node count, edge count, feature dimensions, hidden dimensions
3. **Timing**: Warmup epochs, measured epochs, synchronization method
4. **Memory**: GPU allocated, CPU used, UVM peak (if applicable)
5. **Accuracy**: Train/valid/test splits (for OGB datasets)

---

## Usage Examples

### Standard Evaluation (1M nodes, fits in memory)
```bash
# SpMM mode - fastest
python benchmark_gnn_uvm.py --dataset random --nodes 1000000 \
    --prop spmm --epochs 5 --report_json results.json

# Chunked mode - memory-efficient
python benchmark_gnn_uvm.py --dataset random --nodes 1000000 \
    --prop chunked --chunk_size 2000000 --epochs 5
```

### Oversubscription Experiment (10M nodes, exceeds GPU memory)
```bash
# This will OOM without UVM
python benchmark_gnn_uvm.py --dataset random --nodes 10000000 \
    --prop chunked --epochs 2

# This succeeds with UVM
export CUDA_MANAGED_FORCE_DEVICE_ALLOC=1
python benchmark_gnn_uvm.py --dataset random --nodes 10000000 \
    --prop chunked --epochs 2 --use_uvm --report_json results_uvm.json
```

### OGB Datasets (requires: pip install ogb)
```bash
# ogbn-arxiv (169K nodes, 1.1M edges)
python benchmark_gnn_uvm.py --dataset ogbn-arxiv --prop spmm --epochs 100

# ogbn-products (2.4M nodes, 61M edges)
python benchmark_gnn_uvm.py --dataset ogbn-products --prop chunked --epochs 50
```



---
