# PyTorch UVM Benchmark

This directory contains a PyTorch benchmark with CUDA Unified Virtual Memory (UVM) support, demonstrating how to train models larger than GPU memory capacity.

## Overview

This benchmark implements a custom CUDA allocator using `cudaMallocManaged` to enable PyTorch to train models that exceed GPU physical memory by transparently using CPU memory. The implementation includes:

- **Custom UVM Allocator** (`uvm_allocator.c`): C-based CUDA allocator with memory statistics
- **GNN Benchmark** (`benchmark_gnn_uvm.py`): Graph Neural Network training benchmark
- **Memory Statistics**: Real-time tracking of allocations, deallocations, and peak usage

## Key Results

### Successful Test (2M nodes)

**Without UVM:**
```
✗ OOM during training
Error: CUDA out of memory. Tried to allocate 19.07 GiB
```

**With UVM + CUDA_MANAGED_FORCE_DEVICE_ALLOC=1:**
```
✓ SUCCESS
Peak allocated: 44.47GB (GPU physical memory: 33.7GB)
Epochs: 2
Avg time/epoch: 1.66s
```

This demonstrates **~32% memory oversubscription** - successfully training a model that requires 44.47GB on a GPU with only 33.7GB physical memory.

## Architecture

### 1. UVM Allocator (`uvm_allocator.c`)

The custom allocator provides:

```c
// Core allocation functions
void* uvm_malloc(ssize_t size, int device, cudaStream_t stream)
void uvm_free(void* ptr, size_t size, int device, cudaStream_t stream)

// Statistics functions
size_t uvm_get_allocated_bytes(void)
size_t uvm_get_peak_allocated_bytes(void)
size_t uvm_get_num_allocs(void)
size_t uvm_get_num_frees(void)
void uvm_reset_peak_stats(void)
```

**Key Features:**
- Uses `cudaMallocManaged` for unified memory allocation
- Thread-safe statistics using atomic operations
- Prefetches memory to GPU with `cudaMemPrefetchAsync`
- Logs large allocations (>100MB) for debugging

**Memory Tracking:**
- Maintains `total_allocated`, `peak_allocated`, `num_allocs`, `num_frees`
- Uses atomic operations for thread safety
- Provides real-time visibility into allocation patterns

### 2. GNN Benchmark (`benchmark_gnn_uvm.py`)

A Graph Neural Network (GCN) training benchmark that:

- Generates random graphs with configurable size
- Implements simplified 2-layer GCN without external dependencies
- Tests memory limits with/without UVM
- Reports detailed memory statistics

**Key Components:**
```python
class SimpleGCNLayer(nn.Module):
    """Simplified GCN layer using index_add_ for message passing"""

class GCN(nn.Module):
    """2-layer GCN model"""

def enable_uvm_allocator():
    """Load and enable UVM allocator via CUDAPluggableAllocator"""
```

## Installation

```bash
# Create virtual environment
uv venv

# Install dependencies
uv pip install torch psutil

# Build UVM allocator
make
```

## Usage

### Basic Usage

```bash
# Test without UVM (expect OOM for large graphs)
python benchmark_gnn_uvm.py --nodes=2000000

# Test with UVM (expect success)
CUDA_MANAGED_FORCE_DEVICE_ALLOC=1 python benchmark_gnn_uvm.py --nodes=2000000 --use_uvm
```

### Configuration Options

```
--nodes=N              Number of nodes (default: 10M)
--edges_per_node=N     Average edges per node (default: 10)
--features=N           Feature dimension (default: 128)
--hidden=N             Hidden dimension (default: 256)
--epochs=N             Number of epochs (default: 3)
--use_uvm              Enable UVM allocator
```

### Example Tests

```bash
# Small graph baseline (1M nodes - should work)
python benchmark_gnn_uvm.py --nodes=1000000 --epochs=2

# Medium graph without UVM (2M nodes - will OOM during training)
python benchmark_gnn_uvm.py --nodes=2000000 --epochs=2

# Medium graph with UVM (2M nodes - SUCCESS)
CUDA_MANAGED_FORCE_DEVICE_ALLOC=1 python benchmark_gnn_uvm.py --nodes=2000000 --epochs=2 --use_uvm

# Large graph with UVM (10M nodes - will OOM during data loading)
python benchmark_gnn_uvm.py --nodes=10000000 --epochs=2 --use_uvm
```

## Implementation Details

### Memory Allocation Pattern (2M nodes benchmark)

From UVM allocator logs, we observed:

```
[UVM] Alloc #1:  0.32 GB  - Initial graph data
[UVM] Alloc #2:  1.02 GB  - Node features
[UVM] Alloc #9:  2.05 GB  - Edge indices
[UVM] Alloc #13: 20.48 GB - Temporary buffer (index_add_ operation)
[UVM] Alloc #70: 20.48 GB - Backward pass temporary buffer
Peak allocated:  44.47 GB
```

**Key Observations:**
1. The largest allocations (20.48 GB each) are temporary buffers for `index_add_` operations during forward/backward passes
2. Peak memory usage (44.47 GB) occurs during backward pass
3. PyTorch's caching allocator reuses memory, so actual unique allocations are smaller
4. With 169 allocations and 130 frees, there's active memory management throughout training

### PyTorch CUDAPluggableAllocator Integration

The UVM allocator integrates with PyTorch via the `CUDAPluggableAllocator` API:

```python
# Load shared library
uvm_lib = ctypes.CDLL('uvm_allocator.so')

# Define function signatures for statistics
uvm_lib.uvm_get_allocated_bytes.restype = ctypes.c_size_t
uvm_lib.uvm_get_peak_allocated_bytes.restype = ctypes.c_size_t

# Register with PyTorch
allocator = torch.cuda.memory.CUDAPluggableAllocator(
    'uvm_allocator.so',
    'uvm_malloc',
    'uvm_free'
)
torch.cuda.memory.change_current_allocator(allocator)
```

**Important:** The allocator must be enabled **before** any CUDA operations, including `torch.cuda.get_device_name()`.

### Known Limitations

1. **CUDAPluggableAllocator doesn't support getDeviceStats**
   - `torch.cuda.memory_allocated()` will fail with UVM allocator
   - Workaround: Use custom statistics functions via ctypes
   - See: [PyTorch Issue #133281](https://github.com/pytorch/pytorch/issues/133281)

2. **CUBLAS Internal Allocations**
   - cuBLAS creates internal handles with `cudaMalloc` (not managed by PyTorch)
   - When GPU memory is full with UVM allocations, cuBLAS fails with `CUBLAS_STATUS_ALLOC_FAILED`
   - Setting `CUDA_MANAGED_FORCE_DEVICE_ALLOC=1` helps but doesn't fully solve the issue
   - This limits maximum graph size (~2-3M nodes on 32GB GPU)

3. **Scaling Limits**
   - **1M nodes**: Works without UVM
   - **2M nodes**: Requires UVM, succeeds with `CUDA_MANAGED_FORCE_DEVICE_ALLOC=1`
   - **3M nodes**: Fails due to cuBLAS allocation conflicts
   - **10M nodes**: Requires more sophisticated memory management

## Technical Background

### CUDA Unified Virtual Memory (UVM)

UVM provides a single unified memory space accessible from both CPU and GPU. Key characteristics:

- **Automatic Migration**: CUDA driver automatically migrates pages between CPU and GPU
- **Oversubscription**: Total allocated memory can exceed GPU physical memory
- **Page Faults**: GPU page faults trigger migration from CPU to GPU
- **Performance**: Slower than explicit memory management, but enables larger models

### Environment Variables

- `CUDA_MANAGED_FORCE_DEVICE_ALLOC=1`: Forces driver to prefer GPU for physical storage
  - Required for compatibility with some CUDA libraries (cuBLAS)
  - Limits oversubscription capability
  - See: [CUDA Runtime API Documentation](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html)

## Future Work

Potential improvements to explore:

1. **Hybrid Allocator**: Reserve some GPU memory for cuBLAS, use UVM for tensors
2. **Manual Prefetching**: Use `cudaMemPrefetchAsync` strategically to reduce page faults
3. **Memory-Efficient GCN**: Implement scatter/gather operations with lower memory footprint
4. **Multi-GPU UVM**: Explore UVM with multiple GPUs (requires peer-to-peer compatibility)
5. **Alternative Libraries**: Test with PyG (PyTorch Geometric) or DGL for comparison

## References

- [PyTorch CUDA Memory Management](https://pytorch.org/docs/stable/notes/cuda.html)
- [PyTorch CUDAPluggableAllocator](https://pytorch.org/docs/stable/torch_cuda_memory.html)
- [NVIDIA Unified Memory](https://developer.nvidia.com/blog/unified-memory-cuda-beginners/)
- [CUDA Runtime API](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html)
- [cuBLAS Library User Guide](https://docs.nvidia.com/cuda/cublas/index.html)

## Files

- `uvm_allocator.c` - Custom CUDA UVM allocator with statistics
- `uvm_allocator.so` - Compiled shared library
- `benchmark_gnn_uvm.py` - GNN training benchmark
- `Makefile` - Build configuration
- `test_pytorch.py` - PyTorch installation verification
- `.gitignore` - Git ignore patterns

## Development Notes

### Build System

The Makefile compiles the UVM allocator:

```makefile
CUDA_PATH ?= /usr/local/cuda
CC = gcc
CFLAGS = -shared -fPIC -O3
INCLUDES = -I$(CUDA_PATH)/include
LDFLAGS = -L$(CUDA_PATH)/lib64 -lcudart

uvm_allocator.so: uvm_allocator.c
    $(CC) $(CFLAGS) $(INCLUDES) uvm_allocator.c $(LDFLAGS) -o uvm_allocator.so
```

### Debugging Tips

1. **Enable verbose logging**: The allocator logs allocations >100MB to stderr
2. **Check memory stats**: Use `uvm_get_allocated_bytes()` to track memory usage
3. **Monitor with nvidia-smi**: Watch GPU memory usage in real-time
4. **Profile with nsight**: Use Nsight Systems to analyze UVM page faults

### Testing Strategy

Our testing methodology:

1. **Baseline**: Test without UVM to establish OOM threshold
2. **UVM Enabled**: Test same configuration with UVM
3. **Scale Up**: Incrementally increase graph size to find new limits
4. **Analyze Logs**: Review allocation patterns to understand bottlenecks

## Related Work

- **llama.cpp**: Uses UVM for large language model inference on GH200 systems ([Issue #5026](https://github.com/ggml-org/llama.cpp/issues/5026))
- **PyTorch Request**: Discussion about native UVM support ([Issue #124296](https://github.com/pytorch/pytorch/issues/124296))
- **TorchRec**: Feature request for UVM embedding support ([Issue #125](https://github.com/meta-pytorch/torchrec/issues/125))
