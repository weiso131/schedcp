# FAISS Benchmarking Guide

This guide provides instructions for benchmarking FAISS (Facebook AI Similarity Search) with CUDA support.

## Build Information

- **Location**: `schedcp/workloads/faiss/faiss/`
- **CUDA Version**: 12.9.86
- **BLAS Library**: OpenBLAS
- **Build Type**: Release with AVX2 optimizations
- **Library**: `build/faiss/libfaiss.a` (92 MB)
- **Demo Binary**: `build/demos/demo_ivfpq_indexing` (7.1 MB)

## Quick Start Benchmark

### 1. Basic CPU/GPU Demo

Run the built-in demo that creates a small index and performs searches:

```bash
cd schedcp/workloads/faiss/faiss
./build/demos/demo_ivfpq_indexing
```

This demo benchmarks:
- Index creation and storage
- Search performance
- Runtime: ~5s (GPU-enabled build)

## Running Benchmarks: CPU vs GPU vs UVM

### CPU-Only Benchmarks (Dataset > GPU VRAM, fits in RAM)

**Best for: Datasets bigger than GPU memory but smaller than system RAM**

#### SIFT1B on CPU (1 billion vectors, ~120GB)
```bash
cd schedcp/workloads/faiss/faiss/benchs

# Download SIFT1B dataset
mkdir -p bigann && cd bigann
wget ftp://ftp.irisa.fr/local/texmex/corpus/bigann_base.bvecs       # ~120GB
wget ftp://ftp.irisa.fr/local/texmex/corpus/bigann_learn.bvecs      # ~12GB
wget ftp://ftp.irisa.fr/local/texmex/corpus/bigann_query.bvecs      # ~1.2MB
mkdir -p gnd && cd gnd
wget ftp://ftp.irisa.fr/local/texmex/corpus/gnd/idx_1000M.ivecs     # Ground truth
cd ../..

# Run CPU benchmark (uses RAM only, no GPU)
# Full 1B vectors
python bench_polysemous_1bn.py SIFT1000M IMI2x12,PQ16 nprobe=16,max_codes=10000

# Smaller subsets for testing
python bench_polysemous_1bn.py SIFT100M IMI2x12,PQ16 nprobe=16      # 100M (~12GB RAM)
python bench_polysemous_1bn.py SIFT10M IMI2x12,PQ16 nprobe=16       # 10M (~1.2GB RAM)
```

**Performance:**
- Training: ~2 minutes
- Adding 1B vectors: ~3.1 hours (multithreaded CPU)
- Memory: Uses `mmap` for efficient large file handling
- No GPU required

#### Other CPU benchmarks:
```bash
# CPU flat index
python bench_index_flat.py

# CPU HNSW
python bench_hnsw.py
```

### GPU Benchmarks (Standard Device Memory)

#### SIFT1M on GPU (fits in VRAM)
```bash
cd schedcp/workloads/faiss/faiss/benchs

# Download SIFT1M
mkdir -p sift1M && cd sift1M
wget ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz
tar -xzf sift.tar.gz
cd ..

# Run GPU benchmark
python bench_gpu_sift1m.py
```

#### SIFT1B on GPU (Multi-GPU with memory optimization)
```bash
# Single GPU with memory limits
python bench_gpu_1bn.py SIFT1000M OPQ8_32,IVF262144,PQ8 \
    -nnn 10 \
    -ngpu 1 \
    -tempmem $((1536*1024*1024))

# Multi-GPU (4 GPUs)
python bench_gpu_1bn.py Deep1B OPQ20_80,IVF262144,PQ20 \
    -nnn 10 \
    -ngpu 4 \
    -R 2 \
    -altadd \
    -noptables \
    -tempmem $((1024*1024*1024))

# GPU benchmark options:
# -ngpu N          Number of GPUs
# -tempmem N       Temp GPU memory limit (bytes)
# -R N             Replicas (sharding)
# -noptables       Disable precomputed tables (saves memory)
# -altadd          Alternative add (avoids overflow)
# -float16         Use FP16 (saves memory)
# -qbs N           Query batch size
# -abs N           Add batch size
```

### UVM Benchmarks (CUDA Unified Memory)

**Best for: Datasets bigger than GPU VRAM but smaller than system RAM**

FAISS supports UVM via `MemorySpace::Unified` configuration:

#### Python UVM Example:
```python
# bench_uvm.py
import faiss
import numpy as np
import time

d = 128
nb = 20_000_000  # 20M vectors (~10GB, larger than most GPU VRAM)
nq = 10000

print(f"Dataset: {nb} vectors x {d}D = {nb*d*4/(1024**3):.2f} GB")

# Create GPU index with Unified Memory
res = faiss.StandardGpuResources()
config = faiss.GpuIndexFlatConfig()
config.memorySpace = faiss.MemorySpace_Unified  # Enable UVM!

index = faiss.GpuIndexFlatL2(res, d, config)

# Generate and add data
xb = np.random.rand(nb, d).astype('float32')
xq = np.random.rand(nq, d).astype('float32')

start = time.time()
index.add(xb)
print(f"Add time: {time.time() - start:.2f}s")

# Search
k = 10
start = time.time()
D, I = index.search(xq, k)
print(f"Search time: {time.time() - start:.2f}s")
print(f"QPS: {nq/(time.time()-start):.2f}")
```

```bash
python bench_uvm.py
```

#### C++ UVM Benchmark:
```bash
cd schedcp/workloads/faiss/faiss

# Build performance benchmarks (if not already built)
# cmake -B build -DBUILD_TESTING=ON ...
# make -C build -j$(nproc)

# Run with UVM
./build/faiss/gpu/perf/PerfFlat --use_unified_mem=true --dim=128 --num=10000000
```

### Comparison: CPU vs GPU vs UVM

| Mode | Memory | Dataset Size Limit | Speed | When to Use |
|------|--------|-------------------|-------|-------------|
| **CPU** | RAM | Up to RAM (~128GB) | Slower | Dataset > GPU VRAM, have time |
| **GPU** | VRAM | Up to VRAM (~24GB) | Fastest | Dataset fits in VRAM |
| **UVM** | Unified | Up to RAM (~128GB) | Medium* | Dataset > VRAM but < RAM |

*UVM speed: If working set fits in VRAM ≈ GPU speed. If paging required ≈ slower.

### Dataset Size Reference

| Dataset | Vectors | Dim | Float32 Size | Recommended Mode |
|---------|---------|-----|--------------|------------------|
| SIFT1M | 1M | 128 | ~500 MB | GPU |
| SIFT10M | 10M | 128 | ~5 GB | GPU (8GB+) or UVM |
| SIFT100M | 100M | 128 | ~50 GB | CPU or UVM |
| SIFT1B | 1B | 128 | ~500 GB | CPU (with mmap) |
| Deep1B | 1B | 96 | ~360 GB | CPU (with mmap) |

**For your 24GB GPU:**
- Direct GPU: Up to SIFT10M
- UVM: Up to SIFT100M (if you have 64GB+ RAM)
- CPU: SIFT1B (if you have 128GB+ RAM)

## Performance Metrics to Track

### Search Performance
- **Latency**: Time per query (ms)
- **Throughput**: Queries per second (QPS)
- **Recall@k**: Accuracy of approximate search (R@1, R@10, R@100)

### Index Building
- **Training time**: Time to train quantizers
- **Adding time**: Time to add vectors to index
- **Index size**: Memory footprint (MB/GB)

### GPU Metrics
- **GPU memory usage**: Peak VRAM consumption
- **GPU utilization**: % compute utilization
- **Batch size impact**: Performance vs batch size
- **Multi-GPU scaling**: Speedup with 2/4/8 GPUs

## Monitoring GPU During Benchmarks

```bash
# Terminal 1: Run benchmark
cd schedcp/workloads/faiss/faiss
./build/demos/demo_ivfpq_indexing

# Terminal 2: Monitor GPU
watch -n 0.5 nvidia-smi
```

Or use:
```bash
nvidia-smi dmon -s pucvmet -d 1
```

## Troubleshooting

### CUDA Out of Memory
- Reduce batch size with `-qbs` or `-abs`
- Use `-tempmem` parameter to limit temp memory allocation
- Try `-noptables` to disable precomputed tables
- Use `-float16` for half-precision
- Consider UVM (`MemorySpace::Unified`)

### Slow Performance
- Ensure GPU is not throttling: `nvidia-smi -q -d PERFORMANCE`
- Check CUDA version compatibility
- Verify BLAS library (MKL is faster than OpenBLAS)

### Python Import Errors
- Install faiss-gpu: `conda install -c pytorch faiss-gpu`
- Or use pip: `pip install faiss-gpu`

## Recommended Benchmark Progression

### Level 1: Quick Verification (Minutes)
```bash
# 1. CPU/GPU demo
cd schedcp/workloads/faiss/faiss
./build/demos/demo_ivfpq_indexing
```

### Level 2: Small-Scale (10-30 minutes)
```bash
cd benchs
# Download SIFT1M
mkdir -p sift1M && cd sift1M
wget ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz
tar -xzf sift.tar.gz
cd ..

# Run GPU benchmark
python bench_gpu_sift1m.py
```

### Level 3: Medium-Scale (1-2 hours)
```bash
# Test with 10M or 100M subsets
python bench_polysemous_1bn.py SIFT10M IMI2x12,PQ16 nprobe=16
python bench_polysemous_1bn.py SIFT100M IMI2x12,PQ16 nprobe=16
```

### Level 4: Large-Scale UVM Test (Dataset > VRAM but < RAM)
```bash
# Create and run UVM test with 20M vectors
python3 << 'EOF'
import faiss
import numpy as np
import time

d = 128
nb = 20_000_000  # ~10GB
nq = 10000

print(f"Dataset size: {nb*d*4/(1024**3):.2f} GB")

res = faiss.StandardGpuResources()
config = faiss.GpuIndexFlatConfig()
config.memorySpace = faiss.MemorySpace_Unified

index = faiss.GpuIndexFlatL2(res, d, config)

xb = np.random.rand(nb, d).astype('float32')
xq = np.random.rand(nq, d).astype('float32')

start = time.time()
index.add(xb)
print(f"Add time: {time.time()-start:.2f}s")

k = 10
start = time.time()
D, I = index.search(xq, k)
print(f"Search time: {time.time()-start:.2f}s")
print(f"QPS: {nq/(time.time()-start):.2f}")
EOF
```

### Level 5: Billion-Scale CPU (Hours)
```bash
# Full SIFT1B on CPU (requires ~128GB RAM)
python bench_polysemous_1bn.py SIFT1000M IMI2x12,PQ16 nprobe=16,max_codes=10000
```

## References

- [FAISS Wiki](https://github.com/facebookresearch/faiss/wiki)
- [Benchmarks README](faiss/benchs/README.md)
- [GPU Paper](https://arxiv.org/abs/1702.08734) - Billion-scale similarity search with GPUs
- [Polysemous Codes Paper](https://arxiv.org/abs/1609.01882) - SIFT1B benchmarks
