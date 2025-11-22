#!/usr/bin/env python3
"""
GNN Training Benchmark with UVM

Tests whether UVM allows training larger graphs that would OOM with traditional PyTorch.

Usage:
    # Without UVM (expect OOM for large graphs)
    python benchmark_gnn_uvm.py --nodes=50000000

    # With UVM (expect success)
    python benchmark_gnn_uvm.py --nodes=50000000 --use_uvm

Configuration:
    --nodes: Number of nodes (default: 10M)
    --edges_per_node: Average edges per node (default: 10)
    --features: Feature dimension (default: 128)
    --hidden: Hidden dimension (default: 256)
    --epochs: Number of epochs (default: 3)
    --use_uvm: Enable UVM allocator
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import argparse
import psutil
import os
import ctypes


# Simple GCN layer (no external dependencies needed)
class SimpleGCNLayer(nn.Module):
    """Simplified GCN layer without torch_geometric dependency"""
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x, edge_index):
        # Simple message passing: aggregate neighbors
        num_nodes = x.size(0)
        out = self.linear(x)

        # Aggregate from neighbors (simplified, not optimized)
        # For each edge (src, dst), add src's features to dst
        src, dst = edge_index
        aggregated = torch.zeros_like(out)
        aggregated.index_add_(0, dst, out[src])

        return aggregated


class GCN(nn.Module):
    """2-layer GCN model"""
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.conv1 = SimpleGCNLayer(in_dim, hidden_dim)
        self.conv2 = SimpleGCNLayer(hidden_dim, out_dim)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return x


def generate_random_graph(num_nodes, num_edges, num_features, device='cuda'):
    """Generate a random graph"""
    print(f"  Generating graph with {num_nodes:,} nodes and {num_edges:,} edges...")

    # Random edges
    edge_index = torch.randint(0, num_nodes, (2, num_edges), device=device)

    # Random node features
    x = torch.randn(num_nodes, num_features, device=device)

    # Random labels
    y = torch.randint(0, 10, (num_nodes,), device=device)

    return x, edge_index, y


def get_memory_stats(uvm_lib=None):
    """Get current memory usage"""
    cpu_used = psutil.Process().memory_info().rss / 1e9

    if uvm_lib is not None:
        # Use UVM allocator's custom statistics
        gpu_allocated = uvm_lib.uvm_get_allocated_bytes() / 1e9
        gpu_peak = uvm_lib.uvm_get_peak_allocated_bytes() / 1e9
        return {
            'gpu_allocated': gpu_allocated,
            'gpu_reserved': gpu_peak,
            'cpu_used': cpu_used,
            'total': gpu_allocated + cpu_used
        }
    else:
        gpu_allocated = torch.cuda.memory_allocated() / 1e9
        gpu_reserved = torch.cuda.memory_reserved() / 1e9
        return {
            'gpu_allocated': gpu_allocated,
            'gpu_reserved': gpu_reserved,
            'cpu_used': cpu_used,
            'total': gpu_allocated + cpu_used
        }


def print_memory(prefix="", uvm_lib=None):
    """Print current memory usage"""
    mem = get_memory_stats(uvm_lib)
    print(f"    {prefix}GPU: {mem['gpu_allocated']:.1f}GB / {mem['gpu_reserved']:.1f}GB, "
          f"CPU: {mem['cpu_used']:.1f}GB, Total: {mem['total']:.1f}GB")


def train_epoch(model, x, edge_index, y, optimizer, train_mask):
    """Train for one epoch"""
    model.train()
    optimizer.zero_grad()

    # Forward
    out = model(x, edge_index)

    # Loss on training nodes
    loss = F.cross_entropy(out[train_mask], y[train_mask])

    # Backward
    loss.backward()
    optimizer.step()

    return loss.item()


def enable_uvm_allocator():
    """Enable UVM allocator and return library handle for statistics"""
    so_path = os.path.join(os.path.dirname(__file__), 'uvm_allocator.so')

    if not os.path.exists(so_path):
        raise FileNotFoundError(
            f"UVM allocator not found at {so_path}\n"
            f"Please run 'make' first to build uvm_allocator.so"
        )

    # Load the library for statistics access
    uvm_lib = ctypes.CDLL(so_path)

    # Define function signatures
    uvm_lib.uvm_get_allocated_bytes.restype = ctypes.c_size_t
    uvm_lib.uvm_get_allocated_bytes.argtypes = []

    uvm_lib.uvm_get_peak_allocated_bytes.restype = ctypes.c_size_t
    uvm_lib.uvm_get_peak_allocated_bytes.argtypes = []

    uvm_lib.uvm_get_num_allocs.restype = ctypes.c_size_t
    uvm_lib.uvm_get_num_allocs.argtypes = []

    uvm_lib.uvm_get_num_frees.restype = ctypes.c_size_t
    uvm_lib.uvm_get_num_frees.argtypes = []

    # Enable the allocator in PyTorch
    allocator = torch.cuda.memory.CUDAPluggableAllocator(
        so_path,
        'uvm_malloc',
        'uvm_free'
    )
    torch.cuda.memory.change_current_allocator(allocator)
    print("[UVM] UVM allocator enabled")

    return uvm_lib


def main():
    parser = argparse.ArgumentParser(description='GNN Training Benchmark with UVM')
    parser.add_argument('--nodes', type=int, default=10_000_000,
                       help='Number of nodes (default: 10M)')
    parser.add_argument('--edges_per_node', type=int, default=10,
                       help='Average edges per node (default: 10)')
    parser.add_argument('--features', type=int, default=128,
                       help='Feature dimension (default: 128)')
    parser.add_argument('--hidden', type=int, default=256,
                       help='Hidden dimension (default: 256)')
    parser.add_argument('--epochs', type=int, default=3,
                       help='Number of epochs (default: 3)')
    parser.add_argument('--use_uvm', action='store_true',
                       help='Enable UVM allocator')
    args = parser.parse_args()

    # Calculate number of edges
    num_edges = args.nodes * args.edges_per_node

    # Enable UVM if requested (MUST be before any CUDA operations)
    uvm_lib = None
    if args.use_uvm:
        print("=" * 70)
        print("GNN Training Benchmark with UVM")
        print("=" * 70)
        print("[Step 0] Enabling UVM allocator...")
        uvm_lib = enable_uvm_allocator()

    # Print configuration
    print("=" * 70)
    print("GNN Training Benchmark with UVM")
    print("=" * 70)
    print("Configuration:")
    print(f"  Nodes: {args.nodes:,}")
    print(f"  Edges: {num_edges:,}")
    print(f"  Features: {args.features}")
    print(f"  Hidden dim: {args.hidden}")
    print(f"  Epochs: {args.epochs}")
    print(f"  UVM: {'Yes' if args.use_uvm else 'No'}")
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
    print("=" * 70)

    # Step 1: Generate graph
    print("\n[Step 1/4] Generating random graph...")
    try:
        x, edge_index, y = generate_random_graph(
            args.nodes, num_edges, args.features
        )
        print("  ✓ Graph generated successfully")
        print_memory(uvm_lib=uvm_lib)
    except RuntimeError as e:
        if 'out of memory' in str(e).lower():
            print(f"  ✗ OOM during graph generation")
            print(f"    Error: {e}")
            print("\n" + "=" * 70)
            print("Result: FAILED (OOM during data loading)")
            print("Suggestion: Try with --use_uvm flag")
            print("=" * 70)
            return
        raise

    # Step 2: Create model
    print("\n[Step 2/4] Creating GCN model...")
    try:
        model = GCN(args.features, args.hidden, 10).cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        num_params = sum(p.numel() for p in model.parameters())
        print(f"  ✓ Model created")
        print(f"    Parameters: {num_params:,}")
        print_memory(uvm_lib=uvm_lib)
    except RuntimeError as e:
        if 'out of memory' in str(e).lower():
            print(f"  ✗ OOM during model creation")
            print(f"    Error: {e}")
            return
        raise

    # Step 3: Prepare training
    print("\n[Step 3/4] Preparing training...")
    # Use 10% of nodes for training
    num_train = args.nodes // 10
    train_mask = torch.zeros(args.nodes, dtype=torch.bool, device='cuda')
    train_mask[:num_train] = True
    print(f"  Training nodes: {num_train:,} ({num_train/args.nodes*100:.1f}%)")

    # Step 4: Training loop
    print(f"\n[Step 4/4] Training for {args.epochs} epochs...")
    epoch_times = []

    try:
        for epoch in range(args.epochs):
            start_time = time.time()

            loss = train_epoch(model, x, edge_index, y, optimizer, train_mask)

            epoch_time = time.time() - start_time
            epoch_times.append(epoch_time)

            print(f"  Epoch {epoch+1}/{args.epochs}: "
                  f"loss={loss:.4f}, time={epoch_time:.2f}s")

            if epoch == 0:  # Print memory after first epoch
                print_memory("    ", uvm_lib=uvm_lib)

        print("  ✓ Training completed successfully")

    except RuntimeError as e:
        if 'out of memory' in str(e).lower():
            print(f"  ✗ OOM during training at epoch {epoch+1}")
            print(f"    Error: {e}")
            print("\n" + "=" * 70)
            print("Result: FAILED (OOM during training)")
            print("Suggestion: Try with --use_uvm flag")
            print("=" * 70)
            return
        raise

    # Final results
    print("\n" + "=" * 70)
    print("Final Results:")
    print("=" * 70)
    print(f"  Status: ✓ SUCCESS")
    print(f"  Total epochs: {args.epochs}")
    print(f"  Avg time/epoch: {sum(epoch_times)/len(epoch_times):.2f}s")
    print(f"  Total time: {sum(epoch_times):.2f}s")

    final_mem = get_memory_stats(uvm_lib)
    print(f"\n  Memory Usage:")
    print(f"    GPU: {final_mem['gpu_allocated']:.2f}GB")
    print(f"    CPU: {final_mem['cpu_used']:.2f}GB")
    print(f"    Total: {final_mem['total']:.2f}GB")

    if args.use_uvm and uvm_lib:
        cpu_portion = final_mem['cpu_used'] - 3.0  # Subtract baseline CPU
        if cpu_portion > 1.0:
            print(f"    UVM offload: ~{cpu_portion:.1f}GB to CPU")
        print(f"\n  UVM Statistics:")
        print(f"    Allocations: {uvm_lib.uvm_get_num_allocs()}")
        print(f"    Frees: {uvm_lib.uvm_get_num_frees()}")
        print(f"    Peak allocated: {uvm_lib.uvm_get_peak_allocated_bytes() / 1e9:.2f}GB")

    print("=" * 70)


if __name__ == '__main__':
    main()
