#!/usr/bin/env python3
"""
Plot Total Bandwidth vs Read Ratio for different NUMA interleave configurations
Compares performance across numactl interleave=0, 2, and 3
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Set style for better-looking plots
plt.style.use('seaborn-v0_8-darkgrid')

def load_and_filter_data(csv_file, buffer_size_gb=64.0):
    """Load CSV and filter for specific buffer size"""
    df = pd.read_csv(csv_file)
    # Filter for 64GB buffer size
    df_filtered = df[df['buffer_size_gb'] == buffer_size_gb].copy()
    return df_filtered

def plot_bandwidth_vs_datasize():
    """Create comparison plots for different data sizes with fixed thread count"""
    
    # Define file paths and labels
    numa_configs = [
        {'file': 'cxl_perf_parameter_sweep_numactl0.csv', 'label': 'NUMA Node 0 (DDR5)', 'interleave': '0'},
        {'file': 'cxl_perf_parameter_sweep_numactl2.csv', 'label': 'NUMA Node 2 (CXL 256GB)', 'interleave': '2'},
        {'file': 'cxl_perf_parameter_sweep_numactl3.csv', 'label': 'NUMA Node 3 (CXL 512GB)', 'interleave': '3'}
    ]
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle('Total Bandwidth vs Read Ratio (Thread Count = 172)', fontsize=16, fontweight='bold')
    
    # Color palette for different data sizes
    colors = plt.cm.viridis(np.linspace(0, 1, 10))
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']
    
    for idx, config in enumerate(numa_configs):
        ax = axes[idx]
        
        # Check if file exists
        if not os.path.exists(config['file']):
            print(f"Warning: File {config['file']} not found, skipping...")
            ax.text(0.5, 0.5, f"Data file not found:\n{config['file']}", 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f"numactl interleave={config['interleave']}")
            continue
        
        # Load all data
        df = pd.read_csv(config['file'])
        # Filter for thread count = 172
        df_filtered = df[df['threads'] == 172].copy()
        
        if df_filtered.empty:
            print(f"Warning: No data for threads=172 in {config['file']}")
            ax.text(0.5, 0.5, f"No data for 172 threads", 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f"numactl interleave={config['interleave']}")
            continue
        
        # Get unique data sizes (1 to 64 GB)
        data_sizes = sorted(df_filtered['buffer_size_gb'].unique())
        data_sizes = [size for size in data_sizes if 1 <= size <= 64]
        
        # Plot each data size as a separate line
        for i, size in enumerate(data_sizes):
            df_size = df_filtered[df_filtered['buffer_size_gb'] == size].sort_values('read_ratio')
            
            # Use app_total_bandwidth_mbps if available, otherwise fall back
            bandwidth_col = 'app_total_bandwidth_mbps'
            if bandwidth_col not in df_size.columns:
                # Try alternative column names
                if 'bandwidth_mbps' in df_size.columns:
                    bandwidth_col = 'bandwidth_mbps'
                elif 'total_bandwidth_mbps' in df_size.columns:
                    bandwidth_col = 'total_bandwidth_mbps'
                else:
                    print(f"Warning: No bandwidth column found in {config['file']}")
                    continue
            
            ax.plot(df_size['read_ratio'], 
                   df_size[bandwidth_col],
                   marker=markers[i % len(markers)],
                   color=colors[i % len(colors)],
                   linewidth=2,
                   markersize=8,
                   label=f'{size:.0f} GB',
                   alpha=0.8)
        
        # Customize subplot
        ax.set_xlabel('Read Ratio', fontsize=12)
        ax.set_ylabel('Total Bandwidth (MB/s)', fontsize=12)
        ax.set_title(f"{config['label']}\n(numactl interleave={config['interleave']})", fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=10, title='Buffer Size')
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(bottom=0)
        
        # Add minor gridlines
        ax.minorticks_on()
        ax.grid(which='minor', alpha=0.1)
    
    plt.tight_layout()
    
    # Save figure
    output_file = 'bandwidth_vs_read_ratio_datasize_comparison.pdf'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Figure saved as {output_file}")

def plot_bandwidth_comparison():
    """Create comparison plots for different NUMA configurations"""
    
    # Define file paths and labels
    numa_configs = [
        {'file': 'cxl_perf_parameter_sweep_numactl0.csv', 'label': 'NUMA Node 0 (DDR5)', 'interleave': '0'},
        {'file': 'cxl_perf_parameter_sweep_numactl2.csv', 'label': 'NUMA Node 2 (CXL 256GB)', 'interleave': '2'},
        {'file': 'cxl_perf_parameter_sweep_numactl3.csv', 'label': 'NUMA Node 3 (CXL 512GB)', 'interleave': '3'}
    ]
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle('Total Bandwidth vs Read Ratio (Buffer Size = 64GB)', fontsize=16, fontweight='bold')
    
    # Color palette for different thread counts
    colors = plt.cm.tab10(np.linspace(0, 0.8, 10))
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']
    
    for idx, config in enumerate(numa_configs):
        ax = axes[idx]
        
        # Check if file exists
        if not os.path.exists(config['file']):
            print(f"Warning: File {config['file']} not found, skipping...")
            ax.text(0.5, 0.5, f"Data file not found:\n{config['file']}", 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f"numactl interleave={config['interleave']}")
            continue
        
        # Load data
        df = load_and_filter_data(config['file'])
        
        if df.empty:
            print(f"Warning: No data for buffer_size_gb=64.0 in {config['file']}")
            ax.text(0.5, 0.5, f"No data for 64GB buffer", 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f"numactl interleave={config['interleave']}")
            continue
        
        # Get unique thread counts
        thread_counts = sorted(df['threads'].unique())
        
        # Plot each thread count as a separate line
        for i, threads in enumerate(thread_counts):
            df_thread = df[df['threads'] == threads].sort_values('read_ratio')
            
            # Use app_total_bandwidth_mbps if available, otherwise fall back
            bandwidth_col = 'app_total_bandwidth_mbps'
            if bandwidth_col not in df_thread.columns:
                # Try alternative column names
                if 'bandwidth_mbps' in df_thread.columns:
                    bandwidth_col = 'bandwidth_mbps'
                elif 'total_bandwidth_mbps' in df_thread.columns:
                    bandwidth_col = 'total_bandwidth_mbps'
                else:
                    print(f"Warning: No bandwidth column found in {config['file']}")
                    continue
            
            ax.plot(df_thread['read_ratio'], 
                   df_thread[bandwidth_col],
                   marker=markers[i % len(markers)],
                   color=colors[i % len(colors)],
                   linewidth=2,
                   markersize=8,
                   label=f'{threads} threads',
                   alpha=0.8)
        
        # Customize subplot
        ax.set_xlabel('Read Ratio', fontsize=12)
        ax.set_ylabel('Total Bandwidth (MB/s)', fontsize=12)
        ax.set_title(f"{config['label']}\n(numactl interleave={config['interleave']})", fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=10)
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(bottom=0)
        
        # Add minor gridlines
        ax.minorticks_on()
        ax.grid(which='minor', alpha=0.1)
    
    plt.tight_layout()
    
    # Save figure
    output_file = 'bandwidth_vs_read_ratio_comparison.pdf'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Figure saved as {output_file}")
    
    # Also create a combined plot with all configurations on one graph
    create_combined_plot(numa_configs)
    
    plt.show()

def create_combined_plot(numa_configs):
    """Create a single plot with all NUMA configurations overlaid"""
    
    fig, ax = plt.subplots(figsize=(12, 8))
    fig.suptitle('Total Bandwidth Comparison Across NUMA Configurations\n(Buffer Size = 64GB, Threads = 172)', 
                 fontsize=16, fontweight='bold')
    
    # Use different line styles for different NUMA configs
    line_styles = ['-', '--', '-.']
    # Use different colors for each config
    colors = ['blue', 'red', 'green']
    markers = ['o', 's', '^']
    
    for config_idx, config in enumerate(numa_configs):
        if not os.path.exists(config['file']):
            continue
        
        df = load_and_filter_data(config['file'])
        if df.empty:
            continue
        
        # Filter for thread count = 172 only
        df_thread = df[df['threads'] == 172].sort_values('read_ratio')
        
        if df_thread.empty:
            print(f"Warning: No data for 172 threads in {config['file']}")
            continue
        
        # Determine bandwidth column
        bandwidth_col = 'app_total_bandwidth_mbps'
        if bandwidth_col not in df_thread.columns:
            if 'bandwidth_mbps' in df_thread.columns:
                bandwidth_col = 'bandwidth_mbps'
            elif 'total_bandwidth_mbps' in df_thread.columns:
                bandwidth_col = 'total_bandwidth_mbps'
            else:
                continue
        
        label = f"Node {config['interleave']} ({config['label'].split('(')[1].split(')')[0]})"
        ax.plot(df_thread['read_ratio'], 
               df_thread[bandwidth_col],
               linestyle=line_styles[config_idx % len(line_styles)],
               marker=markers[config_idx % len(markers)],
               color=colors[config_idx % len(colors)],
               linewidth=2,
               markersize=8,
               label=label,
               alpha=0.8)
    
    ax.set_xlabel('Read Ratio', fontsize=14)
    ax.set_ylabel('Total Bandwidth (MB/s)', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.minorticks_on()
    ax.grid(which='minor', alpha=0.1)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(bottom=0)
    
    # Position legend outside the plot
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10, ncol=1)
    
    plt.tight_layout()
    
    # Save combined figure
    output_file = 'bandwidth_vs_read_ratio_combined.pdf'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Combined figure saved as {output_file}")

def print_summary_statistics():
    """Print summary statistics for each configuration"""
    
    numa_configs = [
        {'file': 'cxl_perf_parameter_sweep_numactl0.csv', 'label': 'NUMA Node 0 (DDR5)', 'interleave': '0'},
        {'file': 'cxl_perf_parameter_sweep_numactl2.csv', 'label': 'NUMA Node 2 (CXL 256GB)', 'interleave': '2'},
        {'file': 'cxl_perf_parameter_sweep_numactl3.csv', 'label': 'NUMA Node 3 (CXL 512GB)', 'interleave': '3'}
    ]
    
    print("\n" + "="*80)
    print("SUMMARY STATISTICS (Buffer Size = 64GB)")
    print("="*80)
    
    for config in numa_configs:
        if not os.path.exists(config['file']):
            continue
        
        df = load_and_filter_data(config['file'])
        if df.empty:
            continue
        
        print(f"\n{config['label']} (interleave={config['interleave']})")
        print("-" * 50)
        
        # Use appropriate bandwidth column
        bandwidth_col = 'app_total_bandwidth_mbps'
        if bandwidth_col not in df.columns:
            bandwidth_col = 'bandwidth_mbps' if 'bandwidth_mbps' in df.columns else 'total_bandwidth_mbps'
        
        if bandwidth_col in df.columns:
            print(f"  Average Bandwidth: {df[bandwidth_col].mean():.2f} MB/s")
            print(f"  Max Bandwidth: {df[bandwidth_col].max():.2f} MB/s")
            print(f"  Min Bandwidth: {df[bandwidth_col].min():.2f} MB/s")
            print(f"  Std Dev: {df[bandwidth_col].std():.2f} MB/s")
            
            # Best configuration
            best_row = df.loc[df[bandwidth_col].idxmax()]
            print(f"  Best Config: {best_row['threads']:.0f} threads, "
                  f"read_ratio={best_row['read_ratio']:.2f}, "
                  f"bandwidth={best_row[bandwidth_col]:.2f} MB/s")

if __name__ == "__main__":
    # Change to the numa_results directory
    os.chdir('/root/yunwei37/ai-os/workloads/cxl-micro/numa_results')
    
    # Print summary statistics
    print_summary_statistics()
    
    # Create plots with fixed buffer size (64GB) and varying thread counts
    plot_bandwidth_comparison()
    
    # Create plots with fixed thread count (172) and varying data sizes
    plot_bandwidth_vs_datasize()
    
    print("\nAll plots have been generated successfully!")