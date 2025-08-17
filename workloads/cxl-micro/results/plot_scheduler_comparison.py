#!/usr/bin/env python3

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def load_data(csv_file):
    """Load the CSV data and filter for default and scx_rustland schedulers."""
    df = pd.read_csv(csv_file)
    df = df[df['scheduler'].isin(['default', 'scx_rustland'])]
    # Rename scx_rustland to duplexOS
    df['scheduler'] = df['scheduler'].replace('scx_rustland', 'duplexOS')
    return df

def plot_combined_comparison(df):
    """Plot both comparisons as subplots in a single figure."""
    # Increase font sizes globally
    plt.rcParams.update({'font.size': 18})
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # Subplot 1: Bandwidth vs Threads (read_ratio = 0.5)
    data1 = df[df['read_ratio'] == 0.5]
    threads = sorted(data1['threads'].unique())
    
    for scheduler in ['default', 'duplexOS']:
        scheduler_data = data1[data1['scheduler'] == scheduler]
        scheduler_data = scheduler_data.sort_values('threads')
        
        ax1.plot(scheduler_data['threads'], scheduler_data['bandwidth_mbps'], 
                marker='o', linewidth=3, markersize=12, label=scheduler)
    
    ax1.set_xlabel('Number of Threads', fontsize=20)
    ax1.set_ylabel('Bandwidth (MB/s)', fontsize=20)
    ax1.set_title('Bandwidth vs Threads (Read Ratio = 0.5)', fontsize=22)
    ax1.legend(fontsize=18)
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(labelsize=16)
    ax1.set_xticks(threads)
    
    # Subplot 2: Bandwidth vs Read Ratio (threads = 128)
    data2 = df[df['threads'] == 128]
    read_ratios = sorted(data2['read_ratio'].unique())
    
    for scheduler in ['default', 'duplexOS']:
        scheduler_data = data2[data2['scheduler'] == scheduler]
        scheduler_data = scheduler_data.sort_values('read_ratio')
        
        ax2.plot(scheduler_data['read_ratio'], scheduler_data['bandwidth_mbps'], 
                marker='o', linewidth=3, markersize=12, label=scheduler)
    
    ax2.set_xlabel('Read Ratio', fontsize=20)
    ax2.set_ylabel('Bandwidth (MB/s)', fontsize=20)
    ax2.set_title('Bandwidth vs Read Ratio (Threads = 128)', fontsize=22)
    ax2.legend(fontsize=18)
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(labelsize=16)
    ax2.set_xticks(read_ratios)
    
    plt.tight_layout()
    plt.savefig('/root/yunwei37/ai-os/workloads/cxl-micro/results/cxl_scheduler_performance.pdf', 
                dpi=300, bbox_inches='tight')
    plt.show()

def main():
    # Load the data
    csv_file = '/root/yunwei37/ai-os/workloads/cxl-micro/results/parameter_sweep_multi_schedulers.csv'
    df = load_data(csv_file)
    
    print("Loaded data for schedulers:", df['scheduler'].unique())
    print("Thread counts available:", sorted(df['threads'].unique()))
    print("Read ratios available:", sorted(df['read_ratio'].unique()))
    
    # Generate combined plot
    plot_combined_comparison(df)
    
    print("Combined plot saved as:")
    print("- cxl_scheduler_performance.pdf")

if __name__ == "__main__":
    main()