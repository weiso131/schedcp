#!/usr/bin/env python3
"""
Analyze Memtier benchmark results from CSV and create comparison plots
Each scheduler gets a subplot comparing it with default scheduler
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
import os

def load_results(csv_file='memtier_scheduler_results.csv'):
    """Load results from CSV file"""
    if not os.path.exists(csv_file):
        print(f"Error: {csv_file} not found")
        sys.exit(1)
    
    df = pd.read_csv(csv_file)
    print(f"Loaded {len(df)} rows from {csv_file}")
    print(f"Schedulers found: {df['scheduler'].unique()}")
    return df

def create_comparison_plots(df):
    """Create subplot for each scheduler comparing with default"""
    
    # Get unique schedulers (excluding default)
    schedulers = [s for s in df['scheduler'].unique() if s != 'default']
    
    if len(schedulers) == 0:
        print("No schedulers to compare (only default found)")
        return
    
    # Get default data
    default_data = df[df['scheduler'] == 'default']
    
    # Test cases
    test_cases = ['mixed_1_10', 'mixed_10_1', 'pipeline_16', 
                  'sequential_pattern', 'gaussian_pattern', 'advanced_gaussian_random']
    test_labels = ['Test 1:\nmixed_1_10', 'Test 2:\nmixed_10_1', 'Test 3:\npipeline_16',
                   'Test 4:\nsequential', 'Test 5:\ngaussian', 'Test 6:\nadv_gaussian']
    
    # Create figure with subplots for each scheduler
    n_schedulers = len(schedulers)
    fig = plt.figure(figsize=(20, 6 * n_schedulers))
    
    for idx, scheduler in enumerate(schedulers):
        scheduler_data = df[df['scheduler'] == scheduler]
        
        # Create 3 subplots for this scheduler: throughput, latency, bandwidth
        ax1 = plt.subplot(n_schedulers, 3, idx * 3 + 1)
        ax2 = plt.subplot(n_schedulers, 3, idx * 3 + 2)
        ax3 = plt.subplot(n_schedulers, 3, idx * 3 + 3)
        
        # Prepare data for plotting
        x = np.arange(len(test_cases))
        width = 0.35
        
        # Extract metrics for each test case
        default_throughput = []
        scheduler_throughput = []
        default_latency_p99 = []
        scheduler_latency_p99 = []
        default_bandwidth = []
        scheduler_bandwidth = []
        
        for test in test_cases:
            # Default metrics
            default_test = default_data[default_data['test_case'] == test]
            if not default_test.empty:
                default_throughput.append(default_test['total_ops_per_sec'].values[0])
                default_latency_p99.append(default_test['total_p99_latency_ms'].values[0])
                default_bandwidth.append(default_test['bandwidth_kb_sec'].values[0])
            else:
                default_throughput.append(0)
                default_latency_p99.append(0)
                default_bandwidth.append(0)
            
            # Scheduler metrics
            sched_test = scheduler_data[scheduler_data['test_case'] == test]
            if not sched_test.empty:
                scheduler_throughput.append(sched_test['total_ops_per_sec'].values[0])
                scheduler_latency_p99.append(sched_test['total_p99_latency_ms'].values[0])
                scheduler_bandwidth.append(sched_test['bandwidth_kb_sec'].values[0])
            else:
                scheduler_throughput.append(0)
                scheduler_latency_p99.append(0)
                scheduler_bandwidth.append(0)
        
        # Plot throughput comparison
        bars1 = ax1.bar(x - width/2, default_throughput, width, label='Default', color='skyblue', alpha=0.8)
        bars2 = ax1.bar(x + width/2, scheduler_throughput, width, label=scheduler, color='coral', alpha=0.8)
        
        ax1.set_xlabel('Test Cases')
        ax1.set_ylabel('Throughput (ops/sec)')
        ax1.set_title(f'{scheduler} vs Default - Throughput')
        ax1.set_xticks(x)
        ax1.set_xticklabels(test_labels, rotation=0, fontsize=8)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar in bars1:
            height = bar.get_height()
            if height > 0:
                ax1.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.0f}', ha='center', va='bottom', fontsize=7)
        for bar in bars2:
            height = bar.get_height()
            if height > 0:
                ax1.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.0f}', ha='center', va='bottom', fontsize=7)
        
        # Plot P99 latency comparison
        bars3 = ax2.bar(x - width/2, default_latency_p99, width, label='Default', color='lightgreen', alpha=0.8)
        bars4 = ax2.bar(x + width/2, scheduler_latency_p99, width, label=scheduler, color='salmon', alpha=0.8)
        
        ax2.set_xlabel('Test Cases')
        ax2.set_ylabel('P99 Latency (ms)')
        ax2.set_title(f'{scheduler} vs Default - P99 Latency (Lower is Better)')
        ax2.set_xticks(x)
        ax2.set_xticklabels(test_labels, rotation=0, fontsize=8)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar in bars3:
            height = bar.get_height()
            if height > 0:
                ax2.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.2f}', ha='center', va='bottom', fontsize=7)
        for bar in bars4:
            height = bar.get_height()
            if height > 0:
                ax2.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.2f}', ha='center', va='bottom', fontsize=7)
        
        # Plot bandwidth comparison
        bars5 = ax3.bar(x - width/2, default_bandwidth, width, label='Default', color='mediumpurple', alpha=0.8)
        bars6 = ax3.bar(x + width/2, scheduler_bandwidth, width, label=scheduler, color='gold', alpha=0.8)
        
        ax3.set_xlabel('Test Cases')
        ax3.set_ylabel('Bandwidth (KB/sec)')
        ax3.set_title(f'{scheduler} vs Default - Bandwidth')
        ax3.set_xticks(x)
        ax3.set_xticklabels(test_labels, rotation=0, fontsize=8)
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar in bars5:
            height = bar.get_height()
            if height > 0:
                ax3.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.0f}', ha='center', va='bottom', fontsize=7)
        for bar in bars6:
            height = bar.get_height()
            if height > 0:
                ax3.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.0f}', ha='center', va='bottom', fontsize=7)
    
    plt.suptitle('Memtier Benchmark: Scheduler vs Default Comparison', fontsize=16, y=1.02)
    plt.tight_layout()
    
    # Save figure
    output_file = 'memtier_scheduler_comparison.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Comparison plot saved to {output_file}")
    plt.show()

def create_detailed_comparison(df):
    """Create detailed GET/SET comparison for each scheduler"""
    
    # Get unique schedulers (excluding default)
    schedulers = [s for s in df['scheduler'].unique() if s != 'default']
    
    if len(schedulers) == 0:
        print("No schedulers to compare (only default found)")
        return
    
    # Get default data
    default_data = df[df['scheduler'] == 'default']
    
    # Test cases
    test_cases = ['mixed_1_10', 'mixed_10_1', 'pipeline_16', 
                  'sequential_pattern', 'gaussian_pattern', 'advanced_gaussian_random']
    
    # Create figure for GET/SET detailed comparison
    n_schedulers = len(schedulers)
    fig = plt.figure(figsize=(20, 8 * n_schedulers))
    
    for idx, scheduler in enumerate(schedulers):
        scheduler_data = df[df['scheduler'] == scheduler]
        
        # Create 4 subplots: GET throughput, GET latency, SET throughput, SET latency
        ax1 = plt.subplot(n_schedulers, 4, idx * 4 + 1)
        ax2 = plt.subplot(n_schedulers, 4, idx * 4 + 2)
        ax3 = plt.subplot(n_schedulers, 4, idx * 4 + 3)
        ax4 = plt.subplot(n_schedulers, 4, idx * 4 + 4)
        
        x = np.arange(len(test_cases))
        width = 0.35
        
        # Extract GET/SET metrics
        default_gets_ops = []
        scheduler_gets_ops = []
        default_gets_p99 = []
        scheduler_gets_p99 = []
        default_sets_ops = []
        scheduler_sets_ops = []
        default_sets_p99 = []
        scheduler_sets_p99 = []
        
        for test in test_cases:
            # Default metrics
            default_test = default_data[default_data['test_case'] == test]
            if not default_test.empty:
                default_gets_ops.append(default_test['gets_ops_per_sec'].values[0])
                default_gets_p99.append(default_test['gets_p99_latency_ms'].values[0])
                default_sets_ops.append(default_test['sets_ops_per_sec'].values[0])
                default_sets_p99.append(default_test['sets_p99_latency_ms'].values[0])
            else:
                default_gets_ops.append(0)
                default_gets_p99.append(0)
                default_sets_ops.append(0)
                default_sets_p99.append(0)
            
            # Scheduler metrics
            sched_test = scheduler_data[scheduler_data['test_case'] == test]
            if not sched_test.empty:
                scheduler_gets_ops.append(sched_test['gets_ops_per_sec'].values[0])
                scheduler_gets_p99.append(sched_test['gets_p99_latency_ms'].values[0])
                scheduler_sets_ops.append(sched_test['sets_ops_per_sec'].values[0])
                scheduler_sets_p99.append(sched_test['sets_p99_latency_ms'].values[0])
            else:
                scheduler_gets_ops.append(0)
                scheduler_gets_p99.append(0)
                scheduler_sets_ops.append(0)
                scheduler_sets_p99.append(0)
        
        # Plot GET throughput
        ax1.bar(x - width/2, default_gets_ops, width, label='Default', color='lightblue', alpha=0.8)
        ax1.bar(x + width/2, scheduler_gets_ops, width, label=scheduler, color='darkblue', alpha=0.8)
        ax1.set_xlabel('Test Cases')
        ax1.set_ylabel('GET ops/sec')
        ax1.set_title(f'{scheduler}: GET Throughput')
        ax1.set_xticks(x)
        ax1.set_xticklabels([f'T{i+1}' for i in range(len(test_cases))])
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot GET P99 latency
        ax2.bar(x - width/2, default_gets_p99, width, label='Default', color='lightgreen', alpha=0.8)
        ax2.bar(x + width/2, scheduler_gets_p99, width, label=scheduler, color='darkgreen', alpha=0.8)
        ax2.set_xlabel('Test Cases')
        ax2.set_ylabel('GET P99 Latency (ms)')
        ax2.set_title(f'{scheduler}: GET P99 Latency')
        ax2.set_xticks(x)
        ax2.set_xticklabels([f'T{i+1}' for i in range(len(test_cases))])
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot SET throughput
        ax3.bar(x - width/2, default_sets_ops, width, label='Default', color='lightsalmon', alpha=0.8)
        ax3.bar(x + width/2, scheduler_sets_ops, width, label=scheduler, color='darkred', alpha=0.8)
        ax3.set_xlabel('Test Cases')
        ax3.set_ylabel('SET ops/sec')
        ax3.set_title(f'{scheduler}: SET Throughput')
        ax3.set_xticks(x)
        ax3.set_xticklabels([f'T{i+1}' for i in range(len(test_cases))])
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot SET P99 latency
        ax4.bar(x - width/2, default_sets_p99, width, label='Default', color='plum', alpha=0.8)
        ax4.bar(x + width/2, scheduler_sets_p99, width, label=scheduler, color='purple', alpha=0.8)
        ax4.set_xlabel('Test Cases')
        ax4.set_ylabel('SET P99 Latency (ms)')
        ax4.set_title(f'{scheduler}: SET P99 Latency')
        ax4.set_xticks(x)
        ax4.set_xticklabels([f'T{i+1}' for i in range(len(test_cases))])
        ax4.legend()
        ax4.grid(True, alpha=0.3)
    
    plt.suptitle('Detailed GET/SET Performance: Schedulers vs Default', fontsize=16, y=1.02)
    plt.tight_layout()
    
    # Save figure
    output_file = 'memtier_detailed_comparison.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Detailed comparison plot saved to {output_file}")
    plt.show()

def print_performance_summary(df):
    """Print performance summary table"""
    
    print("\n" + "="*100)
    print("PERFORMANCE SUMMARY")
    print("="*100)
    
    schedulers = df['scheduler'].unique()
    test_cases = df['test_case'].unique()
    
    for scheduler in schedulers:
        scheduler_data = df[df['scheduler'] == scheduler]
        
        print(f"\n{scheduler.upper()}")
        print("-" * 50)
        
        # Calculate averages
        avg_throughput = scheduler_data['total_ops_per_sec'].mean()
        avg_p99_latency = scheduler_data['total_p99_latency_ms'].mean()
        avg_bandwidth = scheduler_data['bandwidth_kb_sec'].mean()
        
        print(f"  Average Throughput: {avg_throughput:,.0f} ops/sec")
        print(f"  Average P99 Latency: {avg_p99_latency:.3f} ms")
        print(f"  Average Bandwidth: {avg_bandwidth:,.0f} KB/sec")
        
        # Best and worst test cases
        best_throughput = scheduler_data.loc[scheduler_data['total_ops_per_sec'].idxmax()]
        worst_throughput = scheduler_data.loc[scheduler_data['total_ops_per_sec'].idxmin()]
        
        print(f"  Best Throughput: {best_throughput['test_case']} ({best_throughput['total_ops_per_sec']:,.0f} ops/sec)")
        print(f"  Worst Throughput: {worst_throughput['test_case']} ({worst_throughput['total_ops_per_sec']:,.0f} ops/sec)")
    
    # Comparison with default
    if 'default' in schedulers:
        default_data = df[df['scheduler'] == 'default']
        default_avg_throughput = default_data['total_ops_per_sec'].mean()
        default_avg_p99 = default_data['total_p99_latency_ms'].mean()
        
        print("\n" + "="*100)
        print("COMPARISON WITH DEFAULT")
        print("="*100)
        
        for scheduler in schedulers:
            if scheduler == 'default':
                continue
            
            scheduler_data = df[df['scheduler'] == scheduler]
            avg_throughput = scheduler_data['total_ops_per_sec'].mean()
            avg_p99 = scheduler_data['total_p99_latency_ms'].mean()
            
            throughput_diff = ((avg_throughput - default_avg_throughput) / default_avg_throughput) * 100
            latency_diff = ((avg_p99 - default_avg_p99) / default_avg_p99) * 100
            
            print(f"\n{scheduler}:")
            print(f"  Throughput: {throughput_diff:+.1f}% vs default")
            print(f"  P99 Latency: {latency_diff:+.1f}% vs default (lower is better)")

def main():
    """Main function"""
    print("Memtier Benchmark Results Analysis")
    print("="*50)
    
    # Load results
    df = load_results()
    
    # Print summary
    print_performance_summary(df)
    
    # Create comparison plots
    print("\nGenerating comparison plots...")
    create_comparison_plots(df)
    
    # Create detailed GET/SET plots
    print("\nGenerating detailed GET/SET plots...")
    create_detailed_comparison(df)
    
    print("\nAnalysis complete!")

if __name__ == "__main__":
    main()