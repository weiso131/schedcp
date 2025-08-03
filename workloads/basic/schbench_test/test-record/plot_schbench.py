#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np
import json

def load_json_data(filename):
    with open(filename, 'r') as f:
        return json.load(f)

def parse_scheduler_name(filename):
    if 'default.json' in filename:
        return 'Default Linux'
    elif 'scx_bpfland_aggressive.json' in filename:
        return 'scx_bpfland (aggressive)'
    elif 'scx_rusty_lowlat.json' in filename:
        return 'scx_rusty (lowlat)'
    return filename

def create_grouped_bar_chart():
    json_files = [
        'default.json',
        'scx_bpfland_aggressive.json',
        'scx_rusty_lowlat.json'
    ]
    
    schedulers = []
    throughputs = []
    p99_latencies = []
    
    for json_file in json_files:
        data = load_json_data(json_file)
        scheduler_name = parse_scheduler_name(json_file)
        schedulers.append(scheduler_name)
        throughputs.append(data['averages']['throughput'])
        p99_latencies.append(data['averages']['99th_percentile_us'])  # Keep in microseconds
    
    # Create figure with less height (matching Linux build benchmark style)
    fig, ax = plt.subplots(figsize=(16, 4))
    
    # Create horizontal bars for throughput
    y = np.arange(len(schedulers))
    width = 0.35
    
    # Use the same colors as in Linux build benchmark
    bars1 = ax.barh(y - width/2, throughputs, width, label='Throughput (req/s)', 
                    color='#1f77b4', alpha=0.8)
    bars2 = ax.barh(y + width/2, [lat/1000 for lat in p99_latencies], width, 
                    label='P99 Latency (ms)', color='#ff7f0e', alpha=0.8)
    
    # Customize the plot with even bigger text, no title
    ax.set_ylabel('Scheduler', fontsize=26, fontweight='bold')
    ax.set_xlabel('Throughput (req/s) / P99 Latency (ms)', fontsize=26, fontweight='bold')
    ax.set_yticks(y)
    ax.set_yticklabels(schedulers, fontsize=24)
    ax.tick_params(axis='x', labelsize=22)
    
    # Add value labels on bars with proper spacing
    for bar, value in zip(bars1, throughputs):
        width = bar.get_width()
        ax.text(width + 10, bar.get_y() + bar.get_height()/2.,
                f'{value:.0f} req/s', ha='left', va='center', fontsize=24, fontweight='bold')
    
    for bar, value in zip(bars2, p99_latencies):
        width = bar.get_width()
        ax.text(width + 0.5, bar.get_y() + bar.get_height()/2.,
                f'{value/1000:.1f} ms', ha='left', va='center', fontsize=24, fontweight='bold')
    
    # No grid for cleaner look
    ax.grid(False)
    
    # # Add throughput improvement relative to baseline
    # baseline_throughput = throughputs[0]
    # baseline_latency = p99_latencies[0]
    # for i, (throughput, latency) in enumerate(zip(throughputs, p99_latencies)):
    #     if i != 0:  # Skip baseline
    #         throughput_improvement = throughput / baseline_throughput
    #         latency_improvement = baseline_latency / latency
    #         # Position text further right to avoid overlap
    #         ax.text(max(throughput, latency/1000) + 100, i - width/2, f'{throughput_improvement:.2f}×', 
    #                ha='left', va='center', fontsize=24, color='green', fontweight='bold')
    #         ax.text(max(throughput, latency/1000) + 100, i + width/2, f'{latency_improvement:.2f}×', 
    #                ha='left', va='center', fontsize=24, color='green', fontweight='bold')
    
    # Legend
    ax.legend(loc='lower right', fontsize=20, framealpha=0.9)
    
    # Adjust x-axis limits to accommodate all text
    ax.set_xlim(0, max(max(throughputs), max([lat/1000 for lat in p99_latencies])) * 1.3)
    
    plt.tight_layout()
    plt.savefig('schbench_performance_comparison.pdf', format='pdf', dpi=300, bbox_inches='tight')
    plt.savefig('schbench_performance_comparison.png', dpi=300, bbox_inches='tight')

if __name__ == "__main__":
    print("Schbench Performance Summary:")
    print("-" * 60)
    
    json_files = [
        'default.json',
        'scx_bpfland_aggressive.json',
        'scx_rusty_lowlat.json'
    ]
    
    for json_file in json_files:
        data = load_json_data(json_file)
        scheduler_name = parse_scheduler_name(json_file)
        throughput = data['averages']['throughput']
        p99_latency = data['averages']['99th_percentile_us'] / 1000
        print(f"{scheduler_name:25} - Throughput: {throughput:7.1f} req/s, P99 Latency: {p99_latency:5.1f} ms")
    
    create_grouped_bar_chart()