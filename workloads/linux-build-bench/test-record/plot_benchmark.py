#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np
import re

# Parse the data from README.md
def parse_benchmark_data(filename):
    with open(filename, 'r') as f:
        content = f.read()
    
    results = {}
    current_scheduler = None
    current_times = []
    
    lines = content.strip().split('\n')
    for line in lines:
        # Match scheduler names
        if 'default (baseline)' in line:
            if current_scheduler:
                results[current_scheduler] = current_times
            current_scheduler = 'default Linux CFS'
            current_times = []
        elif 'scx_rusty' in line:
            if current_scheduler:
                results[current_scheduler] = current_times
            current_scheduler = 'first attempt (scx_rusty)'
            current_times = []
        elif 'scx_layered' in line:
            if current_scheduler:
                results[current_scheduler] = current_times
            current_scheduler = 'Iter 3 times (scx_layered)'
            current_times = []
        # Parse real time values
        elif line.startswith('real'):
            match = re.search(r'(\d+)m([\d.]+)s', line)
            if match:
                minutes = int(match.group(1))
                seconds = float(match.group(2))
                total_seconds = minutes * 60 + seconds
                current_times.append(total_seconds)
    
    # Add the last scheduler
    if current_scheduler:
        results[current_scheduler] = current_times
    
    return results

# Calculate averages
def calculate_averages(results):
    averages = {}
    for scheduler, times in results.items():
        if times:
            averages[scheduler] = np.mean(times)
    return averages

# Plot the data
def create_bar_chart(averages):
    schedulers = list(averages.keys())
    times = list(averages.values())
    
    # Create figure and axis with less height
    fig, ax = plt.subplots(figsize=(14, 5))
    
    # Create horizontal bars
    y = np.arange(len(schedulers))
    bars = ax.barh(y, times, color=['#1f77b4', '#ff7f0e', '#2ca02c'], height=0.5)
    
    # Customize the plot with even bigger text, no title
    ax.set_ylabel('Scheduler', fontsize=24, fontweight='bold')
    ax.set_xlabel('Average Build Time (seconds)', fontsize=24, fontweight='bold')
    ax.set_yticks(y)
    ax.set_yticklabels(schedulers, fontsize=22)
    ax.tick_params(axis='x', labelsize=20)
    
    # Add value labels on bars with proper spacing
    for bar, time in zip(bars, times):
        width = bar.get_width()
        ax.text(width + 0.1, bar.get_y() + bar.get_height()/2.,
                f'{time:.2f}s', ha='left', va='center', fontsize=20, fontweight='bold')
    
    # No grid for cleaner look
    ax.grid(False)
    
    # Add speedup relative to baseline with better positioning
    baseline_time = averages['default Linux CFS']
    for i, (scheduler, time) in enumerate(averages.items()):
        if scheduler != 'default Linux CFS':
            speedup = baseline_time / time
            # Position text further right to avoid overlap
            ax.text(time + 2, i, f'{speedup:.2f}×', 
                   ha='left', va='center', fontsize=20, color='green', fontweight='bold')
    
    # Adjust x-axis limits to accommodate all text
    ax.set_xlim(0, max(times) + 4)
    
    plt.tight_layout()
    plt.savefig('benchmark_results.pdf', format='pdf', dpi=300, bbox_inches='tight')
    plt.savefig('benchmark_results.png', dpi=300, bbox_inches='tight')
    plt.show()

# Main execution
if __name__ == "__main__":
    results = parse_benchmark_data('README.md')
    averages = calculate_averages(results)
    
    print("Benchmark Results Summary:")
    print("-" * 40)
    for scheduler, avg_time in averages.items():
        print(f"{scheduler}: {avg_time:.3f} seconds (average)")
    
    create_bar_chart(averages)