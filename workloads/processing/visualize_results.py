#!/usr/bin/env python3
"""
Visualization script to create figures comparing scheduler performance.
Reads the JSON results from scheduler tests and creates comparative charts.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import argparse
import sys

class SchedulerVisualizer:
    def __init__(self):
        self.results = {}
        self.scheduler_names = []
        
    def load_results(self, result_files):
        """Load results from JSON files."""
        for file in result_files:
            if not Path(file).exists():
                print(f"Warning: {file} not found, skipping...")
                continue
                
            with open(file, 'r') as f:
                data = json.load(f)
                
            # If it's a combined result file
            if isinstance(data, dict) and all(isinstance(v, dict) for v in data.values()):
                self.results.update(data)
            # If it's a single scheduler result
            else:
                # Extract scheduler name from filename
                scheduler_name = Path(file).stem.replace('results_', '')
                self.results[scheduler_name] = data
        
        self.scheduler_names = list(self.results.keys())
        print(f"Loaded results for schedulers: {', '.join(self.scheduler_names)}")
    
    def create_comparison_figure(self, output_file="scheduler_comparison.png"):
        """Create a comprehensive comparison figure."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Scheduler Performance Comparison', fontsize=16, fontweight='bold')
        
        # 1. Total execution time comparison
        self.plot_total_time(ax1)
        
        # 2. Improvement percentage
        self.plot_improvement(ax2)
        
        # 3. Load imbalance ratio
        self.plot_imbalance_ratio(ax3)
        
        # 4. Task timing distribution
        self.plot_task_distribution(ax4)
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {output_file}")
        
    def plot_total_time(self, ax):
        """Plot total execution time for each scheduler."""
        schedulers = []
        times = []
        colors = []
        
        color_map = {
            'default': '#808080',  # Gray for default
            'fifo': '#3498db',     # Blue for FIFO
            'ctest': '#e74c3c',    # Red for ctest (priority)
            'vruntime': '#2ecc71'  # Green for vruntime
        }
        
        for name in self.scheduler_names:
            if 'total_time' in self.results[name]:
                schedulers.append(name.upper())
                times.append(self.results[name]['total_time'])
                colors.append(color_map.get(name, '#95a5a6'))
        
        bars = ax.bar(schedulers, times, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
        
        # Add value labels on bars
        for bar, time in zip(bars, times):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{time:.2f}s', ha='center', va='bottom', fontweight='bold')
        
        ax.set_ylabel('Total Execution Time (seconds)', fontsize=12, fontweight='bold')
        ax.set_title('Total Workload Execution Time', fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        
        # Highlight the best performer
        if times:
            min_time = min(times)
            min_idx = times.index(min_time)
            bars[min_idx].set_linewidth(3)
            bars[min_idx].set_edgecolor('gold')
    
    def plot_improvement(self, ax):
        """Plot improvement percentage compared to default scheduler."""
        if 'default' not in self.results or 'total_time' not in self.results['default']:
            ax.text(0.5, 0.5, 'No default scheduler data', ha='center', va='center')
            ax.set_title('Improvement vs Default Scheduler')
            return
        
        default_time = self.results['default']['total_time']
        schedulers = []
        improvements = []
        colors = []
        
        color_positive = '#2ecc71'  # Green for improvement
        color_negative = '#e74c3c'  # Red for degradation
        
        for name in self.scheduler_names:
            if name != 'default' and 'total_time' in self.results[name]:
                schedulers.append(name.upper())
                time = self.results[name]['total_time']
                improvement = ((default_time - time) / default_time) * 100
                improvements.append(improvement)
                colors.append(color_positive if improvement > 0 else color_negative)
        
        bars = ax.bar(schedulers, improvements, color=colors, alpha=0.8, 
                      edgecolor='black', linewidth=1.5)
        
        # Add value labels
        for bar, imp in zip(bars, improvements):
            height = bar.get_height()
            label_y = height + 0.5 if height > 0 else height - 0.5
            ax.text(bar.get_x() + bar.get_width()/2., label_y,
                   f'{imp:.1f}%', ha='center', va='bottom' if height > 0 else 'top',
                   fontweight='bold')
        
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax.set_ylabel('Improvement (%)', fontsize=12, fontweight='bold')
        ax.set_title('Performance Improvement vs Default CFS', fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    def plot_imbalance_ratio(self, ax):
        """Plot load imbalance ratio for each scheduler."""
        schedulers = []
        ratios = []
        
        for name in self.scheduler_names:
            if 'parallel_execution' in self.results[name]:
                parallel_data = self.results[name]['parallel_execution']
                if 'imbalance_ratio' in parallel_data:
                    schedulers.append(name.upper())
                    ratios.append(parallel_data['imbalance_ratio'])
        
        if not ratios:
            ax.text(0.5, 0.5, 'No imbalance ratio data', ha='center', va='center')
            ax.set_title('Load Imbalance Ratio')
            return
        
        # Color based on imbalance severity
        colors = []
        for ratio in ratios:
            if ratio < 5:
                colors.append('#2ecc71')  # Green - good balance
            elif ratio < 20:
                colors.append('#f39c12')  # Orange - moderate imbalance
            else:
                colors.append('#e74c3c')  # Red - high imbalance
        
        bars = ax.bar(schedulers, ratios, color=colors, alpha=0.8, 
                      edgecolor='black', linewidth=1.5)
        
        # Add value labels
        for bar, ratio in zip(bars, ratios):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{ratio:.1f}', ha='center', va='bottom', fontweight='bold')
        
        ax.set_ylabel('Imbalance Ratio', fontsize=12, fontweight='bold')
        ax.set_title('Workload Imbalance Ratio\n(Lower is Better)', fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    def plot_task_distribution(self, ax):
        """Plot task execution time distribution."""
        data_to_plot = []
        labels = []
        
        for name in self.scheduler_names:
            if 'parallel_execution' in self.results[name]:
                parallel_data = self.results[name]['parallel_execution']
                
                task_times = []
                
                # Collect short task times
                if 'short_tasks' in parallel_data and 'times' in parallel_data['short_tasks']:
                    task_times.extend(parallel_data['short_tasks']['times'])
                
                # Collect long task times
                if 'long_tasks' in parallel_data and 'times' in parallel_data['long_tasks']:
                    task_times.extend(parallel_data['long_tasks']['times'])
                
                if task_times:
                    data_to_plot.append(task_times)
                    labels.append(name.upper())
        
        if not data_to_plot:
            ax.text(0.5, 0.5, 'No task timing data', ha='center', va='center')
            ax.set_title('Task Execution Time Distribution')
            return
        
        # Create box plot
        bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True, notch=True)
        
        # Customize box plot colors
        colors = ['#3498db', '#e74c3c', '#808080', '#2ecc71']
        for patch, color in zip(bp['boxes'], colors[:len(bp['boxes'])]):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        # Customize other elements
        for element in ['whiskers', 'fliers', 'medians', 'caps']:
            plt.setp(bp[element], color='black')
        
        ax.set_ylabel('Task Execution Time (seconds)', fontsize=12, fontweight='bold')
        ax.set_title('Task Execution Time Distribution', fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    def create_detailed_report(self, output_file="scheduler_report.txt"):
        """Create a detailed text report of the results."""
        with open(output_file, 'w') as f:
            f.write("="*80 + "\n")
            f.write("SCHEDULER PERFORMANCE COMPARISON REPORT\n")
            f.write("="*80 + "\n\n")
            
            # Summary table
            f.write("SUMMARY\n")
            f.write("-"*40 + "\n")
            f.write(f"{'Scheduler':<15} {'Total Time':<15} {'Improvement':<15} {'Status':<10}\n")
            f.write("-"*40 + "\n")
            
            default_time = None
            if 'default' in self.results and 'total_time' in self.results['default']:
                default_time = self.results['default']['total_time']
            
            for name in self.scheduler_names:
                if 'total_time' in self.results[name]:
                    time = self.results[name]['total_time']
                    status = self.results[name].get('status', 'unknown')
                    
                    if default_time and name != 'default':
                        improvement = f"{((default_time - time) / default_time * 100):.1f}%"
                    else:
                        improvement = "-"
                    
                    f.write(f"{name:<15} {time:<15.2f} {improvement:<15} {status:<10}\n")
            
            f.write("\n")
            
            # Detailed results for each scheduler
            for name in self.scheduler_names:
                f.write("="*60 + "\n")
                f.write(f"SCHEDULER: {name.upper()}\n")
                f.write("="*60 + "\n")
                
                result = self.results[name]
                
                # Basic info
                f.write(f"Test ID: {result.get('test_id', 'N/A')}\n")
                f.write(f"Test Name: {result.get('test_name', 'N/A')}\n")
                f.write(f"Status: {result.get('status', 'N/A')}\n")
                f.write(f"Total Time: {result.get('total_time', 'N/A'):.2f} seconds\n")
                
                # Parallel execution details
                if 'parallel_execution' in result:
                    pe = result['parallel_execution']
                    f.write("\nParallel Execution Details:\n")
                    f.write(f"  Execution Mode: {pe.get('execution_mode', 'N/A')}\n")
                    f.write(f"  Wall Clock Time: {pe.get('wall_clock_time', 'N/A'):.2f} seconds\n")
                    f.write(f"  Imbalance Ratio: {pe.get('imbalance_ratio', 'N/A'):.2f}\n")
                    
                    # Short tasks
                    if 'short_tasks' in pe:
                        st = pe['short_tasks']
                        f.write(f"\n  Short Tasks:\n")
                        f.write(f"    Count: {st.get('count', 'N/A')}\n")
                        f.write(f"    Average Time: {st.get('avg_time', 'N/A'):.4f} seconds\n")
                        f.write(f"    Min Time: {st.get('min_time', 'N/A'):.4f} seconds\n")
                        f.write(f"    Max Time: {st.get('max_time', 'N/A'):.4f} seconds\n")
                    
                    # Long tasks
                    if 'long_tasks' in pe:
                        lt = pe['long_tasks']
                        f.write(f"\n  Long Tasks:\n")
                        f.write(f"    Count: {lt.get('count', 'N/A')}\n")
                        f.write(f"    Average Time: {lt.get('avg_time', 'N/A'):.4f} seconds\n")
                
                f.write("\n")
        
        print(f"Detailed report saved to {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Visualize scheduler test results")
    parser.add_argument("--input", nargs='+', 
                       default=["results_ctest.json", "results_fifo.json", "results_default.json",
                               "scheduler_comparison_results.json"],
                       help="Input JSON files (default: all result files)")
    parser.add_argument("--output", default="scheduler_comparison.png",
                       help="Output figure file (default: scheduler_comparison.png)")
    parser.add_argument("--report", action='store_true',
                       help="Generate detailed text report")
    args = parser.parse_args()
    
    visualizer = SchedulerVisualizer()
    
    # Load results
    visualizer.load_results(args.input)
    
    if not visualizer.results:
        print("Error: No valid results loaded")
        sys.exit(1)
    
    # Create visualization
    visualizer.create_comparison_figure(args.output)
    
    # Create text report if requested
    if args.report:
        visualizer.create_detailed_report()

if __name__ == "__main__":
    main()