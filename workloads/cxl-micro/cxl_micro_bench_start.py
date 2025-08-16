#!/usr/bin/env python3
"""
CXL-Micro Scheduler Testing Script
Tests different schedulers with double_bandwidth to compare memory bandwidth performance.
"""

import os
import sys
import subprocess
import json
import time
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import argparse
import re

# Add the scheduler module to the path
sys.path.insert(0, '/root/yunwei37/ai-os/')

from scheduler import SchedulerRunner, SchedulerBenchmark


class CXLMicroBenchmarkTester(SchedulerBenchmark):
    """
    Specialized class for testing schedulers with CXL-micro benchmarks.
    
    This class extends SchedulerBenchmark to provide CXL memory bandwidth-specific
    functionality including performance testing and result visualization.
    """
    
    def __init__(self, double_bandwidth_path: str, results_dir: str = "results",
                 scheduler_runner: SchedulerRunner = None):
        """
        Initialize the CXLMicroBenchmarkTester.
        
        Args:
            double_bandwidth_path: Path to double_bandwidth binary
            results_dir: Directory to store results
            scheduler_runner: SchedulerRunner instance to use
        """
        super().__init__(scheduler_runner)
        
        self.double_bandwidth_path = double_bandwidth_path
        self.results_dir = results_dir
        
        # Create results directory
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Default test parameters
        self.test_params = {
            "array_size": 1073741824,  # 1GB default
            "iterations": 10,
            "threads": 8,
            "timeout": 300,
            "read_ratio": 0.5,  # 50% readers, 50% writers
        }
        
        # Environment setup
        self.env = os.environ.copy()
    
    def set_test_params(self, **kwargs):
        """
        Update test parameters.
        
        Args:
            **kwargs: Test parameters to update
        """
        self.test_params.update(kwargs)
    
    def _build_double_bandwidth_command(self) -> list:
        """Build double_bandwidth command with current parameters"""
        cmd = [
            self.double_bandwidth_path,
            "--buffer-size", str(self.test_params["array_size"]),
            "--threads", str(self.test_params["threads"]),
            "--duration", "10",  # Use fixed duration for consistent testing
            "--read-ratio", str(self.test_params["read_ratio"])
        ]
        return cmd
    
    def _parse_bandwidth_output(self, output: str) -> dict:
        """Parse double_bandwidth output to extract performance metrics"""
        try:
            metrics = {}
            
            # Parse output for bandwidth measurements
            # Looking for patterns like:
            # "Bandwidth: X MB/s"
            # "Time: Y seconds"
            # "Throughput: Z GB/s"
            
            bandwidth_pattern = r"Total Bandwidth[:\s]+(\d+\.?\d*)\s*(MB/s|GB/s)"
            time_pattern = r"Time[:\s]+(\d+\.?\d*)\s*(s|seconds|ms)"
            throughput_pattern = r"Throughput[:\s]+(\d+\.?\d*)\s*(MB/s|GB/s)"
            
            # Find bandwidth
            bandwidth_match = re.search(bandwidth_pattern, output, re.IGNORECASE)
            if bandwidth_match:
                value = float(bandwidth_match.group(1))
                unit = bandwidth_match.group(2).upper()
                # Convert to MB/s for consistency
                if "GB" in unit:
                    value *= 1024
                metrics["bandwidth_mbps"] = value
            
            # Find time
            time_match = re.search(time_pattern, output, re.IGNORECASE)
            if time_match:
                value = float(time_match.group(1))
                unit = time_match.group(2).lower()
                # Convert to seconds
                if "ms" in unit:
                    value /= 1000
                metrics["execution_time"] = value
            
            # Find throughput
            throughput_match = re.search(throughput_pattern, output, re.IGNORECASE)
            if throughput_match:
                value = float(throughput_match.group(1))
                unit = throughput_match.group(2).upper()
                # Convert to MB/s for consistency
                if "GB" in unit:
                    value *= 1024
                metrics["throughput_mbps"] = value
            
            # If we couldn't parse specific metrics, try to extract any numbers
            if not metrics:
                # Look for any numeric values in the output
                numbers = re.findall(r'\d+\.?\d*', output)
                if numbers:
                    # Assume the largest number is bandwidth
                    metrics["bandwidth_mbps"] = max(float(n) for n in numbers)
                    metrics["parse_warning"] = "Generic number parsing used"
            
            metrics["raw_output"] = output
            return metrics
            
        except Exception as e:
            return {"error": str(e), "raw_output": output}
    
    def run_cxl_benchmark(self, scheduler_name: str = None) -> dict:
        """
        Run double_bandwidth with specified scheduler.
        
        Args:
            scheduler_name: Name of the scheduler to test (None for default)
            
        Returns:
            Dictionary containing benchmark results
        """
        print(f"Running double_bandwidth with scheduler: {scheduler_name or 'default'}")
        print(f"Parameters: buffer_size={self.test_params['array_size']}, "
              f"threads={self.test_params['threads']}, "
              f"read_ratio={self.test_params['read_ratio']}")
        
        # Build command
        cmd = self._build_double_bandwidth_command()
        timeout = self.test_params["timeout"]
        
        try:
            if scheduler_name:
                # Run with specific scheduler
                exit_code, stdout, stderr = self.runner.run_command_with_scheduler(
                    scheduler_name, cmd, timeout=timeout, env=self.env
                )
            else:
                # Run with default scheduler
                try:
                    result = subprocess.run(
                        cmd, 
                        capture_output=True, 
                        text=True, 
                        timeout=timeout,
                        env=self.env
                    )
                    exit_code, stdout, stderr = result.returncode, result.stdout, result.stderr
                except subprocess.TimeoutExpired:
                    exit_code, stdout, stderr = -1, "", "Command timed out"
                except Exception as e:
                    exit_code, stdout, stderr = -1, "", str(e)
        
        except Exception as e:
            return {
                "scheduler": scheduler_name or "default",
                "error": str(e),
                "exit_code": -1
            }
        
        if exit_code != 0:
            print(f"Warning: double_bandwidth exited with code {exit_code}")
            print(f"stderr: {stderr}")
            return {
                "scheduler": scheduler_name or "default",
                "error": stderr or f"Exit code: {exit_code}",
                "exit_code": exit_code
            }
        
        # Parse results
        metrics = self._parse_bandwidth_output(stdout)
        metrics["scheduler"] = scheduler_name or "default"
        metrics["exit_code"] = exit_code
        metrics["array_size"] = self.test_params["array_size"]
        metrics["threads"] = self.test_params["threads"]
        metrics["read_ratio"] = self.test_params["read_ratio"]
        metrics["iterations"] = self.test_params["iterations"]
        
        return metrics
    
    def run_all_cxl_benchmarks(self, production_only: bool = True) -> dict:
        """
        Run CXL bandwidth tests for all schedulers.
        
        Args:
            production_only: Only test production-ready schedulers
            
        Returns:
            Dictionary mapping scheduler names to benchmark results
        """
        results = {}
        
        # Test default scheduler first
        print("Testing default scheduler...")
        results["default"] = self.run_cxl_benchmark()
        
        # Test each scheduler
        schedulers = self.runner.get_available_schedulers(production_only)
        for scheduler_name in schedulers:
            try:
                print(f"\nTesting scheduler: {scheduler_name}")
                results[scheduler_name] = self.run_cxl_benchmark(scheduler_name)
                
                # Save intermediate results
                self.save_results(results)
                
                # Brief pause between tests
                time.sleep(2)
                
            except Exception as e:
                print(f"Error testing scheduler {scheduler_name}: {e}")
                results[scheduler_name] = {
                    "scheduler": scheduler_name,
                    "error": str(e),
                    "bandwidth_mbps": 0
                }
        
        return results
    
    def save_results(self, results: dict):
        """Save results to JSON file"""
        results_file = os.path.join(self.results_dir, "cxl_scheduler_results.json")
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {results_file}")
    
    def generate_performance_figures(self, results: dict):
        """Generate performance comparison figures"""
        
        # Extract data for plotting
        schedulers = []
        bandwidth_values = []
        
        for scheduler_name, result in results.items():
            if "error" in result:
                continue
            
            schedulers.append(scheduler_name)
            bandwidth_values.append(result.get("bandwidth_mbps", 0))
        
        if not schedulers:
            print("No valid results to plot")
            return
        
        # Create figure with single plot
        plt.figure(figsize=(12, 8))
        
        # Bandwidth Performance
        bars = plt.bar(schedulers, bandwidth_values, color='skyblue', alpha=0.8)
        plt.xlabel('Scheduler')
        plt.ylabel('Bandwidth (MB/s)')
        plt.title('CXL Memory Bandwidth Scheduler Performance Comparison')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.0f}',
                    ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        
        # Save figure
        figure_path = os.path.join(self.results_dir, "cxl_scheduler_performance.png")
        plt.savefig(figure_path, dpi=300, bbox_inches='tight')
        print(f"Performance figure saved to {figure_path}")
        
        # Print summary
        self.print_performance_summary(results)
    
    def print_performance_summary(self, results: dict):
        """Print performance summary"""
        print("\n" + "="*60)
        print("CXL MEMORY BANDWIDTH SCHEDULER PERFORMANCE SUMMARY")
        print("="*60)
        
        for scheduler_name, result in results.items():
            if "error" in result:
                print(f"\n{scheduler_name:15} ERROR: {result['error']}")
                continue
            
            print(f"\n{scheduler_name:15}")
            print(f"  Bandwidth:       {result.get('bandwidth_mbps', 0):8.1f} MB/s")
            if "execution_time" in result:
                print(f"  Execution Time:  {result.get('execution_time', 0):8.2f} seconds")
            print(f"  Array Size:      {result.get('array_size', 0) / (1024**3):8.2f} GB")
            print(f"  Threads:         {result.get('threads', 0):8d}")
            print(f"  Read Ratio:      {result.get('read_ratio', 0):8.2f}")
    
    def run_parameter_sweep(self, scheduler_name: str = None, 
                           thread_counts: list = None, read_ratios: list = None):
        """
        Run parameter sweep for a specific scheduler testing thread counts and read ratios.
        
        Args:
            scheduler_name: Name of the scheduler to test
            thread_counts: List of thread counts to test
            read_ratios: List of read ratios to test (0.0-1.0)
        """
        thread_counts = thread_counts or [1, 2, 4, 8, 16, 32]
        read_ratios = read_ratios or [0.0, 0.25, 0.5, 0.75, 1.0]
        
        # Keep array size fixed at 1GB for consistency
        array_size = int(1 * 1024**3)
        
        print(f"Running parameter sweep for scheduler: {scheduler_name or 'default'}")
        print(f"Thread counts: {thread_counts}")
        print(f"Read ratios: {read_ratios}")
        print(f"Fixed array size: {array_size/(1024**3):.1f} GB")
        
        results = []
        total_tests = len(thread_counts) * len(read_ratios)
        test_count = 0
        
        for threads in thread_counts:
            for read_ratio in read_ratios:
                test_count += 1
                print(f"\nTest {test_count}/{total_tests}: "
                      f"threads={threads}, read_ratio={read_ratio:.2f}")
                
                # Update test parameters
                self.set_test_params(threads=threads, array_size=array_size, read_ratio=read_ratio)
                
                # Run benchmark
                result = self.run_cxl_benchmark(scheduler_name)
                
                if "error" not in result:
                    results.append({
                        'scheduler': scheduler_name or 'default',
                        'threads': threads,
                        'read_ratio': read_ratio,
                        'bandwidth_mbps': result.get('bandwidth_mbps', 0),
                        'execution_time': result.get('execution_time', 0),
                        'array_size_gb': array_size / (1024**3)
                    })
                    print(f"  Bandwidth: {result.get('bandwidth_mbps', 0):.2f} MB/s")
                else:
                    print(f"  Failed: {result['error']}")
                
                time.sleep(1)  # Brief pause between tests
        
        if results:
            # Save results
            df = pd.DataFrame(results)
            results_file = os.path.join(self.results_dir, 
                                      f"parameter_sweep_{scheduler_name or 'default'}.csv")
            df.to_csv(results_file, index=False)
            print(f"\nParameter sweep results saved to {results_file}")
            
            # Generate parameter sweep visualization with grid layout
            self._generate_parameter_sweep_plot(df, scheduler_name or 'default')
    
    def _generate_parameter_sweep_plot(self, df, scheduler_name):
        """Generate parameter sweep visualization with grid layout for each parameter configuration"""
        
        # Create main figure with subplots in a grid layout
        # We'll create a 3x2 grid for better visualization
        fig = plt.figure(figsize=(18, 16))
        fig.suptitle(f'Parameter Sweep Results for {scheduler_name}', fontsize=18, y=0.995)
        
        # Create a GridSpec for flexible subplot arrangement
        gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.25)
        
        # 1. Main heatmap - bandwidth for threads vs read_ratio
        ax1 = fig.add_subplot(gs[0, :])  # Top row, spans both columns
        bandwidth_pivot = df.pivot(index='threads', columns='read_ratio', values='bandwidth_mbps')
        sns.heatmap(bandwidth_pivot, annot=True, fmt='.0f', cmap='viridis', ax=ax1, cbar_kws={'label': 'MB/s'})
        ax1.set_title('Memory Bandwidth Heatmap (MB/s)', fontsize=14, pad=10)
        ax1.set_xlabel('Read Ratio', fontsize=12)
        ax1.set_ylabel('Thread Count', fontsize=12)
        
        # 2. Line plot - bandwidth vs threads for different read ratios
        ax2 = fig.add_subplot(gs[1, 0])
        read_ratios = sorted(df['read_ratio'].unique())
        colors = plt.cm.coolwarm(np.linspace(0, 1, len(read_ratios)))
        for i, ratio in enumerate(read_ratios):
            subset = df[df['read_ratio'] == ratio]
            ax2.plot(subset['threads'], subset['bandwidth_mbps'], 
                    marker='o', label=f'Read={ratio:.2f}', color=colors[i], linewidth=2)
        ax2.set_xlabel('Thread Count', fontsize=11)
        ax2.set_ylabel('Bandwidth (MB/s)', fontsize=11)
        ax2.set_title('Bandwidth vs Thread Count', fontsize=12)
        ax2.legend(loc='best', fontsize=9)
        ax2.grid(True, alpha=0.3)
        ax2.set_xscale('log', base=2)
        
        # 3. Line plot - bandwidth vs read ratio for different thread counts
        ax3 = fig.add_subplot(gs[1, 1])
        thread_counts = sorted(df['threads'].unique())
        colors = plt.cm.plasma(np.linspace(0, 0.9, len(thread_counts)))
        for i, threads in enumerate(thread_counts):
            subset = df[df['threads'] == threads]
            ax3.plot(subset['read_ratio'], subset['bandwidth_mbps'], 
                    marker='s', label=f'{threads}T', color=colors[i], linewidth=2)
        ax3.set_xlabel('Read Ratio', fontsize=11)
        ax3.set_ylabel('Bandwidth (MB/s)', fontsize=11)
        ax3.set_title('Bandwidth vs Read Ratio', fontsize=12)
        ax3.legend(loc='best', fontsize=9, ncol=2)
        ax3.grid(True, alpha=0.3)
        
        # 4. Efficiency heatmap - bandwidth per thread
        ax4 = fig.add_subplot(gs[2, 0])
        df['bandwidth_per_thread'] = df['bandwidth_mbps'] / df['threads']
        efficiency_pivot = df.pivot(index='threads', columns='read_ratio', values='bandwidth_per_thread')
        sns.heatmap(efficiency_pivot, annot=True, fmt='.0f', cmap='plasma', ax=ax4, cbar_kws={'label': 'MB/s per thread'})
        ax4.set_title('Bandwidth Efficiency (MB/s per thread)', fontsize=12)
        ax4.set_xlabel('Read Ratio', fontsize=11)
        ax4.set_ylabel('Thread Count', fontsize=11)
        
        # 5. 3D surface plot for bandwidth
        ax5 = fig.add_subplot(gs[2, 1], projection='3d')
        threads_mesh, ratios_mesh = np.meshgrid(
            sorted(df['threads'].unique()),
            sorted(df['read_ratio'].unique())
        )
        bandwidth_mesh = np.zeros_like(threads_mesh, dtype=float)
        
        for i, ratio in enumerate(sorted(df['read_ratio'].unique())):
            for j, threads in enumerate(sorted(df['threads'].unique())):
                val = df[(df['read_ratio'] == ratio) & (df['threads'] == threads)]['bandwidth_mbps']
                if not val.empty:
                    bandwidth_mesh[i, j] = val.values[0]
        
        surf = ax5.plot_surface(threads_mesh, ratios_mesh, bandwidth_mesh, 
                                cmap='viridis', alpha=0.8, edgecolor='none')
        ax5.set_xlabel('Threads', fontsize=10)
        ax5.set_ylabel('Read Ratio', fontsize=10)
        ax5.set_zlabel('Bandwidth (MB/s)', fontsize=10)
        ax5.set_title('3D Bandwidth Surface', fontsize=12)
        ax5.view_init(elev=25, azim=45)
        fig.colorbar(surf, ax=ax5, shrink=0.5, aspect=5)
        
        plt.tight_layout()
        
        # Save figure
        figure_path = os.path.join(self.results_dir, f"parameter_sweep_{scheduler_name}.png")
        plt.savefig(figure_path, dpi=300, bbox_inches='tight')
        print(f"Parameter sweep figure saved to {figure_path}")
        
        # Also generate individual parameter configuration figures
        self._generate_individual_config_plots(df, scheduler_name)


def main():
    """Main function for CXL-micro scheduler testing"""
    parser = argparse.ArgumentParser(description="Test schedulers with CXL-micro double_bandwidth")
    parser.add_argument("--double-bandwidth-path", 
                       default="/root/yunwei37/ai-os/workloads/cxl-micro/double_bandwidth",
                       help="Path to double_bandwidth binary")
    parser.add_argument("--results-dir", default="results", 
                       help="Directory to store results")
    parser.add_argument("--production-only", action="store_true", 
                       help="Test only production schedulers")
    parser.add_argument("--threads", type=int, default=8, 
                       help="Number of threads for testing")
    parser.add_argument("--array-size", type=int, default=1073741824, 
                       help="Array size in bytes (default 1GB)")
    parser.add_argument("--iterations", type=int, default=3, 
                       help="Number of iterations per test")
    parser.add_argument("--read-ratio", type=float, default=0.5,
                       help="Ratio of readers (0.0-1.0, default: 0.5)")
    parser.add_argument("--timeout", type=int, default=30000, 
                       help="Timeout in seconds")
    parser.add_argument("--parameter-sweep", action="store_true",
                       help="Run parameter sweep for each scheduler")
    parser.add_argument("--scheduler", type=str, default=None,
                       help="Test specific scheduler only")
    
    args = parser.parse_args()
    
    # Create tester instance
    tester = CXLMicroBenchmarkTester(args.double_bandwidth_path, args.results_dir)
    
    # Validate read_ratio
    if args.read_ratio < 0.0 or args.read_ratio > 1.0:
        print("Error: read-ratio must be between 0.0 and 1.0")
        sys.exit(1)
    
    # Update test parameters
    tester.set_test_params(
        threads=args.threads,
        array_size=args.array_size,
        iterations=args.iterations,
        timeout=args.timeout,
        read_ratio=args.read_ratio
    )
    
    # Check if binary exists
    if not os.path.exists(args.double_bandwidth_path):
        print(f"Error: double_bandwidth not found at {args.double_bandwidth_path}")
        sys.exit(1)
    
    if args.parameter_sweep:
        if args.scheduler:
            print(f"Running parameter sweep for scheduler: {args.scheduler}")
            tester.run_parameter_sweep(args.scheduler)
        else:
            print("Running parameter sweep for all schedulers...")
            # Run parameter sweep for default scheduler
            tester.run_parameter_sweep()
            
            # Run parameter sweep for each scheduler
            schedulers = tester.runner.get_available_schedulers(args.production_only)
            for scheduler_name in schedulers:
                try:
                    tester.run_parameter_sweep(scheduler_name)
                except Exception as e:
                    print(f"Error in parameter sweep for {scheduler_name}: {e}")
    else:
        if args.scheduler:
            print(f"Testing scheduler: {args.scheduler}")
            result = tester.run_cxl_benchmark(args.scheduler)
            results = {args.scheduler: result}
        else:
            print("Starting CXL memory bandwidth scheduler performance tests...")
            results = tester.run_all_cxl_benchmarks(production_only=args.production_only)
        
        # Generate figures
        tester.generate_performance_figures(results)
    
    print("\nTesting complete!")


if __name__ == "__main__":
    main()