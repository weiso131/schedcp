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
        return [
            self.double_bandwidth_path,
            str(self.test_params["array_size"]),
            str(self.test_params["iterations"]),
            str(self.test_params["threads"])
        ]
    
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
        print(f"Parameters: array_size={self.test_params['array_size']}, "
              f"iterations={self.test_params['iterations']}, "
              f"threads={self.test_params['threads']}")
        
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
        execution_times = []
        
        for scheduler_name, result in results.items():
            if "error" in result:
                continue
            
            schedulers.append(scheduler_name)
            bandwidth_values.append(result.get("bandwidth_mbps", 0))
            execution_times.append(result.get("execution_time", 0))
        
        if not schedulers:
            print("No valid results to plot")
            return
        
        # Create figure with subplots
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle('CXL Memory Bandwidth Scheduler Performance Comparison', fontsize=16)
        
        # Plot 1: Bandwidth Performance
        bars1 = ax1.bar(schedulers, bandwidth_values, color='skyblue', alpha=0.8)
        ax1.set_xlabel('Scheduler')
        ax1.set_ylabel('Bandwidth (MB/s)')
        ax1.set_title('Memory Bandwidth Performance')
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar in bars1:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.0f}',
                    ha='center', va='bottom', fontsize=8)
        
        # Plot 2: Execution Time (if available)
        if any(execution_times):
            bars2 = ax2.bar(schedulers, execution_times, color='lightcoral', alpha=0.8)
            ax2.set_xlabel('Scheduler')
            ax2.set_ylabel('Execution Time (seconds)')
            ax2.set_title('Execution Time Comparison')
            ax2.tick_params(axis='x', rotation=45)
            ax2.grid(True, alpha=0.3)
            
            # Add value labels on bars
            for bar in bars2:
                height = bar.get_height()
                if height > 0:
                    ax2.text(bar.get_x() + bar.get_width()/2., height,
                            f'{height:.2f}',
                            ha='center', va='bottom', fontsize=8)
        
        # Plot 3: Normalized Performance Score
        if len(schedulers) > 1 and max(bandwidth_values) > 0:
            # Normalize bandwidth (higher is better)
            norm_bandwidth = np.array(bandwidth_values) / max(bandwidth_values)
            
            bars3 = ax3.bar(schedulers, norm_bandwidth, color='lightgreen', alpha=0.8)
            ax3.set_xlabel('Scheduler')
            ax3.set_ylabel('Normalized Performance Score')
            ax3.set_title('Relative Performance\n(Higher is Better)')
            ax3.tick_params(axis='x', rotation=45)
            ax3.grid(True, alpha=0.3)
            ax3.set_ylim(0, 1.1)
            
            # Add value labels on bars
            for bar in bars3:
                height = bar.get_height()
                ax3.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.2%}',
                        ha='center', va='bottom', fontsize=8)
        
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
    
    def run_parameter_sweep(self, scheduler_name: str = None, 
                           thread_counts: list = None, array_sizes: list = None):
        """
        Run parameter sweep for a specific scheduler.
        
        Args:
            scheduler_name: Name of the scheduler to test
            thread_counts: List of thread counts to test
            array_sizes: List of array sizes to test (in bytes)
        """
        thread_counts = thread_counts or [1, 2, 4, 8, 16, 32]
        # Array sizes in GB converted to bytes
        array_sizes = array_sizes or [
            int(0.5 * 1024**3),  # 0.5 GB
            int(1 * 1024**3),    # 1 GB
            int(2 * 1024**3),    # 2 GB
            int(4 * 1024**3),    # 4 GB
        ]
        
        print(f"Running parameter sweep for scheduler: {scheduler_name or 'default'}")
        print(f"Thread counts: {thread_counts}")
        print(f"Array sizes (GB): {[s/(1024**3) for s in array_sizes]}")
        
        results = []
        total_tests = len(thread_counts) * len(array_sizes)
        test_count = 0
        
        for threads in thread_counts:
            for array_size in array_sizes:
                test_count += 1
                print(f"\nTest {test_count}/{total_tests}: "
                      f"threads={threads}, array_size={array_size/(1024**3):.1f}GB")
                
                # Update test parameters
                self.set_test_params(threads=threads, array_size=array_size)
                
                # Run benchmark
                result = self.run_cxl_benchmark(scheduler_name)
                
                if "error" not in result:
                    results.append({
                        'scheduler': scheduler_name or 'default',
                        'threads': threads,
                        'array_size_gb': array_size / (1024**3),
                        'bandwidth_mbps': result.get('bandwidth_mbps', 0),
                        'execution_time': result.get('execution_time', 0)
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
            
            # Generate parameter sweep visualization
            self._generate_parameter_sweep_plot(df, scheduler_name or 'default')
    
    def _generate_parameter_sweep_plot(self, df, scheduler_name):
        """Generate parameter sweep visualization"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'Parameter Sweep Results for {scheduler_name}', fontsize=16)
        
        # 1. Heatmap for bandwidth
        bandwidth_pivot = df.pivot(index='threads', columns='array_size_gb', values='bandwidth_mbps')
        sns.heatmap(bandwidth_pivot, annot=True, fmt='.0f', cmap='viridis', ax=ax1)
        ax1.set_title('Memory Bandwidth (MB/s)')
        ax1.set_xlabel('Array Size (GB)')
        ax1.set_ylabel('Thread Count')
        
        # 2. Line plot for different array sizes
        array_sizes = sorted(df['array_size_gb'].unique())
        for size in array_sizes:
            subset = df[df['array_size_gb'] == size]
            ax2.plot(subset['threads'], subset['bandwidth_mbps'], 
                    marker='o', label=f'{size} GB')
        ax2.set_xlabel('Thread Count')
        ax2.set_ylabel('Bandwidth (MB/s)')
        ax2.set_title('Bandwidth vs Thread Count')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Line plot for different thread counts
        thread_counts = sorted(df['threads'].unique())
        for threads in thread_counts:
            subset = df[df['threads'] == threads]
            ax3.plot(subset['array_size_gb'], subset['bandwidth_mbps'], 
                    marker='s', label=f'{threads} threads')
        ax3.set_xlabel('Array Size (GB)')
        ax3.set_ylabel('Bandwidth (MB/s)')
        ax3.set_title('Bandwidth vs Array Size')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Efficiency plot (bandwidth per thread)
        df['bandwidth_per_thread'] = df['bandwidth_mbps'] / df['threads']
        efficiency_pivot = df.pivot(index='threads', columns='array_size_gb', 
                                   values='bandwidth_per_thread')
        sns.heatmap(efficiency_pivot, annot=True, fmt='.0f', cmap='plasma', ax=ax4)
        ax4.set_title('Bandwidth Efficiency (MB/s per thread)')
        ax4.set_xlabel('Array Size (GB)')
        ax4.set_ylabel('Thread Count')
        
        plt.tight_layout()
        
        # Save figure
        figure_path = os.path.join(self.results_dir, f"parameter_sweep_{scheduler_name}.png")
        plt.savefig(figure_path, dpi=300, bbox_inches='tight')
        print(f"Parameter sweep figure saved to {figure_path}")


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
    parser.add_argument("--iterations", type=int, default=10, 
                       help="Number of iterations per test")
    parser.add_argument("--timeout", type=int, default=300, 
                       help="Timeout in seconds")
    parser.add_argument("--parameter-sweep", action="store_true",
                       help="Run parameter sweep for each scheduler")
    parser.add_argument("--scheduler", type=str, default=None,
                       help="Test specific scheduler only")
    
    args = parser.parse_args()
    
    # Create tester instance
    tester = CXLMicroBenchmarkTester(args.double_bandwidth_path, args.results_dir)
    
    # Update test parameters
    tester.set_test_params(
        threads=args.threads,
        array_size=args.array_size,
        iterations=args.iterations,
        timeout=args.timeout
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