#!/usr/bin/env python3
"""
Redis Scheduler Testing Script
Tests different schedulers with Redis benchmarks to compare performance.
"""

import os
import sys
import subprocess
import json
import time
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse

# Add the scheduler module to the path
sys.path.insert(0, '/root/yunwei37/ai-os/')

from scheduler import SchedulerRunner, SchedulerBenchmark
from redis_benchmark import RedisBenchmark


class RedisBenchmarkTester(SchedulerBenchmark):
    """
    Specialized class for testing schedulers with Redis benchmarks.
    
    This class extends SchedulerBenchmark to provide Redis-specific
    functionality including performance testing and result visualization.
    """
    
    def __init__(self, redis_dir: str = "redis-src", results_dir: str = "results",
                 scheduler_runner: SchedulerRunner = None):
        """
        Initialize the RedisBenchmarkTester.
        
        Args:
            redis_dir: Path to Redis source directory
            results_dir: Directory to store results
            scheduler_runner: SchedulerRunner instance to use
        """
        super().__init__(scheduler_runner)
        
        self.redis_dir = redis_dir
        self.results_dir = results_dir
        
        # Create results directory
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Default test parameters
        self.test_params = {
            "clients": 50,
            "requests": 100000,
            "data_size": 3,
            "pipeline": 1,
            "keyspace": None,
            "timeout": 300,
        }
        
        # Redis configuration options
        self.redis_config = {
            "io_threads": 4,
            "io_threads_do_reads": "yes",
            "maxmemory": "1gb",
            "hz": 100
        }
    
    def set_test_params(self, **kwargs):
        """
        Update test parameters.
        
        Args:
            **kwargs: Test parameters to update
        """
        self.test_params.update(kwargs)
    
    def set_redis_config(self, **kwargs):
        """
        Update Redis configuration.
        
        Args:
            **kwargs: Redis config parameters to update
        """
        self.redis_config.update(kwargs)
    
    def run_redis_benchmark(self, scheduler_name: str = None) -> dict:
        """
        Run Redis benchmark with specified scheduler.
        
        Args:
            scheduler_name: Name of the scheduler to test (None for default)
            
        Returns:
            Dictionary containing benchmark results
        """
        print(f"Running Redis benchmark with scheduler: {scheduler_name or 'default'}")
        tests_display = self.test_params.get('tests') or 'all default tests'
        print(f"Parameters: clients={self.test_params['clients']}, "
              f"requests={self.test_params['requests']}, "
              f"tests={tests_display}")
        
        try:
            # Create Redis benchmark instance with current configuration
            redis_bench = RedisBenchmark(
                redis_dir=self.redis_dir,
                results_dir=self.results_dir,
                config_options=self.redis_config
            )
            
            # Function to run benchmark
            def run_benchmark():
                # Remove timeout from test_params for run_comprehensive_benchmark
                benchmark_params = {k: v for k, v in self.test_params.items() if k != 'timeout'}
                return redis_bench.run_comprehensive_benchmark(**benchmark_params)
            
            if scheduler_name:
                # Run with specific scheduler
                # Since Redis benchmark manages its own process, we need to run it differently
                # We'll use a wrapper approach
                print(f"Starting scheduler: {scheduler_name}")
                
                # First ensure no other scheduler is running
                self.runner.stop_all_schedulers()
                # Also kill any lingering scx schedulers
                subprocess.run(["sudo", "pkill", "-f", "scx_"], capture_output=True)
                time.sleep(2)
                
                try:
                    proc = self.runner.start_scheduler(scheduler_name)
                    # Check if scheduler started successfully
                    if proc.poll() is not None:
                        stdout, stderr = proc.communicate()
                        error_msg = f"Failed to start scheduler {scheduler_name}: Scheduler failed to start: {stderr if stderr else stdout}"
                        return {
                            "scheduler": scheduler_name,
                            "error": error_msg,
                            "exit_code": -1
                        }
                except Exception as e:
                    error_msg = f"Failed to start scheduler {scheduler_name}: {str(e)}"
                    return {
                        "scheduler": scheduler_name,
                        "error": error_msg,
                        "exit_code": -1
                    }
                
                # Small delay to ensure scheduler is running
                time.sleep(2)
                
                try:
                    # Run benchmark while scheduler is active
                    results = run_benchmark()
                    
                    # Stop scheduler
                    self.runner.stop_scheduler()
                    
                    if not results:
                        return {
                            "scheduler": scheduler_name,
                            "error": "Benchmark failed to produce results",
                            "exit_code": -1
                        }
                    
                    # Add scheduler info to results
                    for result in results:
                        result["scheduler"] = scheduler_name
                    
                    return {
                        "scheduler": scheduler_name,
                        "results": results,
                        "summary": redis_bench.generate_summary(results),
                        "exit_code": 0
                    }
                    
                except Exception as e:
                    self.runner.stop_scheduler()
                    return {
                        "scheduler": scheduler_name,
                        "error": str(e),
                        "exit_code": -1
                    }
            else:
                # Run with default scheduler
                results = run_benchmark()
                if not results:
                    return {
                        "scheduler": "default",
                        "error": "Benchmark failed to produce results",
                        "exit_code": -1
                    }
                
                # Add scheduler info to results
                for result in results:
                    result["scheduler"] = "default"
                
                return {
                    "scheduler": "default",
                    "results": results,
                    "summary": redis_bench.generate_summary(results),
                    "exit_code": 0
                }
        
        except Exception as e:
            return {
                "scheduler": scheduler_name or "default",
                "error": str(e),
                "exit_code": -1
            }
    
    def extract_metrics_from_results(self, benchmark_result: dict) -> dict:
        """Extract key metrics from Redis benchmark results"""
        metrics = {
            "scheduler": benchmark_result.get("scheduler", "unknown"),
            "latency_p50": [],
            "latency_p95": [],
            "latency_p99": [],
            "latency_avg": [],
            "throughput_rps": [],
            "test_names": []
        }
        
        if "error" in benchmark_result:
            return metrics
        
        results = benchmark_result.get("results", [])
        
        # Process each test result
        for result in results:
            test_name = result.get("test_name", "unknown")
            parsed_metrics = result.get("parsed_metrics", [])
            
            # Use parsed_metrics as primary source (more detailed)
            if parsed_metrics:
                for metric in parsed_metrics:
                    test = metric.get("test", test_name.replace("_operations", ""))
                    rps = metric.get("rps", 0)
                    
                    if rps > 0:  # Only include valid metrics
                        metrics["test_names"].append(test)
                        metrics["throughput_rps"].append(rps)
                        
                        # Extract latency metrics
                        metrics["latency_avg"].append(metric.get("avg_latency_ms", 0))
                        metrics["latency_p50"].append(metric.get("p50_latency_ms", 0))
                        metrics["latency_p95"].append(metric.get("p95_latency_ms", 0))
                        metrics["latency_p99"].append(metric.get("p99_latency_ms", 0))
            
            # Fallback to regular metrics if parsed_metrics is empty
            elif result.get("metrics"):
                test_metrics = result.get("metrics", {})
                for test_name_key, test_data in test_metrics.items():
                    if isinstance(test_data, dict):
                        rps = test_data.get("requests_per_second", 0)
                        if rps > 0:
                            metrics["test_names"].append(test_name_key)
                            metrics["throughput_rps"].append(rps)
                            
                            # For fallback, we don't have detailed latency info
                            metrics["latency_avg"].append(0)
                            metrics["latency_p50"].append(0)
                            metrics["latency_p95"].append(0)
                            metrics["latency_p99"].append(0)
        
        return metrics
    
    def run_all_redis_benchmarks(self, production_only: bool = True) -> dict:
        """
        Run Redis benchmarks for all schedulers.
        
        Args:
            production_only: Only test production-ready schedulers
            
        Returns:
            Dictionary mapping scheduler names to benchmark results
        """
        results = {}
        
        # Test default scheduler first
        print("Testing default scheduler...")
        results["default"] = self.run_redis_benchmark()
        
        # Test each scheduler
        schedulers = self.runner.get_available_schedulers(production_only)
        for scheduler_name in schedulers:
            try:
                print(f"\nTesting scheduler: {scheduler_name}")
                results[scheduler_name] = self.run_redis_benchmark(scheduler_name)
                
                # Save intermediate results
                self.save_results(results)
                
                # Brief pause between tests
                time.sleep(3)
                
            except Exception as e:
                print(f"Error testing scheduler {scheduler_name}: {e}")
                results[scheduler_name] = {
                    "scheduler": scheduler_name,
                    "error": str(e),
                    "exit_code": -1
                }
        
        return results
    
    def save_results(self, results: dict):
        """Save results to JSON file"""
        # Use a single file name without timestamp
        results_file = os.path.join(self.results_dir, "redis_scheduler_results.json")
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {results_file}")
    
    def generate_performance_figures(self, results: dict):
        """Generate performance comparison figures"""
        
        # Extract metrics from all results
        all_metrics = {}
        for scheduler_name, result in results.items():
            if "error" not in result:
                all_metrics[scheduler_name] = self.extract_metrics_from_results(result)
        
        if not all_metrics:
            print("No valid results to plot")
            return
        
        # Create separate figures for different metrics
        self._plot_throughput_comparison(all_metrics)
        self._plot_latency_comparison(all_metrics)
        self._plot_combined_performance(all_metrics)
        
        # Print summary
        self.print_performance_summary(results)
    
    def _plot_throughput_comparison(self, all_metrics: dict):
        """Plot throughput comparison across schedulers"""
        fig, ax = plt.subplots(figsize=(15, 8))
        
        # Prepare data for plotting
        schedulers = []
        test_types = set()
        
        # Collect all test types
        for metrics in all_metrics.values():
            test_types.update(metrics["test_names"])
        
        test_types = sorted(list(test_types))
        
        # Create data matrix
        data = []
        for scheduler_name, metrics in all_metrics.items():
            scheduler_data = []
            for test_type in test_types:
                # Find throughput for this test type
                if test_type in metrics["test_names"]:
                    idx = metrics["test_names"].index(test_type)
                    throughput = metrics["throughput_rps"][idx]
                else:
                    throughput = 0
                scheduler_data.append(throughput)
            data.append(scheduler_data)
            schedulers.append(scheduler_name)
        
        # Create grouped bar chart
        x = np.arange(len(test_types))
        width = 0.8 / len(schedulers)
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(schedulers)))
        
        for i, (scheduler, color) in enumerate(zip(schedulers, colors)):
            offset = (i - len(schedulers)/2 + 0.5) * width
            bars = ax.bar(x + offset, data[i], width, label=scheduler, color=color, alpha=0.8)
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:.0f}',
                           ha='center', va='bottom', fontsize=8, rotation=90)
        
        ax.set_xlabel('Redis Operations')
        ax.set_ylabel('Throughput (Requests/Second)')
        ax.set_title('Redis Throughput Performance by Scheduler')
        ax.set_xticks(x)
        ax.set_xticklabels(test_types, rotation=45)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save figure
        figure_path = os.path.join(self.results_dir, "redis_throughput_comparison.png")
        plt.savefig(figure_path, dpi=300, bbox_inches='tight')
        print(f"Throughput comparison figure saved to {figure_path}")
        plt.close()
    
    def _plot_latency_comparison(self, all_metrics: dict):
        """Plot latency comparison across schedulers"""
        latency_types = ["latency_avg", "latency_p50", "latency_p95", "latency_p99"]
        latency_labels = ["Average Latency", "P50 Latency", "P95 Latency", "P99 Latency"]
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Redis Latency Performance by Scheduler', fontsize=16)
        axes = axes.flatten()
        
        for idx, (latency_type, latency_label) in enumerate(zip(latency_types, latency_labels)):
            ax = axes[idx]
            
            # Collect data for this latency metric
            scheduler_names = []
            avg_latencies = []
            
            for scheduler_name, metrics in all_metrics.items():
                if metrics[latency_type]:  # If we have latency data
                    # Calculate average latency across all operations
                    valid_latencies = [l for l in metrics[latency_type] if l > 0]
                    if valid_latencies:
                        avg_latency = np.mean(valid_latencies)
                        scheduler_names.append(scheduler_name)
                        avg_latencies.append(avg_latency)
            
            if scheduler_names and avg_latencies:
                colors = plt.cm.Set3(np.linspace(0, 1, len(scheduler_names)))
                bars = ax.bar(scheduler_names, avg_latencies, color=colors, alpha=0.8)
                ax.set_xlabel('Scheduler')
                ax.set_ylabel('Latency (ms)')
                ax.set_title(f'{latency_label} (Lower is Better)')
                ax.tick_params(axis='x', rotation=45)
                ax.grid(True, alpha=0.3)
                
                # Add value labels on bars
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:.2f}',
                           ha='center', va='bottom', fontsize=8)
            else:
                ax.text(0.5, 0.5, 'No latency data available', 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'{latency_label} (No Data)')
        
        plt.tight_layout()
        
        # Save figure
        figure_path = os.path.join(self.results_dir, "redis_latency_comparison.png")
        plt.savefig(figure_path, dpi=300, bbox_inches='tight')
        print(f"Latency comparison figure saved to {figure_path}")
        plt.close()
    
    def _plot_combined_performance(self, all_metrics: dict):
        """Plot combined performance score"""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Calculate performance scores
        scheduler_names = []
        performance_scores = []
        throughput_scores = []
        latency_scores = []
        
        # Collect average metrics for each scheduler
        scheduler_stats = {}
        for scheduler_name, metrics in all_metrics.items():
            if metrics["throughput_rps"]:
                avg_throughput = np.mean([t for t in metrics["throughput_rps"] if t > 0])
                # Use average latency for combined score, fallback to P95 if not available
                valid_latencies = [l for l in metrics["latency_avg"] if l > 0]
                if not valid_latencies:
                    valid_latencies = [l for l in metrics["latency_p95"] if l > 0]
                avg_latency = np.mean(valid_latencies) if valid_latencies else float('inf')
                
                scheduler_stats[scheduler_name] = {
                    "throughput": avg_throughput,
                    "latency": avg_latency
                }
        
        if not scheduler_stats:
            print("No valid performance data for combined score")
            return
        
        # Normalize scores
        max_throughput = max(s["throughput"] for s in scheduler_stats.values())
        min_latency = min(s["latency"] for s in scheduler_stats.values() if s["latency"] != float('inf'))
        
        for scheduler_name, stats in scheduler_stats.items():
            # Throughput score (higher is better, normalize to 0-1)
            throughput_score = stats["throughput"] / max_throughput
            
            # Latency score (lower is better, invert and normalize to 0-1)
            if stats["latency"] != float('inf'):
                latency_score = min_latency / stats["latency"]
            else:
                latency_score = 0
            
            # Combined score (weighted average)
            combined_score = 0.6 * throughput_score + 0.4 * latency_score
            
            scheduler_names.append(scheduler_name)
            performance_scores.append(combined_score)
            throughput_scores.append(throughput_score)
            latency_scores.append(latency_score)
        
        # Create stacked bar chart
        width = 0.6
        ax.bar(scheduler_names, throughput_scores, width, label='Throughput Score (60%)', alpha=0.8)
        ax.bar(scheduler_names, latency_scores, width, bottom=throughput_scores, 
               label='Latency Score (40%)', alpha=0.8)
        
        ax.set_xlabel('Scheduler')
        ax.set_ylabel('Normalized Performance Score')
        ax.set_title('Redis Combined Performance Score\n(Higher is Better)')
        ax.tick_params(axis='x', rotation=45)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add total score labels
        for i, score in enumerate(performance_scores):
            ax.text(i, score + 0.05, f'{score:.2f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        
        # Save figure
        figure_path = os.path.join(self.results_dir, "redis_combined_performance.png")
        plt.savefig(figure_path, dpi=300, bbox_inches='tight')
        print(f"Combined performance figure saved to {figure_path}")
        plt.close()
    
    def print_performance_summary(self, results: dict):
        """Print performance summary"""
        print("\n" + "="*60)
        print("REDIS SCHEDULER PERFORMANCE SUMMARY")
        print("="*60)
        
        for scheduler_name, result in results.items():
            if "error" in result:
                print(f"\n{scheduler_name:15} ERROR: {result['error']}")
                continue
            
            print(f"\n{scheduler_name:15}")
            
            # Extract key metrics
            metrics = self.extract_metrics_from_results(result)
            
            if metrics["throughput_rps"]:
                avg_throughput = np.mean([t for t in metrics["throughput_rps"] if t > 0])
                print(f"  Avg Throughput:  {avg_throughput:8.0f} req/s")
            
            # Print average latency if available, otherwise P95
            if metrics["latency_avg"]:
                valid_latencies = [l for l in metrics["latency_avg"] if l > 0]
                if valid_latencies:
                    avg_latency = np.mean(valid_latencies)
                    print(f"  Avg Latency:     {avg_latency:8.2f} ms")
            elif metrics["latency_p95"]:
                valid_latencies = [l for l in metrics["latency_p95"] if l > 0]
                if valid_latencies:
                    avg_latency = np.mean(valid_latencies)
                    print(f"  Avg P95 Latency: {avg_latency:8.2f} ms")
            
            # Print test results
            if result.get("summary"):
                summary = result["summary"]
                print(f"  Tests Completed: {summary.get('successful_tests', 0)}")
                print(f"  Total Duration:  {summary.get('total_duration', 0):8.2f} seconds")

    def run_parameter_sweep_multi_schedulers(self, schedulers: list = None,
                                            data_sizes: list = None,
                                            production_only: bool = False):
        """
        Run data size parameter sweep for multiple schedulers and generate comparison plots.
        
        Args:
            schedulers: List of scheduler names to test (None for all + default)
            data_sizes: List of data sizes to test (in bytes)
            production_only: Only test production-ready schedulers if schedulers is None
        """
        data_sizes = data_sizes or [1, 16, 64, 256, 1024, 4096]  # bytes
        
        # Get schedulers to test
        if schedulers is None:
            schedulers = ['default'] + self.runner.get_available_schedulers(production_only)
        elif 'default' not in schedulers:
            schedulers = ['default'] + schedulers
                
        print(f"Running data size parameter sweep for schedulers: {schedulers}")
        print(f"Data sizes: {data_sizes} bytes")
        
        all_results = []
        total_tests = len(schedulers) * len(data_sizes)
        test_count = 0
        
        for scheduler_name in schedulers:
            print(f"\n{'='*50}")
            print(f"Testing scheduler: {scheduler_name}")
            print(f"{'='*50}")
            
            for data_size in data_sizes:
                test_count += 1
                print(f"\nTest {test_count}/{total_tests}: "
                      f"scheduler={scheduler_name}, data_size={data_size}B")
                
                # Update test parameters
                self.set_test_params(
                    data_size=data_size,
                    requests=self.test_params["requests"],
                    timeout=self.test_params["timeout"]
                )
                
                # Run benchmark
                scheduler_to_test = None if scheduler_name == 'default' else scheduler_name
                result = self.run_redis_benchmark(scheduler_to_test)
                
                if "error" not in result:
                    # Extract metrics from the result
                    metrics = self.extract_metrics_from_results(result)
                    
                    # Add each test result as a separate row
                    for i, test_name in enumerate(metrics["test_names"]):
                        all_results.append({
                            'scheduler': scheduler_name,
                            'data_size': data_size,
                            'test_name': test_name,
                            'throughput_rps': metrics["throughput_rps"][i] if i < len(metrics["throughput_rps"]) else 0,
                            'latency_avg': metrics["latency_avg"][i] if i < len(metrics["latency_avg"]) else 0,
                            'latency_p50': metrics["latency_p50"][i] if i < len(metrics["latency_p50"]) else 0,
                            'latency_p95': metrics["latency_p95"][i] if i < len(metrics["latency_p95"]) else 0,
                            'latency_p99': metrics["latency_p99"][i] if i < len(metrics["latency_p99"]) else 0,
                            'requests': self.test_params["requests"],
                            'clients': self.test_params["clients"],
                            'pipeline': self.test_params["pipeline"]
                        })
                    
                    # Calculate average throughput for this configuration
                    avg_throughput = np.mean([t for t in metrics["throughput_rps"] if t > 0]) if metrics["throughput_rps"] else 0
                    print(f"  Avg Throughput: {avg_throughput:.0f} req/s")
                else:
                    print(f"  Failed: {result['error']}")
                    # Add failure record
                    all_results.append({
                        'scheduler': scheduler_name,
                        'data_size': data_size,
                        'test_name': 'failed',
                        'throughput_rps': 0,
                        'latency_avg': 0,
                        'latency_p50': 0,
                        'latency_p95': 0,
                        'latency_p99': 0,
                        'requests': self.test_params["requests"],
                        'clients': self.test_params["clients"],
                        'pipeline': self.test_params["pipeline"]
                    })
                
                time.sleep(2)  # Brief pause between tests
        
        if all_results:
            # Save results
            df = pd.DataFrame(all_results)
            results_file = os.path.join(self.results_dir, "redis_data_size_sweep.csv")
            df.to_csv(results_file, index=False)
            print(f"\nData size parameter sweep results saved to {results_file}")
            
            # Generate visualization
            self._plot_data_size_sweep(df)
    
    def _plot_data_size_sweep(self, df):
        """Plot performance vs data size for each scheduler"""
        data_sizes = sorted(df['data_size'].unique())
        schedulers = sorted(df['scheduler'].unique())
        
        # Color palette for different schedulers
        colors = plt.cm.tab10(np.linspace(0, 1, len(schedulers)))
        scheduler_colors = {sched: colors[i] for i, sched in enumerate(schedulers)}
        
        # Create dual plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle('Redis Performance vs Data Size', fontsize=16)
        
        # Plot throughput and latency for each scheduler
        for scheduler in schedulers:
            sched_data = df[df['scheduler'] == scheduler]
            color = scheduler_colors[scheduler]
            
            # Aggregate data by data size
            avg_throughput = []
            avg_latency = []
            
            for data_size in data_sizes:
                size_data = sched_data[sched_data['data_size'] == data_size]
                
                # Average throughput for this data size
                avg_tput = size_data['throughput_rps'].mean() if not size_data.empty else 0
                avg_throughput.append(avg_tput)
                
                # Average latency (prefer P95, fallback to average)
                if not size_data.empty:
                    latency_values = size_data['latency_p95']
                    if latency_values.sum() == 0:
                        latency_values = size_data['latency_avg']
                    avg_lat = latency_values.mean() if latency_values.sum() > 0 else 0
                else:
                    avg_lat = 0
                avg_latency.append(avg_lat)
            
            # Plot throughput
            ax1.plot(data_sizes, avg_throughput, 'o-', 
                    color=color, label=scheduler, linewidth=2, markersize=6)
            
            # Plot latency  
            ax2.plot(data_sizes, avg_latency, 'o-',
                    color=color, label=scheduler, linewidth=2, markersize=6)
        
        # Configure throughput plot
        ax1.set_xlabel('Data Size (bytes)')
        ax1.set_ylabel('Average Throughput (req/s)')
        ax1.set_title('Throughput vs Data Size')
        ax1.set_xscale('log')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Configure latency plot
        ax2.set_xlabel('Data Size (bytes)')
        ax2.set_ylabel('Average P95 Latency (ms)')
        ax2.set_title('P95 Latency vs Data Size')
        ax2.set_xscale('log')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save figure
        figure_path = os.path.join(self.results_dir, "redis_data_size_sweep.png")
        plt.savefig(figure_path, dpi=300, bbox_inches='tight')
        print(f"Data size sweep figure saved to {figure_path}")
        plt.close()


def main():
    """Main function for Redis scheduler testing"""
    parser = argparse.ArgumentParser(description="Test schedulers with Redis benchmarks")
    parser.add_argument("--redis-dir", default="redis-src",
                       help="Path to Redis source directory")
    parser.add_argument("--results-dir", default="results", 
                       help="Directory to store results")
    parser.add_argument("--production-only", action="store_true", 
                       help="Test only production schedulers")
    parser.add_argument("--clients", type=int, default=50, 
                       help="Number of Redis clients")
    parser.add_argument("--requests", type=int, default=1000000, 
                       help="Number of requests per test")
    parser.add_argument("--data-size", type=int, default=1,
                       help="Data size in bytes")
    parser.add_argument("--pipeline", type=int, default=16,
                       help="Pipeline requests")
    parser.add_argument("--tests", default=None,
                       help="Comma-separated list of tests to run")
    parser.add_argument("--io-threads", type=int, default=64,
                       help="Redis I/O threads")
    parser.add_argument("--maxmemory", default="256gb",
                       help="Redis max memory")
    parser.add_argument("--timeout", type=int, default=300,
                       help="Timeout in seconds")
    parser.add_argument("--scheduler", type=str, default=None,
                       help="Test specific scheduler only")
    parser.add_argument("--parameter-sweep", action="store_true",
                       help="Run parameter sweep comparing all schedulers with different data sizes")
    
    args = parser.parse_args()
    
    # Create tester instance
    tester = RedisBenchmarkTester(args.redis_dir, args.results_dir)
    
    # Update test parameters
    tester.set_test_params(
        clients=args.clients,
        requests=args.requests,
        data_size=args.data_size,
        pipeline=args.pipeline,
        tests=args.tests,
        timeout=args.timeout
    )
    
    # Update Redis configuration
    tester.set_redis_config(
        io_threads=args.io_threads,
        maxmemory=args.maxmemory
    )
    
    # Check if Redis binaries exist
    redis_server = os.path.join(args.redis_dir, "src", "redis-server")
    if not os.path.exists(redis_server):
        print(f"Error: Redis not found at {args.redis_dir}")
        print("Please ensure Redis is built in the specified directory")
        sys.exit(1)
    
    if args.parameter_sweep:
        print("Running data size parameter sweep...")
        schedulers = None
        if args.scheduler:
            schedulers = [args.scheduler]
        tester.run_parameter_sweep_multi_schedulers(
            schedulers=schedulers,
            production_only=args.production_only
        )
    else:
        if args.scheduler:
            print(f"Testing scheduler: {args.scheduler}")
            result = tester.run_redis_benchmark(args.scheduler)
            results = {args.scheduler: result}
        else:
            print("Starting Redis scheduler performance tests...")
            results = tester.run_all_redis_benchmarks(production_only=args.production_only)
        
        # Generate figures
        tester.generate_performance_figures(results)
    
    print("\nTesting complete!")


if __name__ == "__main__":
    main()