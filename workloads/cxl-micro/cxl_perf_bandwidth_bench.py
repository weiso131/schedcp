#!/usr/bin/env python3
"""
CXL-Micro Perf Bandwidth Benchmark
Integrates perf counters with double_bandwidth testing to calculate various bandwidth metrics.
Based on bandwidth_analysis.md methodology.
"""

import os
import sys
import subprocess
import json
import time
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import argparse
import re
from typing import Dict, List, Tuple, Optional

# Add the scheduler module to the path
sys.path.insert(0, '/root/yunwei37/ai-os/')

from scheduler import SchedulerRunner, SchedulerBenchmark


class PerfBandwidthCalculator:
    """Calculate bandwidth metrics from perf counters using cache line methodology"""
    
    CACHE_LINE_SIZE = 64  # bytes
    MB = 1024 * 1024
    
    # Enhanced perf counters from bandwidth_analysis.md
    PERF_COUNTERS = [
        "instructions",
        "cycles",
        "bus-cycles", 
        "cache-references",
        "cache-misses",
        "L1-dcache-loads",
        "L1-dcache-load-misses",
        "L1-dcache-stores",
        "mem_inst_retired.all_loads",
        "mem_inst_retired.all_stores",
        "longest_lat_cache.miss",
        "cycle_activity.stalls_total",
        "mem_load_l3_miss_retired.local_dram",
        "mem_load_l3_miss_retired.remote_dram",
        "ocr.demand_rfo.l3_miss"
    ]
    
    @classmethod
    def calculate_bandwidth_from_counter(cls, counter_value: int, wall_time: float) -> float:
        """
        Calculate bandwidth from perf counter value.
        
        Args:
            counter_value: Counter value from perf
            wall_time: Wall clock time in seconds
            
        Returns:
            Bandwidth in MB/s
        """
        return (counter_value * cls.CACHE_LINE_SIZE) / wall_time / cls.MB
    
    @classmethod
    def parse_perf_output(cls, perf_output: str) -> Dict[str, float]:
        """
        Parse perf stat output to extract counter values and timing.
        
        Args:
            perf_output: Raw perf stat output
            
        Returns:
            Dictionary with counter values and timing information
        """
        results = {}
        
        # Parse counter values
        for line in perf_output.split('\n'):
            line = line.strip()
            if not line or line.startswith('#'):
                continue
                
            # Look for counter pattern: "value counter_name"
            parts = line.split()
            if len(parts) >= 2:
                try:
                    # Remove commas from numbers
                    value_str = parts[0].replace(',', '')
                    counter_name = parts[1]
                    
                    # Skip if value is not a number
                    if value_str.replace('.', '').isdigit():
                        results[counter_name] = float(value_str)
                except (ValueError, IndexError):
                    continue
        
        # Parse timing information
        time_match = re.search(r'(\d+\.?\d*)\s+seconds time elapsed', perf_output)
        if time_match:
            results['wall_time'] = float(time_match.group(1))
        
        return results
    
    @classmethod
    def calculate_all_bandwidths(cls, perf_data: Dict[str, float]) -> Dict[str, float]:
        """
        Calculate all bandwidth metrics from perf counter data.
        
        Args:
            perf_data: Dictionary with perf counter values and wall_time
            
        Returns:
            Dictionary with calculated bandwidth metrics
        """
        if 'wall_time' not in perf_data:
            raise ValueError("Wall time not found in perf data")
        
        wall_time = perf_data['wall_time']
        bandwidth_metrics = {}
        
        # Method 1: Cache System Bandwidth (Total system memory load)
        if 'cache-references' in perf_data:
            bandwidth_metrics['cache_references_bw'] = cls.calculate_bandwidth_from_counter(
                perf_data['cache-references'], wall_time
            )
        
        # Method 2: Cross-Cache Bandwidth (Memory pressure)
        if 'cache-misses' in perf_data:
            bandwidth_metrics['cache_misses_bw'] = cls.calculate_bandwidth_from_counter(
                perf_data['cache-misses'], wall_time
            )
        
        # Method 3: Longest Latency Cache Miss Bandwidth
        if 'longest_lat_cache.miss' in perf_data:
            bandwidth_metrics['longest_lat_cache_miss_bw'] = cls.calculate_bandwidth_from_counter(
                perf_data['longest_lat_cache.miss'], wall_time
            )
        
        # Method 4: L1 Cache Bandwidth
        if 'L1-dcache-loads' in perf_data:
            bandwidth_metrics['l1_dcache_loads_bw'] = cls.calculate_bandwidth_from_counter(
                perf_data['L1-dcache-loads'], wall_time
            )
        
        if 'L1-dcache-load-misses' in perf_data:
            bandwidth_metrics['l1_dcache_load_misses_bw'] = cls.calculate_bandwidth_from_counter(
                perf_data['L1-dcache-load-misses'], wall_time
            )
        
        if 'L1-dcache-stores' in perf_data:
            bandwidth_metrics['l1_dcache_stores_bw'] = cls.calculate_bandwidth_from_counter(
                perf_data['L1-dcache-stores'], wall_time
            )
        
        # Method 5: DRAM Bandwidth (Actual memory utilization)
        if 'mem_load_l3_miss_retired.local_dram' in perf_data:
            bandwidth_metrics['local_dram_bw'] = cls.calculate_bandwidth_from_counter(
                perf_data['mem_load_l3_miss_retired.local_dram'], wall_time
            )
        
        if 'mem_load_l3_miss_retired.remote_dram' in perf_data:
            bandwidth_metrics['remote_dram_bw'] = cls.calculate_bandwidth_from_counter(
                perf_data['mem_load_l3_miss_retired.remote_dram'], wall_time
            )
        
        # Method 6: Memory Instructions Bandwidth
        if 'mem_inst_retired.all_loads' in perf_data and 'mem_inst_retired.all_stores' in perf_data:
            total_mem_ops = perf_data['mem_inst_retired.all_loads'] + perf_data['mem_inst_retired.all_stores']
            bandwidth_metrics['memory_instructions_bw'] = cls.calculate_bandwidth_from_counter(
                total_mem_ops, wall_time
            )
        
        # Method 7: RFO (Read For Ownership) L3 miss bandwidth
        if 'ocr.demand_rfo.l3_miss' in perf_data:
            bandwidth_metrics['rfo_l3_miss_bw'] = cls.calculate_bandwidth_from_counter(
                perf_data['ocr.demand_rfo.l3_miss'], wall_time
            )
        
        # Calculate efficiency metrics
        if 'cache-references' in perf_data and 'cache-misses' in perf_data:
            bandwidth_metrics['cache_miss_rate'] = (perf_data['cache-misses'] / perf_data['cache-references']) * 100
        
        if 'L1-dcache-loads' in perf_data and 'L1-dcache-load-misses' in perf_data:
            bandwidth_metrics['l1_dcache_miss_rate'] = (perf_data['L1-dcache-load-misses'] / perf_data['L1-dcache-loads']) * 100
        
        if 'instructions' in perf_data and 'cycles' in perf_data:
            bandwidth_metrics['instructions_per_cycle'] = perf_data['instructions'] / perf_data['cycles']
        
        if 'cycle_activity.stalls_total' in perf_data and 'cycles' in perf_data:
            bandwidth_metrics['cycle_stall_rate'] = (perf_data['cycle_activity.stalls_total'] / perf_data['cycles']) * 100
        
        return bandwidth_metrics


class CXLPerfBandwidthTester(SchedulerBenchmark):
    """
    Enhanced CXL benchmark tester with perf counter integration.
    """
    
    def __init__(self, double_bandwidth_path: str, results_dir: str = "results",
                 scheduler_runner: SchedulerRunner = None):
        """
        Initialize the CXLPerfBandwidthTester.
        
        Args:
            double_bandwidth_path: Path to double_bandwidth binary
            results_dir: Directory to store results
            scheduler_runner: SchedulerRunner instance to use
        """
        super().__init__(scheduler_runner)
        
        self.double_bandwidth_path = double_bandwidth_path
        self.results_dir = results_dir
        self.perf_calc = PerfBandwidthCalculator()
        
        # Create results directory
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Default test parameters
        self.test_params = {
            "buffer_size": 1073741824,  # 1GB default  
            "threads": 4,
            "duration": 10,
            "read_ratio": 0.5,
            "timeout": 300,
        }
        
        # Environment setup
        self.env = os.environ.copy()
    
    def set_test_params(self, **kwargs):
        """Update test parameters"""
        self.test_params.update(kwargs)
    
    def _build_perf_command(self, target_cmd: List[str]) -> List[str]:
        """
        Build perf stat command with enhanced counters.
        
        Args:
            target_cmd: Command to run under perf
            
        Returns:
            Complete perf stat command
        """
        perf_cmd = [
            'sudo', 'perf', 'stat',
            '-e', ','.join(self.perf_calc.PERF_COUNTERS)
        ] + target_cmd
        
        return perf_cmd
    
    def _build_double_bandwidth_command(self) -> List[str]:
        """Build double_bandwidth command with current parameters"""
        cmd = [
            self.double_bandwidth_path,
            "--buffer-size", str(self.test_params["buffer_size"]),
            "--threads", str(self.test_params["threads"]),
            "--duration", str(self.test_params["duration"]),
            "--read-ratio", str(self.test_params["read_ratio"]),
            "--json"  # Use JSON output for easier parsing
        ]
        return cmd
    
    def _parse_combined_output(self, stdout: str, stderr: str) -> Dict:
        """
        Parse combined output from perf + double_bandwidth.
        
        Args:
            stdout: Standard output (contains JSON from double_bandwidth)
            stderr: Standard error (contains perf stat output)
            
        Returns:
            Combined results dictionary
        """
        results = {}
        
        # Parse application output (JSON)
        try:
            app_metrics = json.loads(stdout.strip())
            results['application'] = app_metrics
            
            # Rename some fields for compatibility
            if "total_bandwidth_mbps" in app_metrics:
                results['application']['bandwidth_mbps'] = app_metrics["total_bandwidth_mbps"]
            if "test_duration" in app_metrics:
                results['application']['execution_time'] = app_metrics["test_duration"]
                
        except json.JSONDecodeError as e:
            results['application'] = {"error": f"Failed to parse JSON: {e}", "raw_output": stdout}
        
        # Parse perf output
        try:
            perf_data = self.perf_calc.parse_perf_output(stderr)
            results['perf_counters'] = perf_data
            
            # Calculate bandwidth metrics from perf counters
            if perf_data:
                bandwidth_metrics = self.perf_calc.calculate_all_bandwidths(perf_data)
                results['perf_bandwidths'] = bandwidth_metrics
            else:
                results['perf_bandwidths'] = {"error": "No perf data parsed"}
                
        except Exception as e:
            results['perf_counters'] = {"error": f"Failed to parse perf output: {e}"}
            results['perf_bandwidths'] = {"error": str(e)}
        
        # Add test parameters
        results['test_params'] = self.test_params.copy()
        
        return results
    
    def run_perf_bandwidth_benchmark(self, scheduler_name: str = None) -> Dict:
        """
        Run double_bandwidth with perf counters under specified scheduler.
        
        Args:
            scheduler_name: Name of the scheduler to test (None for default)
            
        Returns:
            Dictionary containing combined benchmark and perf results
        """
        print(f"Running perf + double_bandwidth with scheduler: {scheduler_name or 'default'}")
        print(f"Parameters: buffer_size={self.test_params['buffer_size']}, "
              f"threads={self.test_params['threads']}, "
              f"duration={self.test_params['duration']}, "
              f"read_ratio={self.test_params['read_ratio']}")
        
        # Build commands
        target_cmd = self._build_double_bandwidth_command()
        perf_cmd = self._build_perf_command(target_cmd)
        timeout = self.test_params["timeout"]
        
        print(f"Running command: {' '.join(perf_cmd)}")
        
        try:
            if scheduler_name:
                # Run with specific scheduler
                exit_code, stdout, stderr = self.runner.run_command_with_scheduler(
                    scheduler_name, perf_cmd, timeout=timeout, env=self.env
                )
            else:
                # Run with default scheduler
                try:
                    result = subprocess.run(
                        perf_cmd, 
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
            print(f"Warning: perf command exited with code {exit_code}")
            print(f"stderr: {stderr}")
            return {
                "scheduler": scheduler_name or "default",
                "error": stderr or f"Exit code: {exit_code}",
                "exit_code": exit_code,
                "raw_stdout": stdout,
                "raw_stderr": stderr
            }
        
        # Parse combined results
        results = self._parse_combined_output(stdout, stderr)
        results["scheduler"] = scheduler_name or "default"
        results["exit_code"] = exit_code
        
        return results
    
    def run_all_perf_benchmarks(self, production_only: bool = True) -> Dict:
        """
        Run perf bandwidth tests for all schedulers.
        
        Args:
            production_only: Only test production-ready schedulers
            
        Returns:
            Dictionary mapping scheduler names to benchmark results
        """
        results = {}
        
        # Test default scheduler first
        print("Testing default scheduler...")
        results["default"] = self.run_perf_bandwidth_benchmark()
        
        # Test each scheduler
        schedulers = self.runner.get_available_schedulers(production_only)
        for scheduler_name in schedulers:
            try:
                print(f"\nTesting scheduler: {scheduler_name}")
                results[scheduler_name] = self.run_perf_bandwidth_benchmark(scheduler_name)
                
                # Save intermediate results
                self.save_results(results)
                
                # Brief pause between tests
                time.sleep(2)
                
            except Exception as e:
                print(f"Error testing scheduler {scheduler_name}: {e}")
                results[scheduler_name] = {
                    "scheduler": scheduler_name,
                    "error": str(e)
                }
        
        return results
    
    def save_results(self, results: Dict):
        """Save results to JSON file"""
        results_file = os.path.join(self.results_dir, "cxl_perf_bandwidth_results.json")
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {results_file}")
    
    def generate_comparison_figures(self, results: Dict):
        """Generate comprehensive bandwidth comparison figures"""
        
        # Extract data for plotting
        scheduler_data = []
        
        for scheduler_name, result in results.items():
            if "error" in result:
                continue
            
            row = {"scheduler": scheduler_name}
            
            # Application bandwidth
            if "application" in result and "bandwidth_mbps" in result["application"]:
                row["app_bandwidth"] = result["application"]["bandwidth_mbps"]
            else:
                row["app_bandwidth"] = 0
            
            # Perf bandwidth metrics
            if "perf_bandwidths" in result and not "error" in result["perf_bandwidths"]:
                perf_bw = result["perf_bandwidths"]
                row.update({
                    "cache_references_bw": perf_bw.get("cache_references_bw", 0),
                    "cache_misses_bw": perf_bw.get("cache_misses_bw", 0),
                    "longest_lat_cache_miss_bw": perf_bw.get("longest_lat_cache_miss_bw", 0),
                    "l1_dcache_loads_bw": perf_bw.get("l1_dcache_loads_bw", 0),
                    "l1_dcache_load_misses_bw": perf_bw.get("l1_dcache_load_misses_bw", 0),
                    "local_dram_bw": perf_bw.get("local_dram_bw", 0),
                    "remote_dram_bw": perf_bw.get("remote_dram_bw", 0),
                    "cache_miss_rate": perf_bw.get("cache_miss_rate", 0),
                    "l1_dcache_miss_rate": perf_bw.get("l1_dcache_miss_rate", 0),
                    "instructions_per_cycle": perf_bw.get("instructions_per_cycle", 0),
                    "cycle_stall_rate": perf_bw.get("cycle_stall_rate", 0)
                })
            
            scheduler_data.append(row)
        
        if not scheduler_data:
            print("No valid results to plot")
            return
        
        df = pd.DataFrame(scheduler_data)
        
        # Create comprehensive comparison figure
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('CXL Perf Bandwidth Analysis Comparison', fontsize=16, fontweight='bold')
        
        # Plot 1: Main bandwidth comparison
        ax1 = axes[0, 0]
        bandwidth_columns = ['app_bandwidth', 'cache_references_bw', 'cache_misses_bw', 'longest_lat_cache_miss_bw']
        bandwidth_labels = ['Application', 'Cache Refs', 'Cache Misses', 'Longest Lat Miss']
        
        x = np.arange(len(df))
        width = 0.2
        
        for i, (col, label) in enumerate(zip(bandwidth_columns, bandwidth_labels)):
            if col in df.columns:
                ax1.bar(x + i*width, df[col], width, label=label, alpha=0.8)
        
        ax1.set_xlabel('Scheduler')
        ax1.set_ylabel('Bandwidth (MB/s)')
        ax1.set_title('Primary Bandwidth Metrics')
        ax1.set_xticks(x + width*1.5)
        ax1.set_xticklabels(df['scheduler'], rotation=45)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Memory hierarchy bandwidth
        ax2 = axes[0, 1]
        memory_columns = ['l1_dcache_loads_bw', 'l1_dcache_load_misses_bw', 'local_dram_bw', 'remote_dram_bw']
        memory_labels = ['L1 Loads', 'L1 Misses', 'Local DRAM', 'Remote DRAM']
        
        for i, (col, label) in enumerate(zip(memory_columns, memory_labels)):
            if col in df.columns:
                ax2.bar(x + i*width, df[col], width, label=label, alpha=0.8)
        
        ax2.set_xlabel('Scheduler')
        ax2.set_ylabel('Bandwidth (MB/s)')
        ax2.set_title('Memory Hierarchy Bandwidth')
        ax2.set_xticks(x + width*1.5)
        ax2.set_xticklabels(df['scheduler'], rotation=45)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Efficiency metrics
        ax3 = axes[1, 0]
        efficiency_columns = ['cache_miss_rate', 'l1_dcache_miss_rate']
        efficiency_labels = ['Cache Miss Rate (%)', 'L1 Miss Rate (%)']
        
        for i, (col, label) in enumerate(zip(efficiency_columns, efficiency_labels)):
            if col in df.columns:
                ax3.bar(x + i*width, df[col], width, label=label, alpha=0.8)
        
        ax3.set_xlabel('Scheduler')
        ax3.set_ylabel('Miss Rate (%)')
        ax3.set_title('Cache Efficiency Metrics')
        ax3.set_xticks(x + width*0.5)
        ax3.set_xticklabels(df['scheduler'], rotation=45)
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Performance metrics
        ax4 = axes[1, 1]
        if 'instructions_per_cycle' in df.columns and 'cycle_stall_rate' in df.columns:
            ax4_twin = ax4.twinx()
            
            bars1 = ax4.bar(x - width/2, df['instructions_per_cycle'], width, 
                           label='IPC', alpha=0.8, color='skyblue')
            bars2 = ax4_twin.bar(x + width/2, df['cycle_stall_rate'], width, 
                               label='Stall Rate (%)', alpha=0.8, color='orange')
            
            ax4.set_xlabel('Scheduler')
            ax4.set_ylabel('Instructions per Cycle', color='skyblue')
            ax4_twin.set_ylabel('Cycle Stall Rate (%)', color='orange')
            ax4.set_title('CPU Performance Metrics')
            ax4.set_xticks(x)
            ax4.set_xticklabels(df['scheduler'], rotation=45)
            ax4.grid(True, alpha=0.3)
            
            # Add legends
            lines1, labels1 = ax4.get_legend_handles_labels()
            lines2, labels2 = ax4_twin.get_legend_handles_labels()
            ax4.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        
        plt.tight_layout()
        
        # Save figure
        figure_path = os.path.join(self.results_dir, "cxl_perf_bandwidth_comparison.png")
        plt.savefig(figure_path, dpi=300, bbox_inches='tight')
        print(f"Comparison figure saved to {figure_path}")
        
        # Print detailed summary
        self.print_detailed_summary(results)
    
    def print_detailed_summary(self, results: Dict):
        """Print detailed performance summary with perf metrics"""
        print("\n" + "="*80)
        print("CXL PERF BANDWIDTH ANALYSIS SUMMARY")
        print("="*80)
        
        for scheduler_name, result in results.items():
            if "error" in result:
                print(f"\n{scheduler_name:15} ERROR: {result['error']}")
                continue
            
            print(f"\n{scheduler_name:15}")
            print("-" * 60)
            
            # Application metrics
            if "application" in result:
                app = result["application"]
                print(f"  APPLICATION BANDWIDTH:")
                print(f"    Total:           {app.get('bandwidth_mbps', 0):8.1f} MB/s")
                if "execution_time" in app:
                    print(f"    Duration:        {app.get('execution_time', 0):8.2f} seconds")
                if "total_read_bytes" in app and "total_write_bytes" in app:
                    total_gb = (app["total_read_bytes"] + app["total_write_bytes"]) / (1024**3)
                    print(f"    Total Data:      {total_gb:8.2f} GB")
            
            # Perf bandwidth metrics
            if "perf_bandwidths" in result and not "error" in result["perf_bandwidths"]:
                perf_bw = result["perf_bandwidths"]
                print(f"  PERF BANDWIDTH METRICS:")
                print(f"    Cache Refs:      {perf_bw.get('cache_references_bw', 0):8.1f} MB/s")
                print(f"    Cache Misses:    {perf_bw.get('cache_misses_bw', 0):8.1f} MB/s")
                print(f"    Longest Lat:     {perf_bw.get('longest_lat_cache_miss_bw', 0):8.1f} MB/s")
                print(f"    L1 Loads:        {perf_bw.get('l1_dcache_loads_bw', 0):8.1f} MB/s")
                print(f"    L1 Load Misses:  {perf_bw.get('l1_dcache_load_misses_bw', 0):8.1f} MB/s")
                print(f"    Local DRAM:      {perf_bw.get('local_dram_bw', 0):8.1f} MB/s")
                print(f"    Remote DRAM:     {perf_bw.get('remote_dram_bw', 0):8.1f} MB/s")
                
                print(f"  EFFICIENCY METRICS:")
                print(f"    Cache Miss Rate: {perf_bw.get('cache_miss_rate', 0):8.1f} %")
                print(f"    L1 Miss Rate:    {perf_bw.get('l1_dcache_miss_rate', 0):8.1f} %")
                print(f"    IPC:             {perf_bw.get('instructions_per_cycle', 0):8.3f}")
                print(f"    Stall Rate:      {perf_bw.get('cycle_stall_rate', 0):8.1f} %")
                
                # Validation: Compare app bandwidth with cache references
                if "application" in result and "bandwidth_mbps" in result["application"]:
                    app_bw = result["application"]["bandwidth_mbps"]
                    cache_ref_bw = perf_bw.get('cache_references_bw', 0)
                    if cache_ref_bw > 0:
                        correlation = cache_ref_bw / app_bw
                        print(f"  VALIDATION:")
                        print(f"    Cache/App Ratio: {correlation:8.3f} (expect ~1.0)")


def main():
    """Main function for CXL perf bandwidth testing"""
    parser = argparse.ArgumentParser(description="Test schedulers with CXL-micro double_bandwidth and perf counters")
    parser.add_argument("--double-bandwidth-path", 
                       default="/root/yunwei37/ai-os/workloads/cxl-micro/double_bandwidth",
                       help="Path to double_bandwidth binary")
    parser.add_argument("--results-dir", default="results", 
                       help="Directory to store results")
    parser.add_argument("--production-only", action="store_true", 
                       help="Test only production schedulers", default=False)
    parser.add_argument("--threads", type=int, default=4, 
                       help="Number of threads for testing")
    parser.add_argument("--buffer-size", type=int, default=1024*1024*1024, 
                       help="Buffer size in bytes (default 1GB)")
    parser.add_argument("--duration", type=int, default=10, 
                       help="Test duration in seconds")
    parser.add_argument("--read-ratio", type=float, default=0.5,
                       help="Ratio of readers (0.0-1.0, default: 0.5)")
    parser.add_argument("--timeout", type=int, default=300, 
                       help="Timeout in seconds")
    parser.add_argument("--scheduler", type=str, default=None,
                       help="Test specific scheduler only")
    
    args = parser.parse_args()
    
    # Create tester instance
    tester = CXLPerfBandwidthTester(args.double_bandwidth_path, args.results_dir)
    
    # Validate read_ratio
    if args.read_ratio < 0.0 or args.read_ratio > 1.0:
        print("Error: read-ratio must be between 0.0 and 1.0")
        sys.exit(1)
    
    # Update test parameters
    tester.set_test_params(
        threads=args.threads,
        buffer_size=args.buffer_size,
        duration=args.duration,
        timeout=args.timeout,
        read_ratio=args.read_ratio
    )
    
    # Check if binary exists
    if not os.path.exists(args.double_bandwidth_path):
        print(f"Error: double_bandwidth not found at {args.double_bandwidth_path}")
        sys.exit(1)
    
    # Check if we can run perf
    try:
        subprocess.run(['sudo', 'perf', '--version'], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Error: Cannot run 'sudo perf'. Make sure perf is installed and you have sudo access.")
        sys.exit(1)
    
    print("Starting CXL perf bandwidth analysis...")
    print(f"Perf counters: {', '.join(tester.perf_calc.PERF_COUNTERS)}")
    
    if args.scheduler:
        print(f"Testing scheduler: {args.scheduler}")
        result = tester.run_perf_bandwidth_benchmark(args.scheduler)
        results = {args.scheduler: result}
    else:
        print("Testing all schedulers...")
        results = tester.run_all_perf_benchmarks(production_only=args.production_only)
    
    # Generate figures and summary
    tester.generate_comparison_figures(results)
    
    print("\nTesting complete!")


if __name__ == "__main__":
    main()