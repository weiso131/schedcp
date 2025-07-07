#!/usr/bin/env python3
"""
Scheduler Performance Testing Script using schbench
Tests all available schedulers in the ai-os project and generates performance figures.
"""

import os
import sys
import subprocess
import time
import json
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Tuple
import argparse

class SchedulerTester:
    def __init__(self, schbench_path: str, results_dir: str = "results"):
        self.schbench_path = schbench_path
        self.results_dir = results_dir
        self.scheduler_bin_path = "/home/yunwei37/ai-os/scheduler/sche_bin"
        self.scheduler_config_path = "/home/yunwei37/ai-os/scheduler/schedulers.json"
        
        # Create results directory
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Load scheduler configurations
        self.schedulers = self._load_scheduler_config()
        
        # Default test parameters
        self.test_params = {
            "message_threads": 2,
            "message_groups": 4,
            "runtime": 30,  # seconds
            "sleeptime": 10000,  # microseconds
            "operations": 5,   # operations per request
        }
    
    def _load_scheduler_config(self) -> Dict:
        """Load scheduler configuration from JSON file"""
        try:
            with open(self.scheduler_config_path, 'r') as f:
                config = json.load(f)
                # Extract schedulers from nested structure and convert to expected format
                schedulers = {}
                for name, info in config.get("schedulers", {}).items():
                    # Convert the config format to expected format with binary field
                    schedulers[name] = {
                        "binary": name,  # Use scheduler name as binary name
                        "production": info.get("production_ready", False)
                    }
                return schedulers
        except FileNotFoundError:
            print(f"Warning: Could not find scheduler config at {self.scheduler_config_path}")
            return self._get_default_schedulers()
    
    def _get_default_schedulers(self) -> Dict:
        """Fallback scheduler list if config file is not available"""
        return {
            "scx_simple": {"binary": "scx_simple", "production": True},
            "scx_rusty": {"binary": "scx_rusty", "production": True},
            "scx_bpfland": {"binary": "scx_bpfland", "production": True},
            "scx_flash": {"binary": "scx_flash", "production": True},
            "scx_lavd": {"binary": "scx_lavd", "production": True},
            "scx_layered": {"binary": "scx_layered", "production": True},
            "scx_nest": {"binary": "scx_nest", "production": True},
            "scx_p2dq": {"binary": "scx_p2dq", "production": True},
            "scx_flatcg": {"binary": "scx_flatcg", "production": True},
        }
    
    def _run_command(self, cmd: List[str], timeout: int = 60) -> Tuple[int, str, str]:
        """Run a command and return exit code, stdout, stderr"""
        try:
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True, 
                timeout=timeout
            )
            return result.returncode, result.stdout, result.stderr
        except subprocess.TimeoutExpired:
            return -1, "", "Command timed out"
        except Exception as e:
            return -1, "", str(e)
    
    def _start_scheduler(self, scheduler_name: str) -> subprocess.Popen:
        """Start a scheduler process"""
        scheduler_info = self.schedulers.get(scheduler_name)
        if not scheduler_info:
            raise ValueError(f"Unknown scheduler: {scheduler_name}")
        
        binary_path = os.path.join(self.scheduler_bin_path, scheduler_info["binary"])
        
        # Check if binary exists
        if not os.path.exists(binary_path):
            raise FileNotFoundError(f"Scheduler binary not found: {binary_path}")
        
        # Start scheduler process
        try:
            proc = subprocess.Popen(
                [binary_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # Give scheduler time to initialize
            time.sleep(2)
            
            # Check if process is still running
            if proc.poll() is not None:
                stdout, stderr = proc.communicate()
                raise RuntimeError(f"Scheduler failed to start: {stderr}")
            
            return proc
        except Exception as e:
            raise RuntimeError(f"Failed to start scheduler {scheduler_name}: {e}")
    
    def _stop_scheduler(self, proc: subprocess.Popen):
        """Stop a scheduler process"""
        if proc and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
    
    def _parse_schbench_output(self, output: str) -> Dict:
        """Parse schbench output to extract performance metrics"""
        results = {
            "latency_percentiles": {},
            "throughput": 0,
            "cpu_utilization": 0,
            "raw_output": output
        }
        
        lines = output.split('\n')
        parsing_request_latencies = False
        
        for line in lines:
            line = line.strip()
            
            # Check if we're in the Request Latencies section
            if "Request Latencies percentiles" in line:
                parsing_request_latencies = True
                continue
            elif "RPS percentiles" in line or "Wakeup Latencies percentiles" in line:
                parsing_request_latencies = False
                continue
            
            # Parse request latency percentiles (these are the main latency metrics)
            if parsing_request_latencies and "th:" in line:
                parts = line.split()
                for i, part in enumerate(parts):
                    if part.endswith("th:"):
                        percentile = part[:-3]
                        if i + 1 < len(parts):
                            value_str = parts[i + 1]
                            try:
                                # Convert to microseconds if needed
                                value = float(value_str)
                                results["latency_percentiles"][percentile] = value
                            except ValueError:
                                pass
            
            # Parse throughput (average rps)
            elif "average rps:" in line:
                parts = line.split()
                for i, part in enumerate(parts):
                    if part == "rps:" and i + 1 < len(parts):
                        try:
                            results["throughput"] = float(parts[i + 1])
                        except ValueError:
                            pass
        
        return results
    
    def _run_schbench_test(self, scheduler_name: str = None) -> Dict:
        """Run schbench test with specified scheduler"""
        print(f"Running schbench test with scheduler: {scheduler_name or 'default'}")
        
        scheduler_proc = None
        try:
            # Start scheduler if specified
            if scheduler_name:
                scheduler_proc = self._start_scheduler(scheduler_name)
            
            # Build schbench command
            cmd = [
                self.schbench_path,
                "-m", str(self.test_params["message_threads"]),
                "-t", str(self.test_params["message_groups"]),
                "-r", str(self.test_params["runtime"]),
                "-s", str(self.test_params["sleeptime"]),
                "-n", str(self.test_params["operations"]),
            ]
            
            # Run schbench
            exit_code, stdout, stderr = self._run_command(cmd, timeout=self.test_params["runtime"] + 30)
            
            if exit_code != 0:
                print(f"Warning: schbench exited with code {exit_code}")
                print(f"stderr: {stderr}")
            
            # Parse results (schbench outputs to stderr)
            output_to_parse = stderr if stderr.strip() else stdout
            results = self._parse_schbench_output(output_to_parse)
            results["scheduler"] = scheduler_name or "default"
            results["exit_code"] = exit_code
            
            return results
            
        finally:
            # Stop scheduler
            if scheduler_proc:
                self._stop_scheduler(scheduler_proc)
    
    def run_all_tests(self, test_production_only: bool = True) -> Dict:
        """Run tests for all schedulers"""
        results = {}
        
        # Test default scheduler first
        print("Testing default scheduler...")
        results["default"] = self._run_schbench_test()
        
        # Test each scheduler
        for scheduler_name, scheduler_info in self.schedulers.items():
            if test_production_only and not scheduler_info.get("production", False):
                print(f"Skipping non-production scheduler: {scheduler_name}")
                continue
            
            try:
                print(f"Testing scheduler: {scheduler_name}")
                results[scheduler_name] = self._run_schbench_test(scheduler_name)
                
                # Save intermediate results
                self._save_results(results)
                
                # Brief pause between tests
                time.sleep(2)
                
            except Exception as e:
                print(f"Error testing scheduler {scheduler_name}: {e}")
                results[scheduler_name] = {
                    "scheduler": scheduler_name,
                    "error": str(e),
                    "latency_percentiles": {},
                    "throughput": 0
                }
        
        return results
    
    def _save_results(self, results: Dict):
        """Save results to JSON file"""
        results_file = os.path.join(self.results_dir, "scheduler_test_results.json")
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {results_file}")
    
    def generate_figures(self, results: Dict):
        """Generate performance comparison figures"""
        
        # Extract data for plotting
        schedulers = []
        latency_50th = []
        latency_95th = []
        latency_99th = []
        throughput = []
        
        for scheduler_name, result in results.items():
            if "error" in result:
                continue
            
            schedulers.append(scheduler_name)
            
            # Get latency percentiles
            percentiles = result.get("latency_percentiles", {})
            latency_50th.append(percentiles.get("50.0", 0))
            latency_95th.append(percentiles.get("95.0", 0))
            latency_99th.append(percentiles.get("99.0", 0))
            
            # Get throughput
            throughput.append(result.get("throughput", 0))
        
        if not schedulers:
            print("No valid results to plot")
            return
        
        # Create figure with subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Scheduler Performance Comparison with schbench', fontsize=16)
        
        # Plot 1: Latency Percentiles
        x = np.arange(len(schedulers))
        width = 0.25
        
        ax1.bar(x - width, latency_50th, width, label='50th percentile', alpha=0.8)
        ax1.bar(x, latency_95th, width, label='95th percentile', alpha=0.8)
        ax1.bar(x + width, latency_99th, width, label='99th percentile', alpha=0.8)
        
        ax1.set_xlabel('Scheduler')
        ax1.set_ylabel('Latency (us)')
        ax1.set_title('Latency Percentiles')
        ax1.set_xticks(x)
        ax1.set_xticklabels(schedulers, rotation=45, ha='right')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Throughput
        ax2.bar(schedulers, throughput, color='skyblue', alpha=0.8)
        ax2.set_xlabel('Scheduler')
        ax2.set_ylabel('Throughput (requests/sec)')
        ax2.set_title('Throughput Comparison')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: 95th Percentile Latency vs Throughput
        ax3.scatter(throughput, latency_95th, s=100, alpha=0.7)
        for i, scheduler in enumerate(schedulers):
            ax3.annotate(scheduler, (throughput[i], latency_95th[i]), 
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
        ax3.set_xlabel('Throughput (requests/sec)')
        ax3.set_ylabel('95th Percentile Latency (us)')
        ax3.set_title('Latency vs Throughput')
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Performance Summary (Normalized)
        if len(schedulers) > 1:
            # Normalize metrics (higher is better for throughput, lower is better for latency)
            norm_throughput = np.array(throughput) / max(throughput) if max(throughput) > 0 else np.zeros(len(throughput))
            norm_latency = (max(latency_95th) - np.array(latency_95th)) / max(latency_95th) if max(latency_95th) > 0 else np.zeros(len(latency_95th))
            
            # Combined score (simple average)
            combined_score = (norm_throughput + norm_latency) / 2
            
            bars = ax4.bar(schedulers, combined_score, color='lightgreen', alpha=0.8)
            ax4.set_xlabel('Scheduler')
            ax4.set_ylabel('Combined Performance Score')
            ax4.set_title('Overall Performance Score\n(Higher is Better)')
            ax4.tick_params(axis='x', rotation=45)
            ax4.grid(True, alpha=0.3)
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax4.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.2f}',
                        ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        
        # Save figure
        figure_path = os.path.join(self.results_dir, "scheduler_performance_comparison.png")
        plt.savefig(figure_path, dpi=300, bbox_inches='tight')
        print(f"Performance figure saved to {figure_path}")
        
       
        # Print summary
        self._print_summary(results)
    
    def _print_summary(self, results: Dict):
        """Print performance summary"""
        print("\n" + "="*60)
        print("SCHEDULER PERFORMANCE SUMMARY")
        print("="*60)
        
        for scheduler_name, result in results.items():
            if "error" in result:
                print(f"\n{scheduler_name:15} ERROR: {result['error']}")
                continue
            
            print(f"\n{scheduler_name:15}")
            print(f"  Throughput:     {result.get('throughput', 0):8.1f} requests/sec")
            
            percentiles = result.get("latency_percentiles", {})
            print(f"  50th percentile: {percentiles.get('50.0', 0):7.1f} us")
            print(f"  95th percentile: {percentiles.get('95.0', 0):7.1f} us")
            print(f"  99th percentile: {percentiles.get('99.0', 0):7.1f} us")

def main():
    parser = argparse.ArgumentParser(description="Test schedulers with schbench")
    parser.add_argument("--schbench-path", default="../schbench/schbench", 
                       help="Path to schbench binary")
    parser.add_argument("--results-dir", default="results", 
                       help="Directory to store results")
    parser.add_argument("--production-only", action="store_true", 
                       help="Test only production schedulers")
    parser.add_argument("--runtime", type=int, default=30, 
                       help="Test runtime in seconds")
    parser.add_argument("--message-threads", type=int, default=2, 
                       help="Number of message threads")
    parser.add_argument("--message-groups", type=int, default=4, 
                       help="Number of message groups")
    
    args = parser.parse_args()
    
    # Create tester instance
    tester = SchedulerTester(args.schbench_path, args.results_dir)
    
    # Update test parameters
    tester.test_params.update({
        "runtime": args.runtime,
        "message_threads": args.message_threads,
        "message_groups": args.message_groups,
    })
    
    # Check if schbench exists
    if not os.path.exists(args.schbench_path):
        print(f"Error: schbench not found at {args.schbench_path}")
        print("Please build schbench first or specify correct path with --schbench-path")
        sys.exit(1)
    
    # Run tests
    print("Starting scheduler performance tests...")
    results = tester.run_all_tests(test_production_only=args.production_only)
    
    # Generate figures
    tester.generate_figures(results)
    
    print("\nTesting complete!")

if __name__ == "__main__":
    main()