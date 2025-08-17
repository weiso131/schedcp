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
import argparse
import re
from typing import Dict, List


class PerfBandwidthCalculator:
    """Calculate bandwidth metrics from perf counters using cache line methodology"""
    
    CACHE_LINE_SIZE = 64  # bytes
    MB = 1024 * 1024
    
    # L3 and DRAM focused perf counters
    PERF_COUNTERS = [
        "instructions",
        "cycles",
        # L3 cache performance
        "cache-references",
        "cache-misses",
        "longest_lat_cache.miss",
        # Memory instructions for L3 miss rate calculation
        "mem_inst_retired.all_loads", 
        "mem_inst_retired.all_stores",
        # L3 hit/miss analysis
        "mem_load_retired.l3_hit",
        "mem_load_retired.l3_miss", 
        # L3 miss breakdown by destination
        "mem_load_l3_miss_retired.local_dram",
        "mem_load_l3_miss_retired.remote_dram",
        "mem_load_l3_miss_retired.remote_fwd",
        "mem_load_l3_miss_retired.remote_hitm",
        # Offcore requests (L3 miss traffic)
        "offcore_requests.l3_miss_demand_data_rd",
        "offcore_requests.demand_data_rd",
        "offcore_requests.demand_rfo", 
        # Memory controller counters (actual DRAM bandwidth)
        "uncore_imc/cas_count_read/",
        "uncore_imc/cas_count_write/",
        # Memory stalls
        "cycle_activity.stalls_l3_miss",
        "memory_activity.stalls_l3_miss"
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
        
        # Method 1: L3 Cache Analysis
        if 'cache-references' in perf_data:
            bandwidth_metrics['cache_references_bw'] = cls.calculate_bandwidth_from_counter(
                perf_data['cache-references'], wall_time
            )
        
        if 'cache-misses' in perf_data:
            bandwidth_metrics['cache_misses_bw'] = cls.calculate_bandwidth_from_counter(
                perf_data['cache-misses'], wall_time
            )
        
        # Method 2: L3 Hit/Miss Rates (instruction-level analysis, not bandwidth)
        # Note: These measure individual load instructions, not cache line transfers
        if 'mem_load_retired.l3_hit' in perf_data:
            bandwidth_metrics['l3_hit_rate_ops'] = perf_data['mem_load_retired.l3_hit'] / wall_time
        
        if 'mem_load_retired.l3_miss' in perf_data:
            bandwidth_metrics['l3_miss_rate_ops'] = perf_data['mem_load_retired.l3_miss'] / wall_time
        
        # Method 3: L3 Miss Destination Analysis (instruction rates, not bandwidth)
        # Note: These counters measure individual load instructions, not cache lines
        # So we calculate instruction rates (ops/sec) instead of bandwidth
        if 'mem_load_l3_miss_retired.local_dram' in perf_data:
            bandwidth_metrics['l3_miss_local_dram_rate'] = perf_data['mem_load_l3_miss_retired.local_dram'] / wall_time
        
        if 'mem_load_l3_miss_retired.remote_dram' in perf_data:
            bandwidth_metrics['l3_miss_remote_dram_rate'] = perf_data['mem_load_l3_miss_retired.remote_dram'] / wall_time
        
        if 'mem_load_l3_miss_retired.remote_fwd' in perf_data:
            bandwidth_metrics['l3_miss_remote_fwd_rate'] = perf_data['mem_load_l3_miss_retired.remote_fwd'] / wall_time
        
        if 'mem_load_l3_miss_retired.remote_hitm' in perf_data:
            bandwidth_metrics['l3_miss_remote_hitm_rate'] = perf_data['mem_load_l3_miss_retired.remote_hitm'] / wall_time
        
        # Method 4: Offcore L3 Miss Traffic (cache line level)
        if 'offcore_requests.l3_miss_demand_data_rd' in perf_data:
            bandwidth_metrics['offcore_l3_miss_data_rd_bw'] = cls.calculate_bandwidth_from_counter(
                perf_data['offcore_requests.l3_miss_demand_data_rd'], wall_time
            )
        
        if 'offcore_requests.demand_data_rd' in perf_data:
            bandwidth_metrics['offcore_data_read_bw'] = cls.calculate_bandwidth_from_counter(
                perf_data['offcore_requests.demand_data_rd'], wall_time
            )
        
        if 'offcore_requests.demand_rfo' in perf_data:
            bandwidth_metrics['offcore_rfo_bw'] = cls.calculate_bandwidth_from_counter(
                perf_data['offcore_requests.demand_rfo'], wall_time
            )
        
        # Method 5: Memory Controller Bandwidth (actual DRAM)
        if 'uncore_imc/cas_count_read/' in perf_data:
            bandwidth_metrics['dram_read_bw'] = cls.calculate_bandwidth_from_counter(
                perf_data['uncore_imc/cas_count_read/'], wall_time
            )
        
        if 'uncore_imc/cas_count_write/' in perf_data:
            bandwidth_metrics['dram_write_bw'] = cls.calculate_bandwidth_from_counter(
                perf_data['uncore_imc/cas_count_write/'], wall_time
            )
        
        # Method 6: Memory Instructions for Context
        if 'mem_inst_retired.all_loads' in perf_data and 'mem_inst_retired.all_stores' in perf_data:
            total_mem_ops = perf_data['mem_inst_retired.all_loads'] + perf_data['mem_inst_retired.all_stores']
            bandwidth_metrics['memory_instructions_bw'] = cls.calculate_bandwidth_from_counter(
                total_mem_ops, wall_time
            )
        
        # Calculate efficiency metrics
        if 'cache-references' in perf_data and 'cache-misses' in perf_data:
            bandwidth_metrics['cache_miss_rate'] = (perf_data['cache-misses'] / perf_data['cache-references']) * 100
        
        # Precise L3 Hit/Miss Rates
        if 'mem_load_retired.l3_hit' in perf_data and 'mem_load_retired.l3_miss' in perf_data:
            l3_hits = perf_data['mem_load_retired.l3_hit']
            l3_misses = perf_data['mem_load_retired.l3_miss']
            total_l3_accesses = l3_hits + l3_misses
            if total_l3_accesses > 0:
                bandwidth_metrics['l3_hit_rate'] = (l3_hits / total_l3_accesses) * 100
                bandwidth_metrics['l3_miss_rate_precise'] = (l3_misses / total_l3_accesses) * 100
        
        # L3 Miss Destination Breakdown
        if 'mem_load_l3_miss_retired.local_dram' in perf_data:
            local_dram = perf_data['mem_load_l3_miss_retired.local_dram']
            remote_dram = perf_data.get('mem_load_l3_miss_retired.remote_dram', 0)
            remote_fwd = perf_data.get('mem_load_l3_miss_retired.remote_fwd', 0)
            remote_hitm = perf_data.get('mem_load_l3_miss_retired.remote_hitm', 0)
            
            total_l3_miss_loads = local_dram + remote_dram + remote_fwd + remote_hitm
            if total_l3_miss_loads > 0:
                bandwidth_metrics['l3_miss_local_dram_pct'] = (local_dram / total_l3_miss_loads) * 100
                bandwidth_metrics['l3_miss_remote_dram_pct'] = (remote_dram / total_l3_miss_loads) * 100
                bandwidth_metrics['l3_miss_remote_fwd_pct'] = (remote_fwd / total_l3_miss_loads) * 100
                bandwidth_metrics['l3_miss_remote_hitm_pct'] = (remote_hitm / total_l3_miss_loads) * 100
        
        # Overall L3 Miss Rate (instruction-based)
        if 'mem_inst_retired.all_loads' in perf_data and 'mem_load_l3_miss_retired.local_dram' in perf_data:
            total_loads = perf_data['mem_inst_retired.all_loads']
            l3_misses = perf_data['mem_load_l3_miss_retired.local_dram']
            if 'mem_load_l3_miss_retired.remote_dram' in perf_data:
                l3_misses += perf_data['mem_load_l3_miss_retired.remote_dram']
            bandwidth_metrics['l3_miss_rate'] = (l3_misses / total_loads) * 100
        
        if 'instructions' in perf_data and 'cycles' in perf_data:
            bandwidth_metrics['instructions_per_cycle'] = perf_data['instructions'] / perf_data['cycles']
        
        if 'cycle_activity.stalls_total' in perf_data and 'cycles' in perf_data:
            bandwidth_metrics['cycle_stall_rate'] = (perf_data['cycle_activity.stalls_total'] / perf_data['cycles']) * 100
        
        return bandwidth_metrics


class CXLPerfBandwidthTester:
    """
    Simple CXL benchmark tester with perf counter integration.
    """
    
    def __init__(self, double_bandwidth_path: str, results_dir: str = "results"):
        """
        Initialize the CXLPerfBandwidthTester.
        
        Args:
            double_bandwidth_path: Path to double_bandwidth binary
            results_dir: Directory to store results
        """
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
    
    def run_perf_bandwidth_benchmark(self) -> Dict:
        """
        Run double_bandwidth with perf counters.
        
        Returns:
            Dictionary containing combined benchmark and perf results
        """
        print(f"Running perf + double_bandwidth")
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
        
        if exit_code != 0:
            print(f"Warning: perf command exited with code {exit_code}")
            print(f"stderr: {stderr}")
            return {
                "error": stderr or f"Exit code: {exit_code}",
                "exit_code": exit_code,
                "raw_stdout": stdout,
                "raw_stderr": stderr
            }
        
        # Parse combined results
        results = self._parse_combined_output(stdout, stderr)
        results["exit_code"] = exit_code
        
        return results
    
    def save_results(self, results: Dict):
        """Save results to JSON file"""
        results_file = os.path.join(self.results_dir, "cxl_perf_bandwidth_results.json")
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {results_file}")
    
    def print_raw_data(self, results: Dict):
        """Print raw performance data and results with calculation details"""
        print("\n" + "="*80)
        print("RAW DATA OUTPUT")
        print("="*80)
        
        if "error" in results:
            print(f"ERROR: {results['error']}")
            if "raw_stdout" in results:
                print(f"\nRAW STDOUT:\n{results['raw_stdout']}")
            if "raw_stderr" in results:
                print(f"\nRAW STDERR:\n{results['raw_stderr']}")
            return
        
        # Print calculation methodology first
        print("CALCULATION METHODOLOGY:")
        print("-" * 40)
        print("Cache Line Size: 64 bytes")
        print("Bandwidth Formula: (counter_value * cache_line_size) / wall_time / MB")
        print("Where MB = 1024 * 1024 bytes")
        print()
        
        # Show calculation process if perf data exists
        if "perf_counters" in results and "perf_bandwidths" in results:
            perf_counters = results["perf_counters"]
            perf_bw = results["perf_bandwidths"]
            wall_time = perf_counters.get("wall_time", 0)
            
            print("CALCULATION PROCESS:")
            print("-" * 40)
            print(f"Wall Time: {wall_time:.6f} seconds")
            print()
            
            # Show calculations for L3 and DRAM focused metrics (bandwidth only)
            calculations = [
                ("Cache References BW", "cache-references", "cache_references_bw"),
                ("Cache Misses BW", "cache-misses", "cache_misses_bw"), 
                ("Offcore Data Read BW", "offcore_requests.demand_data_rd", "offcore_data_read_bw"),
                ("Offcore RFO BW", "offcore_requests.demand_rfo", "offcore_rfo_bw"),
                ("Offcore L3 Miss Data Read BW", "offcore_requests.l3_miss_demand_data_rd", "offcore_l3_miss_data_rd_bw"),
                ("DRAM Read BW", "uncore_imc/cas_count_read/", "dram_read_bw"),
                ("DRAM Write BW", "uncore_imc/cas_count_write/", "dram_write_bw")
            ]
            
            for name, counter_key, bw_key in calculations:
                if counter_key in perf_counters and bw_key in perf_bw:
                    counter_val = perf_counters[counter_key]
                    calculated_bw = perf_bw[bw_key]
                    print(f"{name}:")
                    print(f"  Counter Value: {counter_val:,.0f}")
                    print(f"  Calculation: ({counter_val:,.0f} * 64) / {wall_time:.6f} / 1048576")
                    print(f"  Result: {calculated_bw:.2f} MB/s")
                    print()
            
            # L3 Hit/Miss Instruction Rates
            print("L3 HIT/MISS INSTRUCTION RATES:")
            print("-" * 40)
            
            l3_instruction_rates = [
                ("L3 Hit Rate", "mem_load_retired.l3_hit", "l3_hit_rate_ops"),
                ("L3 Miss Rate", "mem_load_retired.l3_miss", "l3_miss_rate_ops")
            ]
            
            for name, counter_key, rate_key in l3_instruction_rates:
                if counter_key in perf_counters and rate_key in perf_bw:
                    counter_val = perf_counters[counter_key]
                    rate = perf_bw[rate_key]
                    print(f"{name}:")
                    print(f"  Counter Value: {counter_val:,.0f} load instructions")
                    print(f"  Calculation: {counter_val:,.0f} / {wall_time:.6f}")
                    print(f"  Result: {rate:,.0f} loads/sec")
                    print()
            
            # L3 Miss Destination Rates (instruction-based, not bandwidth)
            print("L3 MISS DESTINATION RATES (Load Instructions/sec):")
            print("-" * 40)
            
            instruction_rates = [
                ("Local DRAM Load Rate", "mem_load_l3_miss_retired.local_dram", "l3_miss_local_dram_rate"),
                ("Remote DRAM Load Rate", "mem_load_l3_miss_retired.remote_dram", "l3_miss_remote_dram_rate"),
                ("Remote Forward Rate", "mem_load_l3_miss_retired.remote_fwd", "l3_miss_remote_fwd_rate"),
                ("Remote HitM Rate", "mem_load_l3_miss_retired.remote_hitm", "l3_miss_remote_hitm_rate")
            ]
            
            for name, counter_key, rate_key in instruction_rates:
                if counter_key in perf_counters and rate_key in perf_bw:
                    counter_val = perf_counters[counter_key]
                    rate = perf_bw[rate_key]
                    print(f"{name}:")
                    print(f"  Counter Value: {counter_val:,.0f} load instructions")
                    print(f"  Calculation: {counter_val:,.0f} / {wall_time:.6f}")
                    print(f"  Result: {rate:,.0f} loads/sec")
                    print()
            
            # Show memory instructions calculation
            if "mem_inst_retired.all_loads" in perf_counters and "mem_inst_retired.all_stores" in perf_counters:
                loads = perf_counters["mem_inst_retired.all_loads"]
                stores = perf_counters["mem_inst_retired.all_stores"] 
                total_mem_ops = loads + stores
                mem_bw = perf_bw.get("memory_instructions_bw", 0)
                print("Memory Instructions BW:")
                print(f"  All Loads: {loads:,.0f}")
                print(f"  All Stores: {stores:,.0f}")
                print(f"  Total Mem Ops: {total_mem_ops:,.0f}")
                print(f"  Calculation: ({total_mem_ops:,.0f} * 64) / {wall_time:.6f} / 1048576")
                print(f"  Result: {mem_bw:.2f} MB/s")
                print()
            
            # Show efficiency calculations
            print("EFFICIENCY CALCULATIONS:")
            print("-" * 40)
            
            if "cache-references" in perf_counters and "cache-misses" in perf_counters:
                cache_refs = perf_counters["cache-references"]
                cache_misses = perf_counters["cache-misses"]
                miss_rate = perf_bw.get("cache_miss_rate", 0)
                print(f"Cache Miss Rate:")
                print(f"  Cache Misses: {cache_misses:,.0f}")
                print(f"  Cache References: {cache_refs:,.0f}")
                print(f"  Calculation: ({cache_misses:,.0f} / {cache_refs:,.0f}) * 100")
                print(f"  Result: {miss_rate:.2f}%")
                print()
            
            # Precise L3 Hit/Miss Analysis
            if "mem_load_retired.l3_hit" in perf_counters and "mem_load_retired.l3_miss" in perf_counters:
                l3_hits = perf_counters["mem_load_retired.l3_hit"]
                l3_misses = perf_counters["mem_load_retired.l3_miss"]
                total_l3_accesses = l3_hits + l3_misses
                l3_hit_rate = perf_bw.get("l3_hit_rate", 0)
                l3_miss_rate_precise = perf_bw.get("l3_miss_rate_precise", 0)
                print(f"Precise L3 Cache Analysis:")
                print(f"  L3 Hits: {l3_hits:,.0f}")
                print(f"  L3 Misses: {l3_misses:,.0f}")
                print(f"  Total L3 Accesses: {total_l3_accesses:,.0f}")
                print(f"  L3 Hit Rate: ({l3_hits:,.0f} / {total_l3_accesses:,.0f}) * 100 = {l3_hit_rate:.2f}%")
                print(f"  L3 Miss Rate: ({l3_misses:,.0f} / {total_l3_accesses:,.0f}) * 100 = {l3_miss_rate_precise:.2f}%")
                print()
            
            # L3 Miss Destination Breakdown (Load Instructions)
            if "mem_load_l3_miss_retired.local_dram" in perf_counters:
                local_dram = perf_counters["mem_load_l3_miss_retired.local_dram"]
                remote_dram = perf_counters.get("mem_load_l3_miss_retired.remote_dram", 0)
                remote_fwd = perf_counters.get("mem_load_l3_miss_retired.remote_fwd", 0)
                remote_hitm = perf_counters.get("mem_load_l3_miss_retired.remote_hitm", 0)
                total_l3_miss_loads = local_dram + remote_dram + remote_fwd + remote_hitm
                
                local_rate = perf_bw.get('l3_miss_local_dram_rate', 0)
                remote_dram_rate = perf_bw.get('l3_miss_remote_dram_rate', 0)
                remote_fwd_rate = perf_bw.get('l3_miss_remote_fwd_rate', 0)
                remote_hitm_rate = perf_bw.get('l3_miss_remote_hitm_rate', 0)
                
                print(f"L3 Miss Destination Analysis (Load Instructions):")
                print(f"  Local DRAM: {local_dram:,.0f} loads ({perf_bw.get('l3_miss_local_dram_pct', 0):.1f}%) - {local_rate:,.0f} loads/sec")
                print(f"  Remote DRAM: {remote_dram:,.0f} loads ({perf_bw.get('l3_miss_remote_dram_pct', 0):.1f}%) - {remote_dram_rate:,.0f} loads/sec")
                print(f"  Remote Forward: {remote_fwd:,.0f} loads ({perf_bw.get('l3_miss_remote_fwd_pct', 0):.1f}%) - {remote_fwd_rate:,.0f} loads/sec")
                print(f"  Remote HitM: {remote_hitm:,.0f} loads ({perf_bw.get('l3_miss_remote_hitm_pct', 0):.1f}%) - {remote_hitm_rate:,.0f} loads/sec")
                print(f"  Total L3 Miss Loads: {total_l3_miss_loads:,.0f}")
                print()
            
            if "instructions" in perf_counters and "cycles" in perf_counters:
                instructions = perf_counters["instructions"]
                cycles = perf_counters["cycles"]
                ipc = perf_bw.get("instructions_per_cycle", 0)
                print(f"Instructions Per Cycle (IPC):")
                print(f"  Instructions: {instructions:,.0f}")
                print(f"  Cycles: {cycles:,.0f}")
                print(f"  Calculation: {instructions:,.0f} / {cycles:,.0f}")
                print(f"  Result: {ipc:.6f}")
                print()
            
            if "cycle_activity.stalls_total" in perf_counters and "cycles" in perf_counters:
                stalls = perf_counters["cycle_activity.stalls_total"]
                cycles = perf_counters["cycles"]
                stall_rate = perf_bw.get("cycle_stall_rate", 0)
                print(f"Cycle Stall Rate:")
                print(f"  Stalls Total: {stalls:,.0f}")
                print(f"  Cycles: {cycles:,.0f}")
                print(f"  Calculation: ({stalls:,.0f} / {cycles:,.0f}) * 100")
                print(f"  Result: {stall_rate:.2f}%")
                print()
        
        # Print complete JSON results
        print("COMPLETE JSON RESULTS:")
        print("=" * 40)
        print(json.dumps(results, indent=2))
    
    def print_detailed_summary(self, results: Dict):
        """Print detailed performance summary with perf metrics"""
        print("\n" + "="*80)
        print("CXL PERF BANDWIDTH ANALYSIS SUMMARY")
        print("="*80)
        
        if "error" in results:
            print(f"ERROR: {results['error']}")
            return
        
        print("-" * 60)
        
        # Application metrics
        if "application" in results:
            app = results["application"]
            print(f"APPLICATION BANDWIDTH:")
            print(f"  Total:           {app.get('bandwidth_mbps', 0):8.1f} MB/s")
            if "execution_time" in app:
                print(f"  Duration:        {app.get('execution_time', 0):8.2f} seconds")
            if "total_read_bytes" in app and "total_write_bytes" in app:
                total_gb = (app["total_read_bytes"] + app["total_write_bytes"]) / (1024**3)
                print(f"  Total Data:      {total_gb:8.2f} GB")
        
        # Perf bandwidth metrics
        if "perf_bandwidths" in results and not "error" in results["perf_bandwidths"]:
            perf_bw = results["perf_bandwidths"]
            print(f"\nPERF BANDWIDTH METRICS:")
            print(f"  Cache Refs:      {perf_bw.get('cache_references_bw', 0):8.1f} MB/s")
            print(f"  Cache Misses:    {perf_bw.get('cache_misses_bw', 0):8.1f} MB/s")
            print(f"  Offcore All:     {perf_bw.get('offcore_all_requests_bw', 0):8.1f} MB/s")
            print(f"  Offcore Data Rd: {perf_bw.get('offcore_data_read_bw', 0):8.1f} MB/s")
            print(f"  Offcore RFO:     {perf_bw.get('offcore_rfo_bw', 0):8.1f} MB/s")
            print(f"  DRAM Read:       {perf_bw.get('dram_read_bw', 0):8.1f} MB/s")
            print(f"  DRAM Write:      {perf_bw.get('dram_write_bw', 0):8.1f} MB/s")
            print(f"  Local DRAM:      {perf_bw.get('local_dram_bw', 0):8.1f} MB/s")
            print(f"  Remote DRAM:     {perf_bw.get('remote_dram_bw', 0):8.1f} MB/s")
            
            print(f"\nEFFICIENCY METRICS:")
            print(f"  Cache Miss Rate: {perf_bw.get('cache_miss_rate', 0):8.1f} %")
            print(f"  L3 Miss Rate:    {perf_bw.get('l3_miss_rate', 0):8.1f} %")
            print(f"  IPC:             {perf_bw.get('instructions_per_cycle', 0):8.3f}")
            print(f"  Stall Rate:      {perf_bw.get('cycle_stall_rate', 0):8.1f} %")
            
            # Validation: Compare app bandwidth with cache references
            if "application" in results and "bandwidth_mbps" in results["application"]:
                app_bw = results["application"]["bandwidth_mbps"]
                cache_ref_bw = perf_bw.get('cache_references_bw', 0)
                if cache_ref_bw > 0:
                    correlation = cache_ref_bw / app_bw
                    print(f"\nVALIDATION:")
                    print(f"  Cache/App Ratio: {correlation:8.3f} (expect ~1.0)")


def main():
    """Main function for CXL perf bandwidth testing"""
    parser = argparse.ArgumentParser(description="Test CXL-micro double_bandwidth with perf counters")
    parser.add_argument("--double-bandwidth-path", 
                       default="/root/yunwei37/ai-os/workloads/cxl-micro/double_bandwidth",
                       help="Path to double_bandwidth binary")
    parser.add_argument("--results-dir", default="results", 
                       help="Directory to store results")
    parser.add_argument("--threads", type=int, default=4, 
                       help="Number of threads for testing")
    parser.add_argument("--buffer-size", type=int, default=1*1024*1024*1024, 
                       help="Buffer size in bytes (default 256GB)")
    parser.add_argument("--duration", type=int, default=10, 
                       help="Test duration in seconds")
    parser.add_argument("--read-ratio", type=float, default=0.5,
                       help="Ratio of readers (0.0-1.0, default: 0.5)")
    parser.add_argument("--timeout", type=int, default=300, 
                       help="Timeout in seconds")
    parser.add_argument("--raw", action="store_true", default=True,
                       help="Print raw data output instead of summary")
    
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
    
    # Run the benchmark
    results = tester.run_perf_bandwidth_benchmark()
    
    # Save and display results
    tester.save_results(results)
    
    if args.raw:
        tester.print_raw_data(results)
    else:
        tester.print_detailed_summary(results)
    
    print("\nTesting complete!")


if __name__ == "__main__":
    main()