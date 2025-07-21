#!/usr/bin/env python3
"""
Redis Benchmark Suite for Scheduler Performance Evaluation
"""
import subprocess
import time
import json
import psutil
import os
import sys
from datetime import datetime

class RedisBenchmark:
    def __init__(self, redis_dir="redis", results_dir="results"):
        self.redis_dir = redis_dir
        self.results_dir = results_dir
        self.redis_cli = os.path.join(redis_dir, "src", "redis-cli")
        self.redis_benchmark = os.path.join(redis_dir, "src", "redis-benchmark")
        self.redis_server = os.path.join(redis_dir, "src", "redis-server")
        
        os.makedirs(results_dir, exist_ok=True)
        
    def start_redis(self):
        """Start Redis server"""
        print("Starting Redis server...")
        config_path = os.path.join(self.redis_dir, "redis.conf")
        cmd = [self.redis_server, config_path]
        
        self.redis_process = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        time.sleep(2)
        
        # Check if Redis is running
        try:
            subprocess.run([self.redis_cli, "ping"], 
                          check=True, capture_output=True, timeout=5)
            print("Redis server started successfully")
            return True
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
            print("Failed to start Redis server")
            return False
    
    def stop_redis(self):
        """Stop Redis server"""
        try:
            subprocess.run([self.redis_cli, "shutdown"], 
                          capture_output=True, timeout=5)
        except:
            pass
        
        if hasattr(self, 'redis_process'):
            self.redis_process.terminate()
            self.redis_process.wait(timeout=5)
    
    def run_benchmark(self, test_name, command_args):
        """Run a specific benchmark test"""
        print(f"Running {test_name} benchmark...")
        
        # Start monitoring
        start_time = time.time()
        start_cpu = psutil.cpu_percent(interval=1)
        start_mem = psutil.virtual_memory().percent
        
        # Run benchmark
        cmd = [self.redis_benchmark] + command_args
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        # End monitoring
        end_time = time.time()
        end_cpu = psutil.cpu_percent(interval=1)
        end_mem = psutil.virtual_memory().percent
        
        # Parse results
        return {
            "test_name": test_name,
            "command": " ".join(cmd),
            "stdout": result.stdout,
            "stderr": result.stderr,
            "return_code": result.returncode,
            "duration": end_time - start_time,
            "cpu_usage": {
                "start": start_cpu,
                "end": end_cpu,
                "average": (start_cpu + end_cpu) / 2
            },
            "memory_usage": {
                "start": start_mem,
                "end": end_mem,
                "average": (start_mem + end_mem) / 2
            },
            "timestamp": datetime.now().isoformat()
        }
    
    def parse_benchmark_output(self, output):
        """Parse Redis benchmark output for metrics"""
        metrics = {}
        lines = output.split('\n')
        
        for line in lines:
            if 'requests per second' in line.lower():
                # Extract RPS
                parts = line.split()
                for i, part in enumerate(parts):
                    if part.replace('.', '').replace(',', '').isdigit():
                        metrics['requests_per_second'] = float(part.replace(',', ''))
                        break
            
            elif 'latency' in line.lower() and 'percentile' in line.lower():
                # Extract latency percentiles
                if '50.00%' in line:
                    metrics['latency_p50'] = self.extract_latency(line)
                elif '95.00%' in line:
                    metrics['latency_p95'] = self.extract_latency(line)
                elif '99.00%' in line:
                    metrics['latency_p99'] = self.extract_latency(line)
        
        return metrics
    
    def extract_latency(self, line):
        """Extract latency value from line"""
        parts = line.split()
        for part in parts:
            if part.replace('.', '').isdigit():
                return float(part)
        return 0.0
    
    def run_comprehensive_benchmark(self):
        """Run comprehensive Redis benchmarks"""
        if not self.start_redis():
            return None
        
        benchmarks = [
            {
                "name": "SET_operations",
                "args": ["-t", "set", "-n", "100000", "-c", "50", "-P", "16", "--csv"]
            },
            {
                "name": "GET_operations", 
                "args": ["-t", "get", "-n", "100000", "-c", "50", "-P", "16", "--csv"]
            },
            {
                "name": "INCR_operations",
                "args": ["-t", "incr", "-n", "50000", "-c", "25", "-P", "16", "--csv"]
            },
            {
                "name": "LPUSH_operations",
                "args": ["-t", "lpush", "-n", "50000", "-c", "25", "-P", "16", "--csv"]
            },
            {
                "name": "LPOP_operations",
                "args": ["-t", "lpop", "-n", "50000", "-c", "25", "-P", "16", "--csv"]
            },
            {
                "name": "SADD_operations",
                "args": ["-t", "sadd", "-n", "50000", "-c", "25", "-P", "16", "--csv"]
            },
            {
                "name": "HSET_operations",
                "args": ["-t", "hset", "-n", "50000", "-c", "25", "-P", "16", "--csv"]
            },
            {
                "name": "mixed_workload",
                "args": ["-n", "100000", "-c", "50", "-P", "16", "--csv"]
            }
        ]
        
        results = []
        
        try:
            for benchmark in benchmarks:
                result = self.run_benchmark(benchmark["name"], benchmark["args"])
                
                # Parse metrics from output
                if result["return_code"] == 0:
                    metrics = self.parse_benchmark_output(result["stdout"])
                    result["metrics"] = metrics
                
                results.append(result)
                time.sleep(1)  # Brief pause between tests
                
        finally:
            self.stop_redis()
        
        return results
    
    def save_results(self, results):
        """Save benchmark results to JSON file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"redis_benchmark_{timestamp}.json"
        filepath = os.path.join(self.results_dir, filename)
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Results saved to {filepath}")
        return filepath
    
    def generate_summary(self, results):
        """Generate benchmark summary"""
        summary = {
            "total_tests": len(results),
            "successful_tests": sum(1 for r in results if r["return_code"] == 0),
            "failed_tests": sum(1 for r in results if r["return_code"] != 0),
            "total_duration": sum(r["duration"] for r in results),
            "average_cpu_usage": sum(r["cpu_usage"]["average"] for r in results) / len(results),
            "average_memory_usage": sum(r["memory_usage"]["average"] for r in results) / len(results),
            "test_summary": []
        }
        
        for result in results:
            test_summary = {
                "test_name": result["test_name"],
                "status": "success" if result["return_code"] == 0 else "failed",
                "duration": result["duration"],
                "metrics": result.get("metrics", {})
            }
            summary["test_summary"].append(test_summary)
        
        return summary

def main():
    benchmark = RedisBenchmark()
    
    print("Starting Redis benchmark suite...")
    print("=" * 50)
    
    results = benchmark.run_comprehensive_benchmark()
    
    if results:
        # Save detailed results
        results_file = benchmark.save_results(results)
        
        # Generate and display summary
        summary = benchmark.generate_summary(results)
        
        print("\n" + "=" * 50)
        print("BENCHMARK SUMMARY")
        print("=" * 50)
        print(f"Total tests: {summary['total_tests']}")
        print(f"Successful: {summary['successful_tests']}")
        print(f"Failed: {summary['failed_tests']}")
        print(f"Total duration: {summary['total_duration']:.2f} seconds")
        print(f"Average CPU usage: {summary['average_cpu_usage']:.2f}%")
        print(f"Average memory usage: {summary['average_memory_usage']:.2f}%")
        
        print("\nTest Results:")
        for test in summary['test_summary']:
            status_symbol = "✓" if test['status'] == 'success' else "✗"
            print(f"  {status_symbol} {test['test_name']}: {test['duration']:.2f}s")
            
            if 'requests_per_second' in test['metrics']:
                print(f"    RPS: {test['metrics']['requests_per_second']:,.0f}")
            if 'latency_p95' in test['metrics']:
                print(f"    P95 Latency: {test['metrics']['latency_p95']:.2f}ms")
    
    else:
        print("Failed to run benchmarks")
        sys.exit(1)

if __name__ == "__main__":
    main()