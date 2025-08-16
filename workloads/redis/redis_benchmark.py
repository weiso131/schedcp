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
import signal
import atexit

class RedisBenchmark:
    def __init__(self, redis_dir="redis-src", results_dir="results"):
        self.redis_dir = redis_dir
        self.results_dir = results_dir
        self.redis_cli = os.path.join(redis_dir, "src", "redis-cli")
        self.redis_benchmark = os.path.join(redis_dir, "src", "redis-benchmark")
        self.redis_server = os.path.join(redis_dir, "src", "redis-server")
        self.redis_process = None
        
        # Check if binaries exist
        if not all(os.path.exists(binary) for binary in [self.redis_server, self.redis_cli, self.redis_benchmark]):
            print(f"Error: Redis binaries not found in {redis_dir}/src/")
            print("Please run 'make build' first to compile Redis")
            sys.exit(1)
        
        os.makedirs(results_dir, exist_ok=True)
        
        # Register cleanup on exit
        atexit.register(self.cleanup)
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle interrupt signals gracefully"""
        print("\nInterrupt received, cleaning up...")
        self.cleanup()
        sys.exit(0)
    
    def cleanup(self):
        """Ensure Redis is stopped on exit"""
        if hasattr(self, 'redis_process') and self.redis_process:
            self.stop_redis()
        
    def start_redis(self):
        """Start Redis server"""
        # First ensure no Redis is running
        self.stop_redis()
        time.sleep(1)
        
        print("Starting Redis server...")
        
        # Use the simple redis config
        config_path = "redis-simple.conf"
        if not os.path.exists(config_path):
            config_path = "redis.conf"
            if not os.path.exists(config_path):
                # Use default config if custom one doesn't exist
                config_path = os.path.join(self.redis_dir, "redis.conf")
        
        cmd = [self.redis_server, config_path]
        
        self.redis_process = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        
        # Wait for Redis to start with retries
        max_retries = 10
        for i in range(max_retries):
            time.sleep(0.5)
            try:
                result = subprocess.run([self.redis_cli, "ping"], 
                              check=True, capture_output=True, timeout=2, text=True)
                if "PONG" in result.stdout:
                    print("Redis server started successfully")
                    return True
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
                if i == max_retries - 1:
                    print("Failed to start Redis server after multiple attempts")
                    if self.redis_process:
                        stdout, stderr = self.redis_process.communicate(timeout=1)
                        print(f"Redis stdout: {stdout.decode() if stdout else 'None'}")
                        print(f"Redis stderr: {stderr.decode() if stderr else 'None'}")
                    return False
                continue
        return False
    
    def stop_redis(self):
        """Stop Redis server"""
        try:
            # Try graceful shutdown first
            subprocess.run([self.redis_cli, "shutdown", "nosave"], 
                          capture_output=True, timeout=2)
            time.sleep(0.5)
        except:
            pass
        
        # Kill the process if it's still running
        if hasattr(self, 'redis_process') and self.redis_process:
            try:
                self.redis_process.terminate()
                self.redis_process.wait(timeout=2)
            except:
                try:
                    self.redis_process.kill()
                    self.redis_process.wait(timeout=1)
                except:
                    pass
            self.redis_process = None
        
        # Final cleanup - kill any remaining redis-server processes
        try:
            subprocess.run(["pkill", "-f", "redis-server"], capture_output=True, timeout=1)
        except:
            pass
    
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
        
        current_test = None
        for line in lines:
            # Detect test type (e.g., "====== SET ======")
            if line.startswith("======") and "======" in line[6:]:
                current_test = line.replace("=", "").strip()
                continue
            
            # Parse CSV output if present
            if '"' in line and ',' in line:
                try:
                    parts = line.strip().strip('"').split('","')
                    if len(parts) >= 2:
                        test_name = parts[0]
                        rps = float(parts[1])
                        if test_name not in metrics:
                            metrics[test_name] = {}
                        metrics[test_name]['requests_per_second'] = rps
                except:
                    pass
            
            # Parse regular output
            elif 'requests per second' in line.lower():
                # Extract RPS
                parts = line.split()
                for i, part in enumerate(parts):
                    cleaned = part.replace(',', '').replace('.', '', part.count('.') - 1)
                    try:
                        rps = float(cleaned)
                        if current_test:
                            if current_test not in metrics:
                                metrics[current_test] = {}
                            metrics[current_test]['requests_per_second'] = rps
                        else:
                            metrics['requests_per_second'] = rps
                        break
                    except:
                        continue
            
            elif 'latency summary' in line.lower():
                current_test = "latency"
            
            # Parse percentile latencies
            elif '%' in line and ('percentile' in line.lower() or 'latency' in line.lower()):
                percentiles = ['50.00', '95.00', '99.00', '99.90']
                for p in percentiles:
                    if p + '%' in line:
                        latency = self.extract_latency(line)
                        if latency > 0:
                            key = f'latency_p{p.split(".")[0]}'
                            if current_test and current_test != "latency":
                                if current_test not in metrics:
                                    metrics[current_test] = {}
                                metrics[current_test][key] = latency
                            else:
                                metrics[key] = latency
        
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

    def run_quick_benchmark(self):
        """Run a quick benchmark test"""
        if not self.start_redis():
            return None
        
        try:
            print("\nRunning quick benchmark test...")
            print("-" * 40)
            
            # Simple SET/GET benchmark
            cmd = [
                self.redis_benchmark,
                "-t", "set,get",
                "-n", "10000",
                "-c", "50",
                "-P", "16",
                "-q"
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                print("Quick benchmark results:")
                print(result.stdout)
                
                metrics = self.parse_benchmark_output(result.stdout)
                return {
                    "status": "success",
                    "output": result.stdout,
                    "metrics": metrics,
                    "timestamp": datetime.now().isoformat()
                }
            else:
                print(f"Benchmark failed: {result.stderr}")
                return None
                
        except subprocess.TimeoutExpired:
            print("Benchmark timed out")
            return None
        except Exception as e:
            print(f"Error during benchmark: {e}")
            return None
        finally:
            self.stop_redis()

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Redis Benchmark Suite')
    parser.add_argument('--quick', action='store_true', help='Run quick benchmark only')
    parser.add_argument('--redis-dir', default='redis-src', help='Redis source directory')
    parser.add_argument('--results-dir', default='results', help='Results output directory')
    args = parser.parse_args()
    
    benchmark = RedisBenchmark(redis_dir=args.redis_dir, results_dir=args.results_dir)
    
    print("Redis Benchmark Suite")
    print("=" * 50)
    
    if args.quick:
        result = benchmark.run_quick_benchmark()
        if result and result.get('metrics'):
            print("\nParsed Metrics:")
            for key, value in result['metrics'].items():
                if isinstance(value, dict):
                    print(f"  {key}:")
                    for k, v in value.items():
                        print(f"    {k}: {v}")
                else:
                    print(f"  {key}: {value}")
    else:
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
                
                metrics = test.get('metrics', {})
                if isinstance(metrics, dict):
                    for key, value in metrics.items():
                        if isinstance(value, dict):
                            for k, v in value.items():
                                print(f"      {k}: {v}")
                        elif 'requests_per_second' in key:
                            print(f"    RPS: {value:,.0f}")
                        elif 'latency' in key:
                            print(f"    {key}: {value:.2f}ms")
        else:
            print("Failed to run benchmarks")
            sys.exit(1)

if __name__ == "__main__":
    main()