#!/usr/bin/env python3
"""
Redis Benchmark Suite for Scheduler Performance Evaluation
"""
import subprocess
import time
import json
import os
import sys
import csv
import io
from datetime import datetime
import signal
import atexit

class RedisBenchmark:
    def __init__(self, redis_dir="redis-src", results_dir="results", config_options=None):
        self.redis_dir = redis_dir
        self.results_dir = results_dir
        self.redis_cli = os.path.join(redis_dir, "src", "redis-cli")
        self.redis_benchmark = os.path.join(redis_dir, "src", "redis-benchmark")
        self.redis_server = os.path.join(redis_dir, "src", "redis-server")
        self.redis_process = None
        self.config_file = None
        self.config_options = config_options or {}
        
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
        
        # Clean up temporary config file
        if self.config_file and os.path.exists(self.config_file):
            try:
                os.remove(self.config_file)
            except:
                pass
        
    def generate_config(self):
        """Generate Redis configuration file"""
        config_lines = [
            "# Dynamically generated Redis configuration",
            "port 6379",
            "bind 127.0.0.1",
            "daemonize no",
            "loglevel warning",
            "maxclients 10000",
            "timeout 0",
            "tcp-keepalive 300",
            "tcp-backlog 511",
            ""
        ]
        
        # Add custom configuration options
        for key, value in self.config_options.items():
            if key == "io_threads" and value > 1:
                config_lines.append(f"io-threads {value}")
            elif key == "io_threads_do_reads":
                config_lines.append(f"io-threads-do-reads {value}")
            elif key == "threads":
                config_lines.append(f"threads {value}")
            elif key == "maxmemory":
                config_lines.append(f"maxmemory {value}")
            elif key == "maxmemory_policy":
                config_lines.append(f"maxmemory-policy {value}")
            elif key == "hz":
                config_lines.append(f"hz {value}")
            elif key == "client_output_buffer_limit":
                config_lines.append(f"client-output-buffer-limit normal {value}")
            else:
                # Generic key-value pairs
                config_lines.append(f"{key.replace('_', '-')} {value}")
        
        # Write to temporary file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.config_file = f"redis_config_{timestamp}.conf"
        
        with open(self.config_file, 'w') as f:
            f.write('\n'.join(config_lines))
        
        return self.config_file
    
    def start_redis(self):
        """Start Redis server"""
        # First ensure no Redis is running
        self.stop_redis()
        time.sleep(1)
        
        print("Starting Redis server...")
        
        # Generate configuration file
        config_path = self.generate_config()
        
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
    
    def parse_csv_output(self, csv_output):
        """Parse CSV output into structured JSON"""
        lines = csv_output.strip().split('\n')
        if len(lines) < 2:
            return None
        
        # Parse header
        header_line = lines[0].strip('"')
        headers = [h.strip('"') for h in header_line.split('","')]
        
        results = []
        for line in lines[1:]:
            if line.strip():
                # Parse data line
                data_line = line.strip('"')
                values = [v.strip('"') for v in data_line.split('","')]
                
                if len(values) == len(headers):
                    row = {}
                    for i, header in enumerate(headers):
                        value = values[i]
                        # Try to convert numeric values
                        try:
                            if '.' in value:
                                row[header] = float(value)
                            else:
                                row[header] = int(value)
                        except ValueError:
                            row[header] = value
                    results.append(row)
        
        return results
    
    def run_benchmark(self, test_name, command_args):
        """Run a specific benchmark test"""
        print(f"Running {test_name} benchmark...")
        
        start_time = time.time()
        
        # Run benchmark
        cmd = [self.redis_benchmark] + command_args
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        end_time = time.time()
        
        # Parse CSV output if present
        parsed_metrics = None
        if "--csv" in command_args and result.stdout:
            parsed_metrics = self.parse_csv_output(result.stdout)
        
        # Parse results
        return {
            "test_name": test_name,
            "command": " ".join(cmd),
            "stdout": result.stdout,
            "stderr": result.stderr,
            "return_code": result.returncode,
            "duration": end_time - start_time,
            "parsed_metrics": parsed_metrics,
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
    
    def run_comprehensive_benchmark(self, clients=50, requests=100000, data_size=3, pipeline=1, 
                                   keyspace=None, tests=None, threads=None, cluster=False, 
                                   precision=0, seed=None):
        """Run comprehensive Redis benchmarks"""
        if not self.start_redis():
            return None
        
        # Build base arguments
        base_args = ["-n", str(requests), "-c", str(clients), "-P", str(pipeline), "--csv"]
        
        if data_size != 3:
            base_args.extend(["-d", str(data_size)])
        if keyspace:
            base_args.extend(["-r", str(keyspace)])
        if threads:
            base_args.extend(["--threads", str(threads)])
        if cluster:
            base_args.append("--cluster")
        if precision != 0:
            base_args.extend(["--precision", str(precision)])
        if seed:
            base_args.extend(["--seed", str(seed)])
        
        # Define test configurations
        if tests:
            # Use specific tests provided
            test_list = [t.strip() for t in tests.split(',')]
            benchmarks = []
            for test in test_list:
                benchmarks.append({
                    "name": f"{test.upper()}_operations",
                    "args": ["-t", test] + base_args
                })
        else:
            # Use default comprehensive test suite
            benchmarks = [
                {
                    "name": "SET_operations",
                    "args": ["-t", "set"] + base_args
                },
                {
                    "name": "GET_operations", 
                    "args": ["-t", "get"] + base_args
                },
                {
                    "name": "INCR_operations",
                    "args": ["-t", "incr"] + base_args
                },
                {
                    "name": "LPUSH_operations",
                    "args": ["-t", "lpush"] + base_args
                },
                {
                    "name": "LPOP_operations",
                    "args": ["-t", "lpop"] + base_args
                },
                {
                    "name": "SADD_operations",
                    "args": ["-t", "sadd"] + base_args
                },
                {
                    "name": "HSET_operations",
                    "args": ["-t", "hset"] + base_args
                },
                {
                    "name": "mixed_workload",
                    "args": base_args  # No specific test, runs all
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
            "redis_config": self.config_options,
            "test_summary": []
        }
        
        for result in results:
            test_summary = {
                "test_name": result["test_name"],
                "status": "success" if result["return_code"] == 0 else "failed",
                "duration": result["duration"],
                "metrics": result.get("metrics", {}),
                "parsed_metrics": result.get("parsed_metrics", [])
            }
            summary["test_summary"].append(test_summary)
        
        return summary

    def run_quick_benchmark(self, clients=50, requests=10000, data_size=3, pipeline=16, 
                           keyspace=None, tests="set,get", threads=None, cluster=False, 
                           precision=0, seed=None):
        """Run a quick benchmark test"""
        if not self.start_redis():
            return None
        
        try:
            print("\nRunning quick benchmark test...")
            print("-" * 40)
            
            # Build command arguments
            cmd = [self.redis_benchmark, "-t", tests, "-n", str(requests), "-c", str(clients), "-P", str(pipeline), "--csv"]
            
            if data_size != 3:
                cmd.extend(["-d", str(data_size)])
            if keyspace:
                cmd.extend(["-r", str(keyspace)])
            if threads:
                cmd.extend(["--threads", str(threads)])
            if cluster:
                cmd.append("--cluster")
            if precision != 0:
                cmd.extend(["--precision", str(precision)])
            if seed:
                cmd.extend(["--seed", str(seed)])
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                print("Quick benchmark results:")
                print(result.stdout)
                
                # Parse CSV output
                parsed_metrics = self.parse_csv_output(result.stdout)
                metrics = self.parse_benchmark_output(result.stdout)
                
                return {
                    "status": "success",
                    "output": result.stdout,
                    "metrics": metrics,
                    "parsed_metrics": parsed_metrics,
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
    parser.add_argument('--io-threads', type=int, help='Number of I/O threads (1-128)', default=64)
    parser.add_argument('--io-threads-do-reads', choices=['yes', 'no'], help='Enable I/O threads for reads', default='yes')
    parser.add_argument('--maxmemory', help='Maximum memory usage (e.g., 1gb, 512mb)', default='256gb')
    parser.add_argument('--maxmemory-policy', choices=['noeviction', 'allkeys-lru', 'volatile-lru', 'allkeys-random', 'volatile-random', 'volatile-ttl'], help='Memory eviction policy')
    parser.add_argument('--hz', type=int, help='Background task frequency (1-500)', default=100)
    parser.add_argument('--explore-configs', action='store_true', help='Run benchmarks with different configurations')
    
    # Redis benchmark parameters
    parser.add_argument('-c', '--clients', type=int, help='Number of parallel connections', default=50)
    parser.add_argument('-n', '--requests', type=int, help='Total number of requests', default=100000)
    parser.add_argument('-d', '--data-size', type=int, help='Data size of SET/GET value in bytes', default=3)
    parser.add_argument('-P', '--pipeline', type=int, help='Pipeline requests', default=1)
    parser.add_argument('-r', '--keyspace', type=int, help='Use random keys in specified range')
    parser.add_argument('-t', '--tests', help='Specific tests to run (comma-separated: set,get,incr,lpush,lpop,sadd,hset)')
    parser.add_argument('--threads', type=int, help='Enable multi-thread mode')
    parser.add_argument('--cluster', action='store_true', help='Enable cluster mode')
    parser.add_argument('--precision', type=int, help='Decimal places in latency output', default=0)
    parser.add_argument('--seed', type=int, help='Random number generator seed')
    
    args = parser.parse_args()
    
    # Build config options
    config_options = {}
    if args.io_threads:
        config_options['io_threads'] = args.io_threads
    if args.io_threads_do_reads:
        config_options['io_threads_do_reads'] = args.io_threads_do_reads
    if args.maxmemory:
        config_options['maxmemory'] = args.maxmemory
    if args.maxmemory_policy:
        config_options['maxmemory_policy'] = args.maxmemory_policy
    if args.hz:
        config_options['hz'] = args.hz
    
    # Build benchmark parameters
    benchmark_params = {
        'clients': args.clients,
        'requests': args.requests,
        'data_size': args.data_size,
        'pipeline': args.pipeline,
        'precision': args.precision,
        'cluster': args.cluster
    }
    
    # Only add optional parameters if they are specified
    if args.keyspace:
        benchmark_params['keyspace'] = args.keyspace
    if args.tests:
        benchmark_params['tests'] = args.tests
    if args.threads:
        benchmark_params['threads'] = args.threads
    if args.seed:
        benchmark_params['seed'] = args.seed
    
    if args.explore_configs:
        # Test different configurations
        configs_to_test = [
            {},  # Default
            {'io_threads': 4},
            {'io_threads': 8},
            {'io_threads': 4, 'io_threads_do_reads': 'yes'},
            {'hz': 100},
            {'maxmemory': '512mb', 'maxmemory_policy': 'allkeys-lru'}
        ]
        
        for i, config in enumerate(configs_to_test):
            print(f"\n{'='*50}")
            print(f"Testing configuration {i+1}/{len(configs_to_test)}: {config}")
            print('='*50)
            
            benchmark = RedisBenchmark(redis_dir=args.redis_dir, results_dir=args.results_dir, config_options=config)
            result = benchmark.run_quick_benchmark(**benchmark_params)
            if result and result.get('parsed_metrics'):
                print("\nParsed Metrics:")
                for metric in result['parsed_metrics']:
                    print(f"  Test: {metric.get('test', 'N/A')}")
                    print(f"  RPS: {metric.get('rps', 'N/A'):,.0f}")
                    if 'avg_latency_ms' in metric:
                        print(f"  Avg Latency: {metric['avg_latency_ms']:.3f}ms")
                    if 'p99_latency_ms' in metric:
                        print(f"  P99 Latency: {metric['p99_latency_ms']:.3f}ms")
        return
    
    benchmark = RedisBenchmark(redis_dir=args.redis_dir, results_dir=args.results_dir, config_options=config_options)
    
    print("Redis Benchmark Suite")
    print("=" * 50)
    
    if args.quick:
        result = benchmark.run_quick_benchmark(**benchmark_params)
        if result and result.get('parsed_metrics'):
            print("\nParsed Metrics:")
            for metric in result['parsed_metrics']:
                print(f"  Test: {metric.get('test', 'N/A')}")
                print(f"  RPS: {metric.get('rps', 'N/A'):,.0f}")
                if 'avg_latency_ms' in metric:
                    print(f"  Avg Latency: {metric['avg_latency_ms']:.3f}ms")
                if 'p99_latency_ms' in metric:
                    print(f"  P99 Latency: {metric['p99_latency_ms']:.3f}ms")
                print()
    else:
        results = benchmark.run_comprehensive_benchmark(**benchmark_params)
        
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
            if summary.get('redis_config'):
                print(f"Redis config: {summary['redis_config']}")
            
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