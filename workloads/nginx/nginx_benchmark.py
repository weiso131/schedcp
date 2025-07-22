#!/usr/bin/env python3
"""
Nginx Benchmark Suite using wrk2 for Scheduler Performance Evaluation
"""
import subprocess
import time
import json
import psutil
import os
import sys
import re
from datetime import datetime

class NginxBenchmark:
    def __init__(self, nginx_dir="nginx", wrk2_dir="wrk2", results_dir="results"):
        self.nginx_dir = nginx_dir
        self.wrk2_dir = wrk2_dir
        self.results_dir = results_dir
        self.nginx_binary = "./nginx-bin"
        self.nginx_config = os.path.abspath("nginx-local.conf")
        self.wrk2_binary = os.path.join(wrk2_dir, "wrk")
        
        os.makedirs(results_dir, exist_ok=True)
        
    def start_nginx(self):
        """Start Nginx server"""
        print("Starting Nginx server...")
        cmd = [self.nginx_binary, "-c", self.nginx_config]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            if result.returncode != 0:
                print(f"Nginx failed to start. Return code: {result.returncode}")
                print(f"Stderr: {result.stderr}")
                return False
                
            time.sleep(2)
            
            # Check if Nginx is running
            health_result = subprocess.run(["curl", "-s", "-o", "/dev/null", "-w", "%{http_code}", 
                                   "http://127.0.0.1:8080/"], 
                                  capture_output=True, text=True, timeout=5)
            
            if health_result.stdout == "200":
                print("Nginx server started successfully")
                return True
            else:
                print(f"Nginx health check failed: HTTP {health_result.stdout}")
                return False
                
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
            print(f"Failed to start Nginx server: {e}")
            return False
    
    def stop_nginx(self):
        """Stop Nginx server"""
        try:
            subprocess.run([self.nginx_binary, "-s", "quit", "-c", self.nginx_config], 
                          capture_output=True, timeout=5)
        except:
            pass
        
        # Force kill if still running
        subprocess.run(["pkill", "-f", "nginx"], capture_output=True)
    
    def run_wrk2_benchmark(self, test_name, threads, connections, duration, rate, url="http://127.0.0.1:8080/"):
        """Run a wrk2 benchmark test"""
        print(f"Running {test_name} benchmark...")
        
        # Start monitoring
        start_time = time.time()
        start_cpu = psutil.cpu_percent(interval=1)
        start_mem = psutil.virtual_memory().percent
        
        # Run wrk2 benchmark
        cmd = [
            self.wrk2_binary,
            f"-t{threads}",
            f"-c{connections}", 
            f"-d{duration}s",
            f"-R{rate}",
            "--latency",
            url
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        # End monitoring
        end_time = time.time()
        end_cpu = psutil.cpu_percent(interval=1)
        end_mem = psutil.virtual_memory().percent
        
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
    
    def parse_wrk2_output(self, output):
        """Parse wrk2 output for metrics"""
        metrics = {}
        
        # Parse requests per second
        rps_match = re.search(r'Requests/sec:\s+(\d+\.?\d*)', output)
        if rps_match:
            metrics['requests_per_second'] = float(rps_match.group(1))
        
        # Parse transfer rate
        transfer_match = re.search(r'Transfer/sec:\s+(\d+\.?\d*)', output)
        if transfer_match:
            metrics['transfer_per_second'] = float(transfer_match.group(1))
        
        # Parse latency statistics
        latency_patterns = [
            (r'50.000%\s+(\d+\.?\d*)(us|ms)', 'latency_p50'),
            (r'75.000%\s+(\d+\.?\d*)(us|ms)', 'latency_p75'),
            (r'90.000%\s+(\d+\.?\d*)(us|ms)', 'latency_p90'),
            (r'99.000%\s+(\d+\.?\d*)(us|ms)', 'latency_p99'),
            (r'99.900%\s+(\d+\.?\d*)(us|ms)', 'latency_p999'),
        ]
        
        for pattern, key in latency_patterns:
            match = re.search(pattern, output)
            if match:
                value = float(match.group(1))
                unit = match.group(2)
                # Convert to milliseconds
                if unit == 'us':
                    value = value / 1000
                metrics[key] = value
        
        # Parse total requests
        total_requests_match = re.search(r'(\d+) requests in', output)
        if total_requests_match:
            metrics['total_requests'] = int(total_requests_match.group(1))
        
        # Parse errors
        error_patterns = [
            (r'Socket errors: connect (\d+)', 'connect_errors'),
            (r'read (\d+)', 'read_errors'),
            (r'write (\d+)', 'write_errors'),
            (r'timeout (\d+)', 'timeout_errors'),
        ]
        
        for pattern, key in error_patterns:
            match = re.search(pattern, output)
            if match:
                metrics[key] = int(match.group(1))
        
        return metrics
    
    def run_comprehensive_benchmark(self):
        """Run comprehensive Nginx benchmarks"""
        if not self.start_nginx():
            return None
        
        benchmarks = [
            {
                "name": "low_load_test",
                "threads": 2,
                "connections": 10,
                "duration": 30,
                "rate": 100
            },
            {
                "name": "medium_load_test",
                "threads": 4,
                "connections": 50,
                "duration": 30,
                "rate": 1000
            },
            {
                "name": "high_load_test",
                "threads": 8,
                "connections": 100,
                "duration": 30,
                "rate": 5000
            },
            {
                "name": "stress_test",
                "threads": 12,
                "connections": 200,
                "duration": 30,
                "rate": 10000
            },
            {
                "name": "burst_test",
                "threads": 16,
                "connections": 400,
                "duration": 60,
                "rate": 20000
            }
        ]
        
        results = []
        
        try:
            for benchmark in benchmarks:
                result = self.run_wrk2_benchmark(
                    benchmark["name"],
                    benchmark["threads"],
                    benchmark["connections"],
                    benchmark["duration"],
                    benchmark["rate"]
                )
                
                # Parse metrics from output
                if result["return_code"] == 0:
                    metrics = self.parse_wrk2_output(result["stdout"])
                    result["metrics"] = metrics
                    result["config"] = {
                        "threads": benchmark["threads"],
                        "connections": benchmark["connections"],
                        "duration": benchmark["duration"],
                        "target_rate": benchmark["rate"]
                    }
                
                results.append(result)
                time.sleep(2)  # Brief pause between tests
                
        finally:
            self.stop_nginx()
        
        return results
    
    def save_results(self, results):
        """Save benchmark results to JSON file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"nginx_benchmark_{timestamp}.json"
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
                "metrics": result.get("metrics", {}),
                "config": result.get("config", {})
            }
            summary["test_summary"].append(test_summary)
        
        return summary

def main():
    benchmark = NginxBenchmark()
    
    print("Starting Nginx benchmark suite with wrk2...")
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
            if 'latency_p99' in test['metrics']:
                print(f"    P99 Latency: {test['metrics']['latency_p99']:.2f}ms")
            if 'config' in test:
                print(f"    Target Rate: {test['config']['target_rate']} RPS, Connections: {test['config']['connections']}")
    
    else:
        print("Failed to run benchmarks")
        sys.exit(1)

if __name__ == "__main__":
    main()