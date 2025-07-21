#!/usr/bin/env python3
"""
Long-tail Workload Evaluation Framework

This framework executes and evaluates workloads with long-tail characteristics
to demonstrate the benefits of custom Linux kernel schedulers.
"""

import json
import subprocess
import time
import os
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import threading
import signal

# Optional psutil import for advanced process monitoring
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    print("Warning: psutil not available. Process monitoring will be limited.")

class ProcessMonitor:
    """Monitor CPU usage and process runtime for long-tail detection"""
    
    def __init__(self):
        self.processes = {}
        self.monitoring = False
        self.monitor_thread = None
        
    def start_monitoring(self):
        """Start monitoring system processes"""
        self.monitoring = True
        self.processes = {}
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.start()
        
    def stop_monitoring(self):
        """Stop monitoring and return results"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()
        return self.processes
        
    def _monitor_loop(self):
        """Main monitoring loop"""
        if not PSUTIL_AVAILABLE:
            return
            
        while self.monitoring:
            try:
                for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'create_time']):
                    try:
                        proc_info = proc.info
                        pid = proc_info['pid']
                        
                        if pid not in self.processes:
                            self.processes[pid] = {
                                'name': proc_info['name'],
                                'start_time': time.time(),
                                'cpu_samples': [],
                                'runtime': 0
                            }
                        
                        # Update runtime and CPU usage
                        self.processes[pid]['runtime'] = time.time() - self.processes[pid]['start_time']
                        cpu_percent = proc_info['cpu_percent']
                        if cpu_percent is not None:
                            self.processes[pid]['cpu_samples'].append(cpu_percent)
                            
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        continue
                        
                time.sleep(0.5)  # Sample every 500ms
                
            except Exception as e:
                print(f"Monitoring error: {e}")
                continue

class WorkloadEvaluator:
    """Main evaluation framework for long-tail workloads"""
    
    def __init__(self, config_file: str = "test_cases.json"):
        self.config_file = config_file
        self.config = self._load_config()
        self.results = {}
        
    def _load_config(self) -> Dict:
        """Load test configuration from JSON file"""
        try:
            with open(self.config_file, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"Error: Configuration file {self.config_file} not found!")
            sys.exit(1)
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON configuration: {e}")
            sys.exit(1)
            
    def list_tests(self):
        """List all available tests"""
        print("Available test cases:")
        print("=" * 50)
        
        for test in self.config['test_cases']:
            print(f"ID: {test['id']}")
            print(f"Name: {test['name']}")
            print(f"Category: {test['category']}")
            print(f"Description: {test['description']}")
            print(f"Expected improvement: {test['expected_improvement']*100:.0f}%")
            print(f"Estimated runtime: {test['workload_characteristics']['expected_runtime_seconds']}s")
            print("-" * 50)
            
    def check_dependencies(self, test_case: Dict) -> Tuple[bool, List[str]]:
        """Check if test dependencies are available"""
        missing_deps = []
        
        for dep in test_case.get('dependencies', []):
            # Check if command exists
            try:
                result = subprocess.run(['which', dep], 
                                      capture_output=True, 
                                      text=True, 
                                      timeout=5)
                if result.returncode != 0:
                    # Try checking if it's a Python package
                    if dep.startswith('python3-'):
                        package = dep.replace('python3-', '').replace('-', '_')
                        try:
                            import importlib
                            importlib.import_module(package)
                        except ImportError:
                            missing_deps.append(dep)
                    else:
                        missing_deps.append(dep)
            except subprocess.TimeoutExpired:
                missing_deps.append(dep)
                
        return len(missing_deps) == 0, missing_deps
        
    def _run_commands(self, commands: List[str], timeout: int = 300) -> Tuple[bool, str, float]:
        """Execute a list of shell commands"""
        start_time = time.time()
        
        for cmd in commands:
            if not cmd.strip():  # Skip empty commands
                continue
                
            try:
                print(f"  Running: {cmd}")
                result = subprocess.run(cmd, 
                                      shell=True, 
                                      capture_output=True, 
                                      text=True, 
                                      timeout=timeout)
                
                if result.returncode != 0:
                    return False, f"Command failed: {cmd}\\nError: {result.stderr}", time.time() - start_time
                    
            except subprocess.TimeoutExpired:
                return False, f"Command timed out: {cmd}", time.time() - start_time
            except Exception as e:
                return False, f"Command error: {cmd}\\nException: {str(e)}", time.time() - start_time
                
        return True, "Success", time.time() - start_time
        
    def run_test(self, test_id: str, monitor_processes: bool = True) -> Dict:
        """Run a specific test case"""
        # Find test case
        test_case = None
        for test in self.config['test_cases']:
            if test['id'] == test_id:
                test_case = test
                break
                
        if not test_case:
            return {'error': f"Test case '{test_id}' not found"}
            
        print(f"Running test: {test_case['name']}")
        print(f"Description: {test_case['description']}")
        
        # Check dependencies
        deps_ok, missing_deps = self.check_dependencies(test_case)
        if not deps_ok:
            return {
                'error': f"Missing dependencies: {', '.join(missing_deps)}",
                'missing_dependencies': missing_deps
            }
            
        result = {
            'test_id': test_id,
            'test_name': test_case['name'],
            'start_time': time.time(),
            'expected_improvement': test_case['expected_improvement']
        }
        
        # Setup phase
        print("Running setup...")
        setup_success, setup_output, setup_time = self._run_commands(
            test_case.get('setup_commands', [])
        )
        
        if not setup_success:
            result.update({
                'status': 'failed',
                'error': f"Setup failed: {setup_output}",
                'setup_time': setup_time
            })
            return result
            
        # Main test execution with monitoring
        monitor = ProcessMonitor() if (monitor_processes and PSUTIL_AVAILABLE) else None
        
        try:
            print("Running main test command...")
            
            if monitor:
                monitor.start_monitoring()
                
            test_start = time.time()
            test_success, test_output, test_time = self._run_commands(
                [test_case['test_command']],
                timeout=test_case['workload_characteristics']['expected_runtime_seconds'] * 3
            )
            test_end = time.time()
            
            if monitor:
                process_data = monitor.stop_monitoring()
            else:
                process_data = {}
                
            # Analyze process data for long-tail detection
            long_tail_analysis = self._analyze_long_tail(process_data) if process_data else {}
            
            result.update({
                'status': 'success' if test_success else 'failed',
                'test_output': test_output if not test_success else 'Test completed successfully',
                'setup_time': setup_time,
                'test_time': test_time,
                'wall_clock_time': test_end - test_start,
                'process_analysis': long_tail_analysis,
                'workload_characteristics': test_case['workload_characteristics']
            })
            
        except Exception as e:
            result.update({
                'status': 'error',
                'error': str(e),
                'setup_time': setup_time
            })
            
        finally:
            # Cleanup phase
            print("Running cleanup...")
            cleanup_success, cleanup_output, cleanup_time = self._run_commands(
                test_case.get('cleanup_commands', [])
            )
            
            result['cleanup_time'] = cleanup_time
            if not cleanup_success:
                result['cleanup_error'] = cleanup_output
                
        result['total_time'] = time.time() - result['start_time']
        return result
        
    def _analyze_long_tail(self, process_data: Dict) -> Dict:
        """Analyze process data to identify long-tail characteristics"""
        if not process_data:
            return {}
            
        # Filter out system processes and focus on test-related processes
        relevant_processes = {}
        long_tail_threshold = 0.5  # 500ms threshold for long-tail detection
        
        for pid, data in process_data.items():
            if data['runtime'] > long_tail_threshold:
                relevant_processes[pid] = data
                
        if not relevant_processes:
            return {'long_tail_detected': False}
            
        runtimes = [p['runtime'] for p in relevant_processes.values()]
        cpu_averages = []
        
        for p in relevant_processes.values():
            if p['cpu_samples']:
                cpu_averages.append(sum(p['cpu_samples']) / len(p['cpu_samples']))
                
        analysis = {
            'long_tail_detected': len(relevant_processes) > 0,
            'process_count': len(relevant_processes),
            'runtime_stats': {
                'min': min(runtimes) if runtimes else 0,
                'max': max(runtimes) if runtimes else 0,
                'avg': sum(runtimes) / len(runtimes) if runtimes else 0,
                'skew_ratio': max(runtimes) / min(runtimes) if runtimes and min(runtimes) > 0 else 1
            },
            'cpu_stats': {
                'avg_cpu_usage': sum(cpu_averages) / len(cpu_averages) if cpu_averages else 0
            }
        }
        
        return analysis
        
    def run_all_tests(self) -> Dict:
        """Run all test cases"""
        print(f"Running all {len(self.config['test_cases'])} test cases...")
        print("=" * 60)
        
        all_results = {}
        failed_tests = []
        
        for test_case in self.config['test_cases']:
            test_id = test_case['id']
            print(f"\\n[{test_id}] Starting test...")
            
            result = self.run_test(test_id)
            all_results[test_id] = result
            
            if result.get('status') == 'failed':
                failed_tests.append(test_id)
                print(f"[{test_id}] FAILED: {result.get('error', 'Unknown error')}")
            else:
                print(f"[{test_id}] COMPLETED in {result.get('total_time', 0):.2f}s")
                
        # Generate summary
        summary = self._generate_summary(all_results)
        
        return {
            'summary': summary,
            'results': all_results,
            'failed_tests': failed_tests
        }
        
    def _generate_summary(self, results: Dict) -> Dict:
        """Generate summary statistics from test results"""
        successful_tests = [r for r in results.values() if r.get('status') == 'success']
        
        if not successful_tests:
            return {'total_tests': len(results), 'successful_tests': 0}
            
        summary = {
            'total_tests': len(results),
            'successful_tests': len(successful_tests),
            'success_rate': len(successful_tests) / len(results),
            'total_runtime': sum(r.get('total_time', 0) for r in successful_tests),
            'avg_runtime': sum(r.get('total_time', 0) for r in successful_tests) / len(successful_tests),
            'long_tail_detected': sum(1 for r in successful_tests 
                                    if r.get('process_analysis', {}).get('long_tail_detected', False)),
            'expected_improvements': [r.get('expected_improvement', 0) for r in successful_tests],
            'categories': {}
        }
        
        # Group by category
        for test in self.config['test_cases']:
            category = test['category']
            if category not in summary['categories']:
                summary['categories'][category] = 0
            if test['id'] in results and results[test['id']].get('status') == 'success':
                summary['categories'][category] += 1
                
        return summary
        
    def save_results(self, results: Dict, filename: str = None):
        """Save test results to JSON file"""
        if filename is None:
            timestamp = int(time.time())
            filename = f"test_results_{timestamp}.json"
            
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
            
        print(f"Results saved to: {filename}")

def main():
    parser = argparse.ArgumentParser(description='Long-tail Workload Evaluation Framework')
    parser.add_argument('--config', default='test_cases.json', 
                       help='Test configuration file (default: test_cases.json)')
    parser.add_argument('--list', action='store_true', 
                       help='List all available test cases')
    parser.add_argument('--test', metavar='TEST_ID', 
                       help='Run a specific test case')
    parser.add_argument('--all', action='store_true', 
                       help='Run all test cases')
    parser.add_argument('--no-monitor', action='store_true', 
                       help='Disable process monitoring')
    parser.add_argument('--save', metavar='FILENAME', 
                       help='Save results to specified file')
    
    args = parser.parse_args()
    
    evaluator = WorkloadEvaluator(args.config)
    
    if args.list:
        evaluator.list_tests()
        return
        
    if args.test:
        result = evaluator.run_test(args.test, monitor_processes=not args.no_monitor)
        print("\\nTest Result:")
        print("=" * 40)
        print(json.dumps(result, indent=2))
        
        if args.save:
            evaluator.save_results({'single_test': result}, args.save)
            
    elif args.all:
        results = evaluator.run_all_tests()
        
        print("\\nSummary:")
        print("=" * 40)
        summary = results['summary']
        print(f"Total tests: {summary['total_tests']}")
        print(f"Successful: {summary['successful_tests']}")
        print(f"Success rate: {summary['success_rate']*100:.1f}%")
        print(f"Total runtime: {summary['total_runtime']:.2f}s")
        print(f"Average runtime: {summary['avg_runtime']:.2f}s")
        print(f"Long-tail detected: {summary['long_tail_detected']} tests")
        
        if results['failed_tests']:
            print(f"Failed tests: {', '.join(results['failed_tests'])}")
            
        if args.save:
            evaluator.save_results(results, args.save)
        else:
            evaluator.save_results(results)
            
    else:
        print("Please specify an action: --list, --test TEST_ID, or --all")
        print("Use --help for more information")

if __name__ == '__main__':
    main()