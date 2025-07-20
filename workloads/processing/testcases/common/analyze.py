#!/usr/bin/env python3
"""
Common analysis script for tracking process/thread runtime in test cases.
This script monitors all child processes and tracks their CPU time and wall clock time.
"""

import os
import sys
import time
import json
import psutil
import threading
from collections import defaultdict
from datetime import datetime

class ProcessTracker:
    def __init__(self, output_file="process_analysis.json"):
        self.output_file = output_file
        self.process_data = defaultdict(dict)
        self.start_time = time.time()
        self.monitoring = True
        self.monitor_thread = None
        
    def start_monitoring(self, target_pids=None):
        """Start monitoring processes. If target_pids is None, monitor all child processes."""
        self.target_pids = target_pids or []
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        
    def _monitor_loop(self):
        """Main monitoring loop."""
        seen_pids = set()
        
        while self.monitoring:
            current_pids = set()
            
            # If no specific PIDs given, find all child processes
            if not self.target_pids:
                try:
                    current_process = psutil.Process()
                    children = current_process.children(recursive=True)
                    current_pids = {p.pid for p in children if p.is_running()}
                except:
                    pass
            else:
                current_pids = set(self.target_pids)
            
            # Monitor each process
            for pid in current_pids:
                try:
                    proc = psutil.Process(pid)
                    if proc.is_running():
                        self._record_process_info(proc)
                        seen_pids.add(pid)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            
            # Check for processes that have ended
            for pid in seen_pids - current_pids:
                if pid in self.process_data:
                    self.process_data[pid]['status'] = 'completed'
                    self.process_data[pid]['end_time'] = time.time() - self.start_time
            
            time.sleep(0.1)  # Sample every 100ms
    
    def _record_process_info(self, proc):
        """Record information about a process."""
        pid = proc.pid
        current_time = time.time() - self.start_time
        
        try:
            # Get process info
            cpu_times = proc.cpu_times()
            memory_info = proc.memory_info()
            
            if pid not in self.process_data:
                self.process_data[pid] = {
                    'pid': pid,
                    'name': proc.name(),
                    'cmdline': ' '.join(proc.cmdline()),
                    'start_time': current_time,
                    'status': 'running',
                    'cpu_samples': [],
                    'memory_samples': []
                }
            
            # Record CPU and memory usage
            self.process_data[pid]['cpu_samples'].append({
                'time': current_time,
                'user_time': cpu_times.user,
                'system_time': cpu_times.system,
                'total_time': cpu_times.user + cpu_times.system
            })
            
            self.process_data[pid]['memory_samples'].append({
                'time': current_time,
                'rss': memory_info.rss,
                'vms': memory_info.vms
            })
            
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    
    def stop_monitoring(self):
        """Stop monitoring and save results."""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()
        
        # Calculate final statistics
        for pid, data in self.process_data.items():
            if data['cpu_samples']:
                # Calculate total CPU time
                last_sample = data['cpu_samples'][-1]
                data['total_cpu_time'] = last_sample['total_time']
                data['user_cpu_time'] = last_sample['user_time']
                data['system_cpu_time'] = last_sample['system_time']
                
                # Calculate wall clock time
                if 'end_time' in data:
                    data['wall_clock_time'] = data['end_time'] - data['start_time']
                else:
                    data['wall_clock_time'] = time.time() - self.start_time - data['start_time']
                
                # Calculate average CPU usage
                if data['wall_clock_time'] > 0:
                    data['avg_cpu_percent'] = (data['total_cpu_time'] / data['wall_clock_time']) * 100
                else:
                    data['avg_cpu_percent'] = 0
        
        # Save to file
        self.save_results()
        return self.process_data
    
    def save_results(self):
        """Save results to JSON file."""
        output = {
            'test_metadata': {
                'start_time': datetime.fromtimestamp(self.start_time).isoformat(),
                'total_duration': time.time() - self.start_time
            },
            'processes': dict(self.process_data)
        }
        
        with open(self.output_file, 'w') as f:
            json.dump(output, f, indent=2)
    
    def print_summary(self):
        """Print a summary of process runtimes."""
        print("\n=== Process Runtime Analysis ===")
        
        # Sort processes by total CPU time
        sorted_procs = sorted(
            self.process_data.values(),
            key=lambda x: x.get('total_cpu_time', 0),
            reverse=True
        )
        
        print(f"{'PID':<8} {'Command':<20} {'CPU Time':<10} {'Wall Time':<10} {'Avg CPU%':<10}")
        print("-" * 70)
        
        for proc in sorted_procs[:10]:  # Show top 10
            pid = proc['pid']
            cmd = proc['name'][:18] + '..' if len(proc['name']) > 20 else proc['name']
            cpu_time = f"{proc.get('total_cpu_time', 0):.2f}s"
            wall_time = f"{proc.get('wall_clock_time', 0):.2f}s"
            cpu_pct = f"{proc.get('avg_cpu_percent', 0):.1f}%"
            
            print(f"{pid:<8} {cmd:<20} {cpu_time:<10} {wall_time:<10} {cpu_pct:<10}")
        
        # Identify long-tail processes (> 5 seconds CPU time)
        long_runners = [p for p in sorted_procs if p.get('total_cpu_time', 0) > 5.0]
        if long_runners:
            print(f"\nLong-running processes (>5s CPU time): {len(long_runners)}")
            for proc in long_runners:
                print(f"  - {proc['name']} (PID {proc['pid']}): {proc.get('total_cpu_time', 0):.2f}s CPU")


def monitor_command(cmd, output_file="process_analysis.json"):
    """Helper function to monitor a command execution."""
    tracker = ProcessTracker(output_file)
    tracker.start_monitoring()
    
    # Execute the command
    start_time = time.time()
    result = os.system(cmd)
    end_time = time.time()
    
    # Stop monitoring
    data = tracker.stop_monitoring()
    
    print(f"\nCommand completed in {end_time - start_time:.2f} seconds")
    tracker.print_summary()
    
    return result, data


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 analyze.py <command>")
        print("   or: python3 analyze.py --monitor-pids <pid1,pid2,...>")
        sys.exit(1)
    
    if sys.argv[1] == "--monitor-pids":
        # Monitor specific PIDs
        pids = [int(p) for p in sys.argv[2].split(',')]
        tracker = ProcessTracker()
        tracker.start_monitoring(pids)
        
        try:
            input("Press Enter to stop monitoring...")
        except KeyboardInterrupt:
            pass
        
        tracker.stop_monitoring()
        tracker.print_summary()
    else:
        # Monitor command execution
        cmd = ' '.join(sys.argv[1:])
        monitor_command(cmd)