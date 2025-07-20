#!/usr/bin/env python3
"""
Results analysis script for test case process monitoring data.
"""

import json
import sys
from datetime import datetime

def analyze_results(json_file):
    """Analyze process monitoring results."""
    
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    processes = data['processes']
    metadata = data['test_metadata']
    
    print(f"Test Duration: {metadata['total_duration']:.2f} seconds")
    print(f"Start Time: {metadata['start_time']}")
    print("")
    
    # Sort processes by CPU time
    proc_list = list(processes.values())
    proc_list.sort(key=lambda x: x.get('total_cpu_time', 0), reverse=True)
    
    print("Process Runtime Analysis:")
    print("=" * 80)
    print(f"{'PID':<8} {'Command':<25} {'CPU Time':<12} {'Wall Time':<12} {'CPU %':<8}")
    print("-" * 80)
    
    long_runners = []
    short_runners = []
    
    for proc in proc_list:
        pid = proc['pid']
        cmd_name = proc['name']
        cmdline = proc.get('cmdline', '')
        
        # Extract meaningful command for display
        if 'pigz' in cmd_name:
            # For pigz, show the file being compressed
            cmd_parts = cmdline.split()
            if len(cmd_parts) > 1:
                filename = cmd_parts[-1] if cmd_parts[-1].endswith(('.dat', '.iso')) else cmd_parts[-1]
                display_cmd = f"pigz {filename[-20:]}"
            else:
                display_cmd = cmd_name
        elif 'ffmpeg' in cmd_name:
            # For ffmpeg, show input file
            cmd_parts = cmdline.split()
            if '-i' in cmd_parts:
                try:
                    i_index = cmd_parts.index('-i')
                    input_file = cmd_parts[i_index + 1] if i_index + 1 < len(cmd_parts) else 'unknown'
                    display_cmd = f"ffmpeg {input_file[-15:]}"
                except:
                    display_cmd = cmd_name
            else:
                display_cmd = cmd_name
        else:
            display_cmd = cmd_name
        
        display_cmd = display_cmd[:24]
        
        cpu_time = proc.get('total_cpu_time', 0)
        wall_time = proc.get('wall_clock_time', 0)
        cpu_percent = proc.get('avg_cpu_percent', 0)
        
        print(f"{pid:<8} {display_cmd:<25} {cpu_time:<12.2f} {wall_time:<12.2f} {cpu_percent:<8.1f}")
        
        # Classify as long or short runner
        if cpu_time > 5.0:  # 5+ seconds of CPU time
            long_runners.append(proc)
        else:
            short_runners.append(proc)
    
    print("")
    print("Long-tail Analysis:")
    print("=" * 50)
    print(f"Total processes: {len(proc_list)}")
    print(f"Long runners (>5s CPU): {len(long_runners)}")
    print(f"Short runners (≤5s CPU): {len(short_runners)}")
    
    if long_runners:
        print(f"\nLong-running processes:")
        for proc in long_runners:
            print(f"  • {proc['name']} (PID {proc['pid']}): {proc.get('total_cpu_time', 0):.1f}s CPU time")
    
    if short_runners:
        total_short_cpu = sum(p.get('total_cpu_time', 0) for p in short_runners)
        avg_short_cpu = total_short_cpu / len(short_runners) if short_runners else 0
        print(f"\nShort-running processes:")
        print(f"  • Average CPU time: {avg_short_cpu:.2f}s")
        print(f"  • Total CPU time: {total_short_cpu:.2f}s")
    
    # Calculate potential scheduler benefit
    if long_runners and short_runners:
        max_long_cpu = max(p.get('total_cpu_time', 0) for p in long_runners)
        total_short_cpu = sum(p.get('total_cpu_time', 0) for p in short_runners)
        
        # Simulate current scheduling (round-robin on 2 CPUs)
        current_time = (max_long_cpu + total_short_cpu) / 2
        
        # Simulate optimized scheduling (long task on CPU 0, short tasks on CPU 1)
        optimized_time = max(max_long_cpu, total_short_cpu)
        
        if current_time > optimized_time:
            improvement = ((current_time - optimized_time) / current_time) * 100
            print(f"\nScheduler Optimization Potential:")
            print(f"  • Current estimated time: {current_time:.1f}s")
            print(f"  • Optimized estimated time: {optimized_time:.1f}s")
            print(f"  • Potential improvement: {improvement:.1f}%")
    
    print("")
    
    # Timeline analysis
    if proc_list:
        print("Timeline Analysis:")
        print("=" * 50)
        
        # Find processes that overlap in time
        overlapping = []
        for i, proc1 in enumerate(proc_list):
            for proc2 in proc_list[i+1:]:
                p1_start = proc1.get('start_time', 0)
                p1_end = p1_start + proc1.get('wall_clock_time', 0)
                p2_start = proc2.get('start_time', 0)
                p2_end = p2_start + proc2.get('wall_clock_time', 0)
                
                # Check for overlap
                if not (p1_end <= p2_start or p2_end <= p1_start):
                    overlapping.append((proc1, proc2))
        
        print(f"Concurrent process pairs: {len(overlapping)}")
        if overlapping:
            print("Processes running concurrently:")
            for proc1, proc2 in overlapping[:5]:  # Show first 5
                print(f"  • {proc1['name']} & {proc2['name']}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 analyze_results.py <process_analysis.json>")
        sys.exit(1)
    
    json_file = sys.argv[1]
    try:
        analyze_results(json_file)
    except FileNotFoundError:
        print(f"Error: File {json_file} not found.")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON file {json_file}.")
        sys.exit(1)