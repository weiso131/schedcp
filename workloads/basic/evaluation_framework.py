#!/usr/bin/env python3
"""
Academic Evaluation Framework for LLM Agent-Based Scheduler Auto-Tuning
"""

import json
import time
import subprocess
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple
import sys
sys.path.append('/root/yunwei37/ai-os/scheduler')
from scheduler_runner import SchedulerRunner

class SchedulerEvaluator:
    def __init__(self, output_dir: str = "evaluation_results"):
        self.runner = SchedulerRunner()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.results = []
        
    def run_baseline_experiment(self, scheduler: str, workload_cmd: str, 
                              duration: int = 300, warmup: int = 60) -> Dict:
        """Run experiment with static configuration"""
        print(f"Running baseline experiment: {scheduler} with {workload_cmd}")
        
        # Start scheduler
        scheduler_proc = self.runner.start_scheduler(scheduler)
        time.sleep(5)  # Let scheduler initialize
        
        # Start performance monitoring
        monitoring_procs = self._start_monitoring()
        
        # Run workload with warmup
        print(f"Warmup period: {warmup}s")
        workload_proc = subprocess.Popen(workload_cmd, shell=True)
        time.sleep(warmup)
        
        # Collect metrics during measurement period
        print(f"Measurement period: {duration}s")
        start_time = time.time()
        metrics = self._collect_realtime_metrics(duration)
        
        # Stop everything
        workload_proc.terminate()
        self._stop_monitoring(monitoring_procs)
        self.runner.stop_scheduler(scheduler_proc)
        
        return {
            "scheduler": scheduler,
            "workload": workload_cmd,
            "duration": duration,
            "metrics": metrics,
            "timestamp": datetime.now().isoformat()
        }
    
    def run_llm_experiment(self, scheduler: str, workload_cmd: str,
                          llm_agent, duration: int = 300, warmup: int = 60) -> Dict:
        """Run experiment with LLM-based tuning"""
        print(f"Running LLM-tuned experiment: {scheduler} with {workload_cmd}")
        
        # Start scheduler
        scheduler_proc = self.runner.start_scheduler(scheduler)
        time.sleep(5)
        
        # Start LLM agent monitoring and tuning
        llm_decisions = []
        
        # Start performance monitoring
        monitoring_procs = self._start_monitoring()
        
        # Run workload
        workload_proc = subprocess.Popen(workload_cmd, shell=True)
        time.sleep(warmup)
        
        # Measurement period with LLM tuning
        start_time = time.time()
        metrics = []
        
        while time.time() - start_time < duration:
            # Collect current metrics
            current_metrics = self._get_instant_metrics()
            metrics.append(current_metrics)
            
            # LLM agent makes decision
            if llm_agent and len(metrics) % 10 == 0:  # Every 10 samples
                decision = llm_agent.make_decision(current_metrics, scheduler)
                if decision:
                    llm_decisions.append({
                        "time": time.time() - start_time,
                        "decision": decision,
                        "metrics_before": current_metrics
                    })
                    self._apply_scheduler_params(scheduler, decision)
            
            time.sleep(1)
        
        # Stop everything
        workload_proc.terminate()
        self._stop_monitoring(monitoring_procs)
        self.runner.stop_scheduler(scheduler_proc)
        
        return {
            "scheduler": scheduler,
            "workload": workload_cmd,
            "duration": duration,
            "metrics": metrics,
            "llm_decisions": llm_decisions,
            "timestamp": datetime.now().isoformat()
        }
    
    def _start_monitoring(self) -> List[subprocess.Popen]:
        """Start various monitoring processes"""
        procs = []
        
        # scxtop monitoring
        scxtop_log = self.output_dir / f"scxtop_{time.time()}.log"
        procs.append(subprocess.Popen(
            f"./scheduler/tools/scxtop > {scxtop_log}", 
            shell=True
        ))
        
        # perf stat monitoring
        perf_log = self.output_dir / f"perf_{time.time()}.log"
        procs.append(subprocess.Popen(
            f"perf stat -a -I 1000 -o {perf_log}", 
            shell=True
        ))
        
        return procs
    
    def _stop_monitoring(self, procs: List[subprocess.Popen]):
        """Stop monitoring processes"""
        for proc in procs:
            proc.terminate()
            proc.wait()
    
    def _collect_realtime_metrics(self, duration: int) -> List[Dict]:
        """Collect metrics in real-time"""
        metrics = []
        start_time = time.time()
        
        while time.time() - start_time < duration:
            metrics.append(self._get_instant_metrics())
            time.sleep(1)
        
        return metrics
    
    def _get_instant_metrics(self) -> Dict:
        """Get current system metrics"""
        # Parse /proc/stat for CPU usage
        with open('/proc/stat', 'r') as f:
            cpu_line = f.readline()
            cpu_values = [int(x) for x in cpu_line.split()[1:]]
        
        # Get load average
        with open('/proc/loadavg', 'r') as f:
            loadavg = f.readline().split()[:3]
        
        # Get scheduler stats if available
        try:
            result = subprocess.run(
                ["./scheduler/tools/scxctl", "stats", "--json"],
                capture_output=True, text=True, timeout=1
            )
            sched_stats = json.loads(result.stdout) if result.returncode == 0 else {}
        except:
            sched_stats = {}
        
        return {
            "timestamp": time.time(),
            "cpu_total": sum(cpu_values),
            "cpu_idle": cpu_values[3],
            "loadavg_1m": float(loadavg[0]),
            "scheduler_stats": sched_stats
        }
    
    def _apply_scheduler_params(self, scheduler: str, params: Dict):
        """Apply new parameters to running scheduler"""
        # This would interact with the scheduler's control interface
        # For now, this is a placeholder
        print(f"Applying params to {scheduler}: {params}")
    
    def generate_figures(self, results: List[Dict]):
        """Generate publication-quality figures"""
        # Figure 1: Performance Comparison
        self._generate_performance_comparison(results)
        
        # Figure 2: Adaptation Timeline
        self._generate_adaptation_timeline(results)
        
        # Figure 3: Latency Distribution
        self._generate_latency_cdf(results)
        
        # Figure 4: Overhead Analysis
        self._generate_overhead_analysis(results)
    
    def _generate_performance_comparison(self, results: List[Dict]):
        """Generate performance comparison bar chart"""
        plt.figure(figsize=(10, 6))
        
        # Extract data for plotting
        schedulers = []
        throughputs = []
        
        for result in results:
            schedulers.append(result['scheduler'])
            # Calculate average throughput from metrics
            metrics = result['metrics']
            avg_throughput = np.mean([m.get('throughput', 0) for m in metrics])
            throughputs.append(avg_throughput)
        
        plt.bar(schedulers, throughputs)
        plt.xlabel('Scheduler Configuration')
        plt.ylabel('Average Throughput (ops/sec)')
        plt.title('Performance Comparison Across Scheduling Approaches')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'figure1_performance_comparison.pdf')
        plt.close()
    
    def _generate_adaptation_timeline(self, results: List[Dict]):
        """Generate adaptation timeline plot"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        
        for result in results:
            if 'llm_decisions' in result:
                # Plot performance over time
                metrics = result['metrics']
                times = [m['timestamp'] - metrics[0]['timestamp'] for m in metrics]
                cpu_usage = [100 - (m['cpu_idle'] / m['cpu_total'] * 100) for m in metrics]
                
                ax1.plot(times, cpu_usage, label=result['scheduler'])
                
                # Mark LLM decisions
                for decision in result['llm_decisions']:
                    ax1.axvline(x=decision['time'], color='red', linestyle='--', alpha=0.5)
                    ax2.scatter(decision['time'], 1, marker='v', s=100)
        
        ax1.set_ylabel('CPU Usage (%)')
        ax1.legend()
        ax1.set_title('System Performance and LLM Adaptation Timeline')
        
        ax2.set_xlabel('Time (seconds)')
        ax2.set_ylabel('LLM Decisions')
        ax2.set_ylim(0.5, 1.5)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'figure2_adaptation_timeline.pdf')
        plt.close()
    
    def _generate_latency_cdf(self, results: List[Dict]):
        """Generate latency CDF plot"""
        plt.figure(figsize=(10, 6))
        
        # Placeholder for latency data extraction
        # This would parse latency data from the collected metrics
        
        plt.xlabel('Latency (microseconds)')
        plt.ylabel('Percentile')
        plt.title('Latency Distribution (CDF)')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(self.output_dir / 'figure3_latency_cdf.pdf')
        plt.close()
    
    def _generate_overhead_analysis(self, results: List[Dict]):
        """Generate overhead analysis chart"""
        plt.figure(figsize=(10, 6))
        
        # Categories of overhead
        categories = ['Scheduler Base', 'LLM Inference', 'Parameter Adjustment']
        
        # Placeholder for overhead data
        # This would analyze the overhead from different components
        
        plt.xlabel('Configuration')
        plt.ylabel('Overhead (microseconds)')
        plt.title('Overhead Analysis of Scheduling Components')
        plt.legend()
        plt.tight_layout()
        plt.savefig(self.output_dir / 'figure4_overhead_analysis.pdf')
        plt.close()


def main():
    """Main evaluation script"""
    evaluator = SchedulerEvaluator()
    
    # Define experiments
    schedulers = ["scx_lavd", "scx_rusty", "scx_bpfland"]
    workloads = [
        "python workloads/basic/scheduler_test/schbench_bench_start.py --threads 16",
        "python workloads/llama.cpp/llamacpp_bench_start.py --model llama-7b --batch-size 4",
        "python workloads/cxl-micro/cxl_micro_bench_start.py --pattern sequential"
    ]
    
    results = []
    
    # Run baseline experiments
    for scheduler in schedulers:
        for workload in workloads:
            print(f"\n{'='*60}")
            print(f"Baseline: {scheduler} - {workload}")
            print(f"{'='*60}")
            
            result = evaluator.run_baseline_experiment(scheduler, workload, duration=60)
            results.append(result)
            
            # Save intermediate results
            with open(evaluator.output_dir / f"result_{len(results)}.json", 'w') as f:
                json.dump(result, f, indent=2)
    
    # Generate figures
    evaluator.generate_figures(results)
    
    print(f"\nEvaluation complete. Results saved to {evaluator.output_dir}")


if __name__ == "__main__":
    main()