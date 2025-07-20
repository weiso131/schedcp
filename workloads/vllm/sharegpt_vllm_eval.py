#!/usr/bin/env python3
"""
ShareGPT evaluation script for vLLM
Tests throughput and latency with real-world conversation patterns across multiple schedulers
"""

import os
import sys
import json
import time
import argparse
import subprocess
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import platform
import psutil
import socket
import torch

# Add the scheduler module to the path
sys.path.insert(0, '/root/yunwei37/ai-os')
try:
    from scheduler import SchedulerRunner, SchedulerBenchmark
except ImportError:
    # Fallback if scheduler module not available
    class SchedulerRunner:
        pass
    class SchedulerBenchmark:
        def __init__(self, scheduler_runner=None):
            self.runner = scheduler_runner

# Ensure we're not importing from local vllm directory
if 'vllm' in os.listdir('.'):
    sys.path = [p for p in sys.path if not p.endswith('/vllm')]

from vllm import LLM, SamplingParams


@dataclass
class BenchmarkResult:
    """Container for benchmark results"""
    prompt: str
    response: str
    prompt_tokens: int
    response_tokens: int
    latency_ms: float
    tokens_per_second: float
    success: bool
    error: Optional[str] = None


class VLLMBenchmark(SchedulerBenchmark):
    """Benchmark vLLM with ShareGPT dataset across multiple schedulers"""
    
    def __init__(self, model_path: str = "facebook/opt-125m",
                 scheduler_runner: SchedulerRunner = None,
                 gpu_memory_utilization: float = 0.9,
                 max_model_len: int = 2048,
                 use_v0: bool = True):
        super().__init__(scheduler_runner)
        
        self.model_path = model_path
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_model_len = max_model_len
        self.use_v0 = use_v0
        self.llm = None
        self.current_scheduler = None
        
        # Force V0 engine if requested
        if use_v0:
            os.environ['VLLM_USE_V1'] = '0'
    
    def setup_scheduler(self, scheduler_name: Optional[str] = None) -> bool:
        """Setup the environment for a specific scheduler"""
        try:
            if scheduler_name and self.runner:
                print(f"Setting up scheduler: {scheduler_name}")
                # Start the scheduler
                success = self.runner.start_scheduler(scheduler_name)
                if success:
                    self.current_scheduler = scheduler_name
                    # Give scheduler time to initialize
                    time.sleep(2)
                return success
            return True
        except Exception as e:
            print(f"Failed to setup scheduler {scheduler_name}: {e}")
            return False
    
    def cleanup_scheduler(self):
        """Stop the current scheduler"""
        if self.current_scheduler and self.runner:
            print(f"Stopping scheduler: {self.current_scheduler}")
            self.runner.stop_scheduler(self.current_scheduler)
            self.current_scheduler = None
            time.sleep(2)
    
    def initialize_llm(self) -> bool:
        """Initialize vLLM engine"""
        try:
            print(f"Initializing vLLM with model: {self.model_path}")
            self.llm = LLM(
                model=self.model_path,
                trust_remote_code=True,
                max_model_len=self.max_model_len,
                gpu_memory_utilization=self.gpu_memory_utilization,
                enforce_eager=True if self.use_v0 else False,  # Disable CUDA graphs for V0
                download_dir="/tmp/vllm_models"  # Specify download directory
            )
            return True
        except Exception as e:
            print(f"Failed to initialize vLLM: {e}")
            if "facebook/opt-3b" in self.model_path or "Llama-3" in self.model_path:
                print("\nNote: The model appears to be unavailable or requires authentication.")
                print("Try using one of these alternatives:")
                print("  - facebook/opt-125m (smaller, faster)")
                print("  - facebook/opt-1.3b (medium size)")
                print("  - facebook/opt-2.7b (medium-large size)")
                print("  - facebook/opt-6.7b (larger, requires more memory)")
                print("  - meta-llama/Llama-3.2-1B (requires HuggingFace token)")
                print("  - meta-llama/Llama-3.2-3B (requires HuggingFace token)")
                print("  - meta-llama/Llama-2-7b-hf (requires HuggingFace token)")
                print("\nFor Llama models, you need to:")
                print("  1. Accept the license agreement on HuggingFace")
                print("  2. Set your HF token: export HF_TOKEN=your_token_here")
                print("\nExample: python sharegpt_vllm_eval.py --model facebook/opt-2.7b --num-samples 100")
            return False
    
    def cleanup_llm(self):
        """Cleanup vLLM resources"""
        if self.llm:
            # vLLM doesn't have explicit cleanup, but we can delete the object
            del self.llm
            self.llm = None
            # Force garbage collection
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    def run_inference(self, prompts: List[str], max_tokens: int = 128) -> List[BenchmarkResult]:
        """Run inference on a batch of prompts"""
        if not self.llm:
            return []
        
        sampling_params = SamplingParams(
            temperature=0.7,
            top_p=0.9,
            max_tokens=max_tokens,
            skip_special_tokens=True
        )
        
        results = []
        
        try:
            # Process in batches for better progress tracking
            batch_size = 100
            for i in tqdm(range(0, len(prompts), batch_size), desc="Processing batches"):
                batch_prompts = prompts[i:i + batch_size]
                
                start_time = time.time()
                outputs = self.llm.generate(batch_prompts, sampling_params)
                end_time = time.time()
                
                batch_time = (end_time - start_time) * 1000  # Convert to ms
                avg_latency_per_prompt = batch_time / len(batch_prompts)
                
                for prompt, output in zip(batch_prompts, outputs):
                    generated_text = output.outputs[0].text
                    prompt_tokens = len(output.prompt_token_ids)
                    response_tokens = len(output.outputs[0].token_ids)
                    
                    # Approximate individual latency
                    latency_ms = avg_latency_per_prompt
                    tokens_per_second = response_tokens / (latency_ms / 1000) if latency_ms > 0 else 0
                    
                    results.append(BenchmarkResult(
                        prompt=prompt,
                        response=generated_text,
                        prompt_tokens=prompt_tokens,
                        response_tokens=response_tokens,
                        latency_ms=latency_ms,
                        tokens_per_second=tokens_per_second,
                        success=True
                    ))
                    
        except Exception as e:
            print(f"Error during inference: {e}")
            # Add failed results for remaining prompts
            for prompt in prompts[len(results):]:
                results.append(BenchmarkResult(
                    prompt=prompt,
                    response="",
                    prompt_tokens=0,
                    response_tokens=0,
                    latency_ms=0,
                    tokens_per_second=0,
                    success=False,
                    error=str(e)
                ))
        
        return results
    
    def load_sharegpt_dataset(self, dataset_path: str, num_samples: int = 1000) -> List[str]:
        """Load ShareGPT dataset and extract prompts"""
        with open(dataset_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        prompts = []
        count = 0
        
        for item in data:
            if count >= num_samples:
                break
            
            if isinstance(item, dict) and 'prompt' in item:
                prompts.append(item['prompt'])
                count += 1
            elif isinstance(item, dict) and 'conversations' in item:
                # ShareGPT Vicuna format
                for conv in item['conversations']:
                    if conv.get('from') in ['human', 'user']:
                        prompts.append(conv['value'])
                        count += 1
                        break
        
        print(f"Loaded {len(prompts)} prompts from ShareGPT dataset")
        return prompts
    
    def analyze_results(self, results: List[BenchmarkResult]) -> Dict:
        """Analyze benchmark results"""
        successful_results = [r for r in results if r.success]
        
        if not successful_results:
            return {"error": "No successful results"}
        
        latencies = [r.latency_ms for r in successful_results]
        tps_values = [r.tokens_per_second for r in successful_results]
        response_tokens = [r.response_tokens for r in successful_results]
        
        return {
            "total_requests": len(results),
            "successful_requests": len(successful_results),
            "failed_requests": len(results) - len(successful_results),
            "latency_mean_ms": np.mean(latencies),
            "latency_median_ms": np.median(latencies),
            "latency_p99_ms": np.percentile(latencies, 99),
            "tokens_per_second_mean": np.mean(tps_values),
            "tokens_per_second_median": np.median(tps_values),
            "total_tokens_generated": sum(response_tokens),
            "avg_response_tokens": np.mean(response_tokens),
            "throughput_requests_per_sec": len(successful_results) / (sum(latencies) / 1000) if sum(latencies) > 0 else 0
        }
    
    def run_benchmark_suite(self, dataset_path: str, 
                          schedulers: List[str] = None,
                          num_samples: int = 1000,
                          max_tokens: int = 128) -> Dict:
        """Run complete benchmark suite across multiple schedulers"""
        prompts = self.load_sharegpt_dataset(dataset_path, num_samples)
        
        if not prompts:
            print("No prompts loaded from dataset")
            return {}
        
        results = {}
        
        # Test default scheduler
        print("\nTesting default scheduler...")
        if self.initialize_llm():
            start_time = time.time()
            benchmark_results = self.run_inference(prompts, max_tokens)
            end_time = time.time()
            
            results["default"] = self.analyze_results(benchmark_results)
            results["default"]["total_time_seconds"] = end_time - start_time
            results["default"]["raw_results"] = benchmark_results
            
            self.cleanup_llm()
            time.sleep(5)  # Wait before next test
        
        # Test each scheduler
        if schedulers and self.runner:
            for scheduler in schedulers:
                print(f"\nTesting scheduler: {scheduler}")
                
                if self.setup_scheduler(scheduler):
                    if self.initialize_llm():
                        start_time = time.time()
                        benchmark_results = self.run_inference(prompts, max_tokens)
                        end_time = time.time()
                        
                        results[scheduler] = self.analyze_results(benchmark_results)
                        results[scheduler]["total_time_seconds"] = end_time - start_time
                        results[scheduler]["raw_results"] = benchmark_results
                        
                        self.cleanup_llm()
                    
                    self.cleanup_scheduler()
                    time.sleep(5)  # Wait before next test
        
        return results
    
    def get_system_info(self) -> Dict:
        """Collect system information"""
        try:
            import torch
            
            # Get CPU info
            cpu_info = {
                "cpu_count": psutil.cpu_count(logical=False),
                "cpu_count_logical": psutil.cpu_count(logical=True),
                "cpu_freq": psutil.cpu_freq().current if psutil.cpu_freq() else "N/A",
                "cpu_percent": psutil.cpu_percent(interval=1)
            }
            
            # Get memory info
            mem = psutil.virtual_memory()
            mem_info = {
                "total_memory_gb": round(mem.total / (1024**3), 2),
                "available_memory_gb": round(mem.available / (1024**3), 2),
                "memory_percent": mem.percent
            }
            
            # Get GPU info if available
            gpu_info = {}
            if torch.cuda.is_available():
                gpu_info = {
                    "gpu_count": torch.cuda.device_count(),
                    "gpu_name": torch.cuda.get_device_name(0),
                    "gpu_memory_gb": round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2)
                }
            
            # Get system info
            system_info = {
                "platform": platform.system(),
                "platform_release": platform.release(),
                "architecture": platform.machine(),
                "hostname": socket.gethostname(),
                "python_version": platform.python_version(),
                "torch_version": torch.__version__ if 'torch' in locals() else "N/A",
                "vllm_version": self._get_vllm_version()
            }
            
            return {
                "system": system_info,
                "cpu": cpu_info,
                "memory": mem_info,
                "gpu": gpu_info,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            return {"error": str(e), "timestamp": datetime.now().isoformat()}
    
    def _get_vllm_version(self) -> str:
        """Get vLLM version"""
        try:
            import vllm
            return vllm.__version__
        except:
            return "Unknown"
    
    def generate_report(self, results: Dict, output_dir: str = "results", 
                       dataset_name: str = None, config_info: Dict = None):
        """Generate comprehensive benchmark report"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Get system info
        system_info = self.get_system_info()
        
        # Save raw results with metadata
        results_with_metadata = {
            "system_info": system_info,
            "dataset": dataset_name,
            "config": config_info,
            "results": {}
        }
        
        # Remove raw_results for JSON serialization
        for scheduler, data in results.items():
            results_with_metadata["results"][scheduler] = {
                k: v for k, v in data.items() if k != "raw_results"
            }
        
        results_file = os.path.join(output_dir, "sharegpt_vllm_results.json")
        with open(results_file, 'w') as f:
            json.dump(results_with_metadata, f, indent=2)
        
        # Generate comparison plots
        self._generate_comparison_plots(results, output_dir)
        
        # Generate summary report
        self._generate_summary_report(results, output_dir, system_info, dataset_name, config_info)
    
    def _generate_comparison_plots(self, results: Dict, output_dir: str):
        """Generate comparison plots"""
        schedulers = list(results.keys())
        
        # Extract metrics
        latency_means = [results[s].get("latency_mean_ms", 0) for s in schedulers]
        latency_p99s = [results[s].get("latency_p99_ms", 0) for s in schedulers]
        tps_means = [results[s].get("tokens_per_second_mean", 0) for s in schedulers]
        throughput_rps = [results[s].get("throughput_requests_per_sec", 0) for s in schedulers]
        
        # Create figure
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('ShareGPT Benchmark Results - vLLM', fontsize=16)
        
        # Latency comparison
        x = np.arange(len(schedulers))
        width = 0.35
        
        ax1.bar(x - width/2, latency_means, width, label='Mean', alpha=0.8)
        ax1.bar(x + width/2, latency_p99s, width, label='P99', alpha=0.8)
        ax1.set_xlabel('Scheduler')
        ax1.set_ylabel('Latency (ms)')
        ax1.set_title('Latency Comparison')
        ax1.set_xticks(x)
        ax1.set_xticklabels(schedulers, rotation=45)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Token throughput comparison
        ax2.bar(schedulers, tps_means, color='green', alpha=0.8)
        ax2.set_xlabel('Scheduler')
        ax2.set_ylabel('Tokens per Second')
        ax2.set_title('Token Generation Throughput')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3)
        
        # Request throughput
        ax3.bar(schedulers, throughput_rps, color='blue', alpha=0.8)
        ax3.set_xlabel('Scheduler')
        ax3.set_ylabel('Requests per Second')
        ax3.set_title('Request Throughput')
        ax3.tick_params(axis='x', rotation=45)
        ax3.grid(True, alpha=0.3)
        
        # Total time comparison
        total_times = [results[s].get("total_time_seconds", 0) for s in schedulers]
        ax4.bar(schedulers, total_times, color='orange', alpha=0.8)
        ax4.set_xlabel('Scheduler')
        ax4.set_ylabel('Total Time (seconds)')
        ax4.set_title('Total Processing Time')
        ax4.tick_params(axis='x', rotation=45)
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save figure
        figure_path = os.path.join(output_dir, "sharegpt_vllm_performance.png")
        plt.savefig(figure_path, dpi=300, bbox_inches='tight')
        print(f"Performance figure saved to {figure_path}")
    
    def _generate_summary_report(self, results: Dict, output_dir: str, 
                                system_info: Dict, dataset_name: str, config_info: Dict):
        """Generate text summary report"""
        report_path = os.path.join(output_dir, "sharegpt_vllm_summary.txt")
        
        with open(report_path, 'w') as f:
            f.write("ShareGPT Benchmark Results - vLLM\n")
            f.write("=" * 60 + "\n\n")
            
            # Write system info
            f.write("System Information:\n")
            f.write("-" * 30 + "\n")
            if "error" not in system_info:
                sys_info = system_info['system']
                f.write(f"Platform: {sys_info['platform']} {sys_info['platform_release']}\n")
                f.write(f"Architecture: {sys_info['architecture']}\n")
                f.write(f"Python: {sys_info['python_version']}\n")
                f.write(f"PyTorch: {sys_info['torch_version']}\n")
                f.write(f"vLLM: {sys_info['vllm_version']}\n")
                
                if system_info.get('gpu'):
                    gpu = system_info['gpu']
                    f.write(f"GPU: {gpu.get('gpu_name', 'N/A')} ({gpu.get('gpu_memory_gb', 'N/A')} GB)\n")
                
                cpu = system_info['cpu']
                f.write(f"CPU: {cpu['cpu_count']} cores ({cpu['cpu_count_logical']} logical)\n")
                
                mem = system_info['memory']
                f.write(f"Memory: {mem['total_memory_gb']} GB total\n")
            else:
                f.write(f"Error collecting system info: {system_info['error']}\n")
            
            # Write test config
            f.write("\nTest Configuration:\n")
            f.write("-" * 30 + "\n")
            f.write(f"Dataset: {dataset_name or 'Unknown'}\n")
            if config_info:
                f.write(f"Model: {config_info.get('model', 'N/A')}\n")
                f.write(f"Samples: {config_info.get('num_samples', 'N/A')}\n")
                f.write(f"Max Tokens: {config_info.get('max_tokens', 'N/A')}\n")
                f.write(f"Engine: V{0 if config_info.get('use_v0') else 1}\n")
            f.write(f"Timestamp: {system_info.get('timestamp', 'N/A')}\n")
            f.write("\n" + "=" * 60 + "\n\n")
            
            # Write results for each scheduler
            for scheduler, data in results.items():
                if "error" in data:
                    f.write(f"{scheduler}: ERROR - {data['error']}\n\n")
                    continue
                
                f.write(f"Scheduler: {scheduler}\n")
                f.write("-" * 30 + "\n")
                f.write(f"Total Requests: {data.get('total_requests', 0)}\n")
                f.write(f"Successful: {data.get('successful_requests', 0)}\n")
                f.write(f"Failed: {data.get('failed_requests', 0)}\n")
                f.write(f"Total Time: {data.get('total_time_seconds', 0):.2f} seconds\n")
                f.write(f"\nLatency (ms):\n")
                f.write(f"  Mean: {data.get('latency_mean_ms', 0):.2f}\n")
                f.write(f"  Median: {data.get('latency_median_ms', 0):.2f}\n")
                f.write(f"  P99: {data.get('latency_p99_ms', 0):.2f}\n")
                f.write(f"\nThroughput:\n")
                f.write(f"  Tokens/sec: {data.get('tokens_per_second_mean', 0):.2f}\n")
                f.write(f"  Requests/sec: {data.get('throughput_requests_per_sec', 0):.2f}\n")
                f.write(f"  Total Tokens: {data.get('total_tokens_generated', 0)}\n")
                f.write(f"  Avg Response Length: {data.get('avg_response_tokens', 0):.1f} tokens\n")
                f.write("\n" + "=" * 60 + "\n\n")
        
        print(f"Summary report saved to {report_path}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark vLLM with ShareGPT dataset")
    parser.add_argument("--model", 
                       default="meta-llama/Llama-3.2-3B",
                       help="Model name or path")
    parser.add_argument("--dataset", 
                       default="../datasets/ShareGPT_V3_unfiltered_cleaned_split.json",
                       help="Path to ShareGPT dataset")
    parser.add_argument("--num-samples", type=int, default=1000,
                       help="Number of samples to test")
    parser.add_argument("--max-tokens", type=int, default=128,
                       help="Maximum tokens to generate per prompt")
    parser.add_argument("--schedulers", nargs="+",
                       default=["scx_lavd", "scx_rusty", "scx_bpfland"],
                       help="List of schedulers to test")
    parser.add_argument("--use-v0", action="store_true",
                       help="Use V0 engine (more stable)")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9,
                       help="GPU memory utilization")
    parser.add_argument("--max-model-len", type=int, default=2048,
                       help="Maximum model context length")
    parser.add_argument("--output-dir", default=None,
                       help="Output directory for results")
    
    args = parser.parse_args()
    
    # Check if dataset exists
    if not os.path.exists(args.dataset):
        # Try alternate location
        alt_dataset = "datasets/ShareGPT_V3_unfiltered_cleaned_split.json"
        if os.path.exists(alt_dataset):
            args.dataset = alt_dataset
        else:
            print(f"Error: Dataset not found at {args.dataset}")
            print("Please download the ShareGPT dataset first")
            sys.exit(1)
    
    # Extract dataset name
    dataset_name = os.path.splitext(os.path.basename(args.dataset))[0]
    
    # Generate output directory
    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = f"results/{dataset_name}_vllm_{timestamp}"
    
    # Try to import scheduler runner
    scheduler_runner = None
    try:
        sys.path.insert(0, '/root/yunwei37/ai-os/scheduler')
        from scheduler_runner import SchedulerRunner
        scheduler_runner = SchedulerRunner()
        print("Scheduler runner loaded successfully")
    except ImportError:
        print("Warning: Scheduler runner not found, will test default scheduler only")
        args.schedulers = None
    
    # Import torch to check CUDA
    try:
        import torch
        if not torch.cuda.is_available():
            print("Warning: CUDA not available, performance will be limited")
    except ImportError:
        print("Warning: PyTorch not properly installed")
    
    # Create benchmark instance
    benchmark = VLLMBenchmark(
        model_path=args.model,
        scheduler_runner=scheduler_runner,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        use_v0=args.use_v0
    )
    
    # Configuration info
    config_info = {
        "model": args.model,
        "num_samples": args.num_samples,
        "max_tokens": args.max_tokens,
        "use_v0": args.use_v0,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "max_model_len": args.max_model_len
    }
    
    # Run benchmarks
    print("Starting ShareGPT benchmark suite for vLLM...")
    print(f"Model: {args.model}")
    print(f"Dataset: {args.dataset}")
    print(f"Samples: {args.num_samples}")
    print(f"Engine: V{0 if args.use_v0 else 1}")
    
    results = benchmark.run_benchmark_suite(
        dataset_path=args.dataset,
        schedulers=args.schedulers,
        num_samples=args.num_samples,
        max_tokens=args.max_tokens
    )
    
    # Generate report
    benchmark.generate_report(results, args.output_dir, dataset_name, config_info)
    
    print("\nBenchmark complete!")
    print(f"Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()