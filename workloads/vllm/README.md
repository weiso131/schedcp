# run vllm on different scheduler


vllm serve Qwen/Qwen3-Next-80B-A3B-Instruct-FP8  --cpu-offload-gb 30

vllm serve Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8  --cpu-offload-gb 30

source ~/workspace/.venv/bin/activate && vllm bench serve --model Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8 --dataset-name sharegpt --num-prompts 100 --dataset_path /home/yunwei37/workspace/schedcp/workloads/vllm/datasets/ShareGPT_V3_unfiltered_cleaned_split.json

## 30b model

## On off cpu analysis

baseline:

```
First time without cache:
============ Serving Benchmark Result ============
---------------Time to First Token----------------
Mean TTFT (ms):                          26484.33  
Median TTFT (ms):                        26525.51  
P99 TTFT (ms):                           47519.02  
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          1007.14   
Median TPOT (ms):                        671.52    
P99 TPOT (ms):                           4050.65   
---------------Inter-token Latency----------------
Mean ITL (ms):                           607.20    
Median ITL (ms):                         564.65    
P99 ITL (ms):                            4057.08   
==================================================
Second time with cache:
GPU KV cache usage: 5.9%, Prefix cache hit rate: 64.1%
============ Serving Benchmark Result ============
Mean TTFT (ms):                          1968.72   
Median TTFT (ms):                        1985.99   
P99 TTFT (ms):                           1997.20   
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          557.48    
Median TPOT (ms):                        572.26    
P99 TPOT (ms):                           584.75    
---------------Inter-token Latency----------------
Mean ITL (ms):                           528.56    
Median ITL (ms):                         565.86    
P99 ITL (ms):                            606.28    
==================================================
```

PCIe GEN 5@16x RX: 53.88 GiB/s TX: 8.039 GiB/s

Off cpu analysis overhead:

```
Mean TTFT (ms):                          2133.11   
Median TTFT (ms):                        2150.46   
P99 TTFT (ms):                           2165.65   
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          557.58    
Median TPOT (ms):                        572.58    
P99 TPOT (ms):                           584.97    
---------------Inter-token Latency----------------
Mean ITL (ms):                           529.41    
Median ITL (ms):                         566.26    
P99 ITL (ms):                            605.66   
```

On cpu analysis(freq = 1000hz) overhead:

```
============ Serving Benchmark Result ============
---------------Time to First Token----------------
Mean TTFT (ms):                          2100.81   
Median TTFT (ms):                        2117.80   
P99 TTFT (ms):                           2125.77   
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          555.89    
Median TPOT (ms):                        571.05    
P99 TPOT (ms):                           584.30    
---------------Inter-token Latency----------------
Mean ITL (ms):                           526.99    
Median ITL (ms):                         563.93    
P99 ITL (ms):                            601.92    
==================================================
```

Wall clock analysis overhead:

```
Mean TTFT (ms):                          2120.45   
Median TTFT (ms):                        2138.19   
P99 TTFT (ms):                           2146.78   
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          556.52    
Median TPOT (ms):                        571.69    
P99 TPOT (ms):                           584.04    
---------------Inter-token Latency----------------
Mean ITL (ms):                           527.54    
Median ITL (ms):                         566.13    
P99 ITL (ms):                            602.84    
==================================================
```

### cupti overhead

```
Mean TTFT (ms):                          1982.50   
Median TTFT (ms):                        1998.57   
P99 TTFT (ms):                           2011.80   
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          557.85    
Median TPOT (ms):                        573.27    
P99 TPOT (ms):                           583.86    
---------------Inter-token Latency----------------
Mean ITL (ms):                           529.97    
Median ITL (ms):                         566.91    
P99 ITL (ms):                            603.44    
```

