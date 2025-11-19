# LLaMA.cpp Benchmark Testing Documentation

## Test

 /home/yunwei37/workspace/gpu/schedcp/workloads/llama.cpp/build/bin/llama-server --gpt-oss-120b-default -ncmoe 64 -c 65536


 GGML_CUDA_ENABLE_UNIFIED_MEMORY=1 /home/yunwei37/workspace/gpu/schedcp/workloads/llama.cpp/build/bin/llama-server --gpt-oss-120b-default -c 65536

##

numactl --interleave=3 python /root/yunwei37/ai-os/workloads/llama.cpp/llamacpp_bench_start.py > llama_test.log

## on my laptop


 /home/yunwei37/ai-os/workloads/llama.cpp/build/bin/llama-bench  -m /home/yunwei37/ai-os/workloads/llama.cpp/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf -n 1
| model                          |       size |     params | backend    | threads |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | ------: | --------------: | -------------------: |
| llama 1B Q4_K - Medium         | 636.18 MiB |     1.10 B | CPU        |       2 |           pp512 |         58.31 ± 1.75 |
| llama 1B Q4_K - Medium         | 636.18 MiB |     1.10 B | CPU        |       2 |             tg1 |         27.86 ± 1.62 |

build: 5aa1105da (6082)

yunwei37@victoryang00-ASUS-Zenbook-S-14-UX5406SA-UX5406SA:~/ai-os/workloads/llama.cpp/llama.cpp$ /home/yunwei37/ai-os/workloads/llama.cpp/llama.cpp/build/bin/llama-bench -m /home/yunwei37/ai-os/workloads/llama.cpp/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf -n 1

| model                          |       size |     params | backend    | ngl |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --------------: | -------------------: |
get_memory_info: [warning] ext_intel_free_memory is not supported (export/set ZES_ENABLE_SYSMAN=1 to support), use total memory as free memory
get_memory_info: [warning] ext_intel_free_memory is not supported (export/set ZES_ENABLE_SYSMAN=1 to support), use total memory as free memory
get_memory_info: [warning] ext_intel_free_memory is not supported (export/set ZES_ENABLE_SYSMAN=1 to support), use total memory as free memory
| llama 1B Q4_K - Medium         | 636.18 MiB |     1.10 B | SYCL       |  99 |           pp512 |      629.25 ± 288.96 |
| llama 1B Q4_K - Medium         | 636.18 MiB |     1.10 B | SYCL       |  99 |             tg1 |        23.02 ± 12.81 |

build: 5aa1105da (6082)

llama.cpp/build/bin/llama-server  -hf mradermacher/Qwen3-42B-A3B-2507-Thinking-Abliterated-uncensored-TOTAL-RECALL-v2-Medium-MASTER-CODER-i1-GGUF:Q4_K_M

