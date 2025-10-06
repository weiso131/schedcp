# run vllm on different scheduler


vllm serve Qwen/Qwen3-Next-80B-A3B-Instruct-FP8  --cpu-offload-gb 30

vllm serve Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8  --cpu-offload-gb 30

~/workspace/.venv/bin/activate && vllm bench serve --model Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8 --dataset-name sharegpt --num-prompts 100 --dataset_path /home/yunwei37/workspace/schedcp/workloads/vllm/datasets/ShareGPT_V3_unfiltered_cleaned_split.json
