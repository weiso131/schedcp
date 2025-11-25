#!/usr/bin/env python3
import matplotlib.pyplot as plt
import json
import numpy as np
from pathlib import Path

# 读取所有结果文件
results_dir = Path(__file__).parent
result_files = list(results_dir.glob("*.json"))

# 解析文件名获取简短标签
def get_label(filename):
    name = filename.stem
    # 提取关键信息
    if "prefetch_adaptive" in name:
        return "SIFT100M UVM+Prefetch adaptive tree"
    elif "SIFT100M" in name and "uvm_baseline" in name:
        return "SIFT100M UVM Baseline"
    elif "SIFT50M" in name:
        return "SIFT50M UVM"
    elif "SIFT20M" in name and "cpu" in name:
        return "SIFT20M CPU only"
    elif "SIFT20M" in name:
        return "SIFT20M GPU only (No oversubscript)"
    else:
        return name

# 加载所有数据
data = {}
for f in result_files:
    with open(f) as fp:
        data[get_label(f)] = json.load(fp)

# 配色
colors = plt.cm.tab10.colors

# ==================== 图1: Build Index 进度 (按vectors_added) ====================
fig1, ax1 = plt.subplots(figsize=(10, 6))

for i, (label, d) in enumerate(data.items()):
    progress = d["index_add"]["progress"]
    vectors = [p["vectors_added"] / 1e6 for p in progress]  # 转为百万
    times = [p["time"] for p in progress]
    ax1.plot(vectors, times, marker='o', markersize=3, label=label, color=colors[i % len(colors)])

ax1.set_xlabel("Vectors Added (millions)")
ax1.set_ylabel("Time (seconds)")
ax1.set_title("FAISS Index Build Time Comparison")
ax1.legend(loc="upper left")
ax1.grid(True, alpha=0.3)

fig1.tight_layout()
fig1.savefig(results_dir / "faiss_buildindex.png", dpi=150)
print(f"Saved: faiss_buildindex.png")

# ==================== 图2: Search 性能 (Bar Chart) ====================
fig2, (ax2a, ax2b) = plt.subplots(1, 2, figsize=(14, 5))

# 获取所有nprobe值
all_nprobes = sorted(set(s["nprobe"] for d in data.values() for s in d["search"]))
labels = list(data.keys())
n_labels = len(labels)
x = np.arange(len(all_nprobes))
width = 0.8 / n_labels

# 左图: QPS vs nprobe (bar)
for i, label in enumerate(labels):
    search = data[label]["search"]
    qps_dict = {s["nprobe"]: s["qps"] for s in search}
    qps_values = [qps_dict.get(np, 0) for np in all_nprobes]
    offset = (i - n_labels/2 + 0.5) * width
    ax2a.bar(x + offset, qps_values, width, label=label, color=colors[i % len(colors)])

ax2a.set_xlabel("nprobe")
ax2a.set_ylabel("QPS (queries per second)")
ax2a.set_title("Search QPS vs nprobe")
ax2a.set_yscale('log')
ax2a.set_xticks(x)
ax2a.set_xticklabels(all_nprobes)
ax2a.legend(loc="upper right")
ax2a.grid(True, alpha=0.3, axis='y')

# 右图: Search Time vs nprobe (bar)
for i, label in enumerate(labels):
    search = data[label]["search"]
    time_dict = {s["nprobe"]: s["search_time"] for s in search}
    search_times = [time_dict.get(np, 0) for np in all_nprobes]
    offset = (i - n_labels/2 + 0.5) * width
    ax2b.bar(x + offset, search_times, width, label=label, color=colors[i % len(colors)])

ax2b.set_xlabel("nprobe")
ax2b.set_ylabel("Search Time (seconds)")
ax2b.set_title("Search Time vs nprobe")
ax2b.set_yscale('log')
ax2b.set_xticks(x)
ax2b.set_xticklabels(all_nprobes)
ax2b.legend(loc="upper left")
ax2b.grid(True, alpha=0.3, axis='y')

fig2.tight_layout()
fig2.savefig(results_dir / "faiss_search.png", dpi=150)
print(f"Saved: faiss_search.png")

plt.show()
