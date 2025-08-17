行，我把“怎么**拿到这些指标**”做成一张可操作的速查表，按：**核心 PMU（按任务）→ offcore/DRAM 定界 → uncore/IMC（按通道/插槽）→ RDT 监控（按租户）→ AMD/ARM 差异** 走。每一步都给到命令 + 参考。

---

# 0) 准备与权限

* 裸机/容器里都能用，但很多计数器需要 root 或放宽：

  ```bash
  # 允许普通用户读取更多 PMU（按需降低）
  sudo sysctl kernel.perf_event_paranoid=1   # 0 或 -1 更宽
  ```

  安全注意见内核文档（perf 可能泄露敏感数据，生产环境谨慎放权）。([Kernel Documentation][1], [Kernel.org][2])

* 列出**这台机器支持的事件名**：

  ```bash
  perf list | less
  ```

（`perf list` 是“事件字典”的权威入口）([man7.org][3])

---

# 1) Core 侧（**按任务**抓“读/写/内存敏感度”）

最少一组能跑通的统计（5–10 秒一个窗口）：

```bash
# 替换 $PID 或 -- <cmd>
sudo perf stat -e \
  instructions,cycles, \
  mem_inst_retired.all_loads,mem_inst_retired.all_stores, \
  longest_lat_cache.miss,cycle_activity.stalls_mem_any, \
  mem_load_l3_miss_retired.local_dram, \
  mem_load_l3_miss_retired.remote_dram, \
  ocr.demand_rfo.l3_miss

   sudo perf stat -e instructions,cycles,mem_inst_retired.all_loads,mem_inst_retired.all_stores,longest_lat_cache.miss,cycle_activity.stalls_total,mem_load_l3_miss_retired.local_dram,mem_load_l3_miss_retired.remote_dram,ocr.demand_rfo.l3_miss numactl --interleave=0,1,2   /root/yunwei37/ai-os/workloads/cxl-micro/double_bandwidth
```

* `all_loads / all_stores`：已退休的 load/store 条数
* `longest_lat_cache.miss`：LLC miss（高延迟路径）
* `stalls_mem_any`：因内存导致的停顿周期
* 这些都是 perf 官方命令族支持的核心计数（事件名字以 `perf list` 为准）。([man7.org][3], [brendangregg.com][4])

---

# 2) Intel：**offcore\_response** 把“打到 DRAM 的读”和“写分配（RFO）导致的 DRAM 读”拆开

示例（本地/远端 DRAM 读 + RFO）：

```bash
sudo perf stat -e \
  'offcore_response.demand_data_rd:llc_miss:local_dram' \
  'offcore_response.demand_data_rd:llc_miss:remote_dram' \
  'offcore_response.rfo:llc_miss:local_dram' \
  -p $PID -- sleep 5
```

* 这些符号事件在 **pmu-events** 里有定义映射，等价于设置 MSR 的过滤位。([Computer Science | Rice University][5])
* 背景与术语（offcore/uncore 概念）有一篇系统讲解可参考。([web.eece.maine.edu][6])
* Intel SDM 提供“性能事件总表/编码入口”，遇到机型差异可查这里（含链接到官方 perfmon events 网站）。([Intel][7])
  （补充：社区讨论也验证了 `LLC-load-misses` 等价映射到 offcore 的常见变体，便于交叉核对。）([Stack Overflow][8])

---

# 3) Uncore / IMC（**按通道/插槽**拿“地面真相”的读/写字节）

**目的**：得到“每条内存通道的实际读写量”，验证是否进入 write-drain、高写占比等。

列出你机器上的 iMC PMU：

```bash
ls /sys/bus/event_source/devices | grep -E 'uncore_imc|imc'
```

常用事件名：`uncore_imc_*/cas_count_read` / `cas_count_write`（一次 CAS ≈ 64B）。([Intel][9])

**按通道统计 5 秒带宽**：

```bash
# 按需添加更多通道编号
sudo perf stat -a -e \
  uncore_imc_0/cas_count_read/,uncore_imc_0/cas_count_write/,\
  uncore_imc_1/cas_count_read/,uncore_imc_1/cas_count_write/ \
  -- sleep 5
```

* 这些计数器在现代内核/平台上通过 perf 直接支持；Intel 工程论坛与文档也有示例与说明。([Intel Community][10], [Intel][9])

> 算法：`BW_read = 64 * ΣCAS_READ / 时间`，`BW_write` 同理（64B 是 DDR 突发宽度）。这能给出**插槽级 R/W 比例**，配合调度/限额做反馈。

---

# 4) RDT（**按租户**监控/限额）：MBM/CMT + CAT/MBA

两种路径：

**(A) 原生 resctrl（内核接口）**

```bash
# 挂载（可选开启 mba_MBps / L3/L2 CDP）
sudo mount -t resctrl resctrl /sys/fs/resctrl -o mba_MBps,cdp

# 建一个组，绑任务
sudo mkdir /sys/fs/resctrl/prod
echo $PID | sudo tee /sys/fs/resctrl/prod/tasks

# 读取该组监控：LLC 占用与内/总带宽（mon_data 下）
cat /sys/fs/resctrl/prod/mon_data/mon_L3_00/llc_occupancy
cat /sys/fs/resctrl/prod/mon_data/mon_L3_00/mbm_total_bytes
cat /sys/fs/resctrl/prod/mon_data/mon_L3_00/mbm_local_bytes

# 配置 CAT / MBA（示例）
echo "L3:0=00ff;MB:0=70" | sudo tee /sys/fs/resctrl/prod/schemata
```

* resctrl 文档与选项（`mba_MBps`、CDP、监控项位置等）见内核/社区文档。([Kernel.org][11], [kernel.googlesource.com][12])

**(B) PQoS（intel-cmt-cat 工具，命令友好）**

```bash
# 监控某 PID 的本地/总内存带宽（MBM）
sudo pqos -I -p "mbl:$PID;mbr:$PID"

# 分配 MBA 限额（比如 60%）
sudo pqos -e "mba:1=60" -a "pid:1=$PID"
```

* `pqos` 的 MBM/MBA 用法与更多示例见官方 wiki 和 manpage。([GitHub][13], [Debian Manpages][14])

---

# 5) AMD / ARM 平台怎么拿

**AMD EPYC（Zen2/3/4/5）**

* **Data Fabric（DF）带宽**：在一些发行版上以 `amd_df/*` 或**原始事件编码**方式暴露；官方 HPC Tuning Guide 给出了 perf 的 raw event 示例（包括 DF 的读写 beats → 带宽推导）。
  示例（文档中的 raw 事件写法，具体数值按 PPR/内核 pmu-events JSON 针对你的代际调整）：

  ```bash
  # 示例来自 AMD HPC 文档（Zen4/9004/9005 系列），以 raw DFEvent 编码：
  sudo perf stat -a -e \
    amd_l3/config=0x0300C00000400090/,amd_l3/config=0x0300C00000401F9a/ \
    -- sleep 5
  # DF raw 事件（若内核未提供符号名，按 PPR/pmu-events JSON 直接给 raw 值）
  ```

  参考里明确提到 DF 目前常用 raw 事件并给出了事件号列表；新内核对 Zen5 也补充了 DF 事件与指标集。([AMD][15], [Phoronix][16])
* 也可以查内核自带的 **pmu-events JSON**（`tools/perf/pmu-events/arch/x86/amdzen*/data-fabric.json`），看看你的代际是否已有符号事件。([GitLab][17])
* AMD 官方/社区也有 perf/PMU 工具仓库辅助排错。([GitHub][18])

**ARM（Neoverse/SoC）**

* **Core PMU**：`L1D_CACHE_REFILL`、`L2D_CACHE_REFILL`、`LL_CACHE_MISS`、`STALL_BACKEND_MEM` 等（`perf list` 查看具体名字）。([Arm Developer][19])
* **互连/内存控制器 PMU**：很多 SoC 有专门的 uncore PMU。

  * 例如 **Arm CMN-600/700** 提供 `arm_cmn` 事件，内核有专门文档说明如何用 perf 访问。([Kernel.org][20])
  * SoC 的 **DMC/EMC（DDR 控制器）** 也常暴露 read\_bytes/write\_bytes 事件（按各厂商文档使用）。([Kernel.org][21], [Arm Developer][22])

---

# 6) 把这些指标“串”进你的调度/限额环

**最小闭环**（每 100–200ms）：

1. **按任务**：用 §1+§2 的 core/offcore 统计出 *内存强度* 与 *R/W 比例*；
2. **按插槽/通道**：用 §3 的 IMC 读写字节得出 **写占比** 与 **是否 write-drain 倾向**；
3. **按租户**（多租户时）：用 §4 的 MBM/CMT 给出每租户 **带宽/LLC 占用**；
4. **动作**：

   * 调度器降低同时运行的 *write-heavy* 任务数量，或缩短它们的 timeslice；
   * 必要时对背景租户下发 **MBA**，对受保护租户加 **CAT ways**；
5. **校验**：监控 P99、IMC 队列高水位/写比例回落，再逐步放松限额。

---

# 7) 常见坑

* **事件名与过滤位**跟微架构/内核版本强相关 → 先 `perf list`，再按机型查 SDM/pmu-events JSON。([man7.org][3], [Intel][7])
* **uncore 名称**在不同代/厂商差异很大（`uncore_imc_*`、`amd_df`、`arm_cmn`…）→ 以本机 `/sys/bus/event_source/devices` 为准；
* **RDT MBM/MBA** 精度与步进随平台不同（`mba_MBps` 需要平台支持）→ 按内核 resctrl 文档和 PQoS 指南配置与校准。([kernel.googlesource.com][12], [Debian Manpages][14])

---

## 参考（边看边做）

* `perf list` / `perf stat` 官方手册与使用示例。([man7.org][3], [brendangregg.com][4])
* Intel SDM + perfmon events 入口（offcore 编码对照）。([Intel][7])
* Offcore/Uncore 概览论文笔记。([web.eece.maine.edu][6])
* Intel iMC/uncore 指南与社区示例（`cas_count_read/write`）。([Intel][9], [Intel Community][10])
* RDT/resctrl 与 PQoS（MBM/CMT/CAT/MBA）。([Kernel.org][11], [kernel.googlesource.com][12], [Debian Manpages][14], [GitHub][13])
* AMD DF/Zen 事件与 HPC Tuning 指南（含 raw 事件样例）。([AMD][15], [Phoronix][16])
* ARM PMU/CMN 文档。([Arm Developer][19], [Kernel.org][20])

---

如果你把 **CPU 型号（如：Intel Sapphire Rapids / AMD 9x04 / Ampere / Neoverse V2）** 发给我，我可以把上面的 `perf stat -e ...` 列表**按你的机器**定制到可以直接拷贝运行的版本（包括 offcore 掩码/uncore 设备名）。

[1]: https://docs.kernel.org/admin-guide/perf-security.html?utm_source=chatgpt.com "Perf events and tool security"
[2]: https://www.kernel.org/doc/html/v5.7/admin-guide/perf-security.html?utm_source=chatgpt.com "Perf Events and tool security"
[3]: https://man7.org/linux/man-pages/man1/perf-list.1.html?utm_source=chatgpt.com "perf-list(1) - Linux manual page"
[4]: https://www.brendangregg.com/perf.html?utm_source=chatgpt.com "Linux perf Examples"
[5]: https://www.cs.rice.edu/~la5/doc/perf-doc/d9/d94/pmu-events_8c_source.html?utm_source=chatgpt.com "Linux Perf: pmu-events/pmu-events.c Source File"
[6]: https://web.eece.maine.edu/~vweaver/projects/perf_events/uncore/offcore_uncore.pdf?utm_source=chatgpt.com "Offcore, Uncore, and Northbridge Performance Events in ..."
[7]: https://www.intel.com/content/www/us/en/developer/articles/technical/intel-sdm.html?utm_source=chatgpt.com "Manuals for Intel® 64 and IA-32 Architectures"
[8]: https://stackoverflow.com/questions/66392569/performance-counters-for-dram-accesses?utm_source=chatgpt.com "Performance Counters for DRAM Accesses - intel"
[9]: https://www.intel.com/content/dam/www/public/us/en/documents/design-guides/xeon-e5-2600-uncore-guide.pdf?utm_source=chatgpt.com "xeon-e5-2600-uncore-guide.pdf"
[10]: https://community.intel.com/t5/Software-Tuning-Performance/Writeback-in-DRAM/m-p/1052796?profile.language=en&utm_source=chatgpt.com "Solved: Linux perf support for uncore"
[11]: https://www.kernel.org/doc/Documentation/x86/resctrl.rst?utm_source=chatgpt.com "resctrl.rst"
[12]: https://kernel.googlesource.com/pub/scm/linux/kernel/git/kvms390/vfio-ccw/%2B/refs/heads/vfio-ccw/Documentation/x86/resctrl.rst?utm_source=chatgpt.com "Documentation/x86/resctrl.rst - pub/scm/linux ..."
[13]: https://github.com/intel/intel-cmt-cat/wiki/MBM-MBA-how-to-guide?utm_source=chatgpt.com "MBM MBA how to guide · intel/intel-cmt-cat Wiki"
[14]: https://manpages.debian.org/unstable/intel-cmt-cat/pqos.8.en.html?utm_source=chatgpt.com "pqos(8) — intel-cmt-cat — Debian unstable"
[15]: https://www.amd.com/content/dam/amd/en/documents/epyc-technical-docs/tuning-guides/58479_amd-epyc-9005-tg-hpc.pdf?utm_source=chatgpt.com "High Performance Computing (HPC) Tuning Guide"
[16]: https://www.phoronix.com/news/AMD-Zen-5-Perf-Events-Linux-613?utm_source=chatgpt.com "New AMD Zen 5 Perf Events Going Into Linux 6.13"
[17]: https://gitlab.elettra.trieste.it/intel_socfpga/linux-socfpga/-/blob/socfpga-6.1.68-lts/tools/perf/pmu-events/arch/x86/amdzen3/data-fabric.json?utm_source=chatgpt.com "tools/perf/pmu-events/arch/x86/amdzen3/data-fabric.json"
[18]: https://github.com/AMDESE/amd-perf-tools?utm_source=chatgpt.com "AMDESE/amd-perf-tools"
[19]: https://developer.arm.com/documentation/101430/latest/Debug-descriptions/Performance-Monitoring-Unit/PMU-events?utm_source=chatgpt.com "PMU events"
[20]: https://www.kernel.org/doc/html/v5.10/admin-guide/perf/arm-cmn.html?utm_source=chatgpt.com "Arm Coherent Mesh Network PMU"
[21]: https://www.kernel.org/doc/html/v5.9/admin-guide/perf/hisi-pmu.html?utm_source=chatgpt.com "HiSilicon SoC uncore Performance Monitoring Unit (PMU)"
[22]: https://developer.arm.com/documentation/100180/0103/performance-optimization-and-monitoring/about-the-performance-monitoring-unit?utm_source=chatgpt.com "About the Performance Monitoring Unit"

