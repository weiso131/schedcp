
```bash
====================================================================================================
DETAILED COMPARISON: DEFAULT vs DUPLEXOS (scx_nest)
====================================================================================================

Read Heavy (mixed_1_10):
--------------------------------------------------
  Throughput:
    Default:   4,613,149 ops/sec
    DuplexOS:  3,595,037 ops/sec
    Difference: -22.1% (worse)
  P99 Latency:
    Default:      4.40 ms
    DuplexOS:     3.76 ms
    Difference: -14.5% (better)
  GET Operations:
    Default:   4,193,352 ops/sec
    DuplexOS:  3,267,889 ops/sec
    Difference: -22.1%
  SET Operations:
    Default:     419,797 ops/sec
    DuplexOS:    327,148 ops/sec
    Difference: -22.1%

Write Heavy (mixed_10_1):
--------------------------------------------------
  Throughput:
    Default:   1,910,116 ops/sec
    DuplexOS:  1,611,840 ops/sec
    Difference: -15.6% (worse)
  P99 Latency:
    Default:      4.85 ms
    DuplexOS:     5.05 ms
    Difference: +4.3% (worse)
  GET Operations:
    Default:     173,630 ops/sec
    DuplexOS:    146,516 ops/sec
    Difference: -15.6%
  SET Operations:
    Default:   1,736,487 ops/sec
    DuplexOS:  1,465,323 ops/sec
    Difference: -15.6%

Balance Pipeline (pipeline_16):
--------------------------------------------------
  Throughput:
    Default:   1,105,012 ops/sec
    DuplexOS:  1,869,029 ops/sec
    Difference: +69.1% (better)
  P99 Latency:
    Default:      7.17 ms
    DuplexOS:     4.51 ms
    Difference: -37.1% (better)
  GET Operations:
    Default:     552,506 ops/sec
    DuplexOS:    934,515 ops/sec
    Difference: +69.1%
  SET Operations:
    Default:     552,506 ops/sec
    DuplexOS:    934,515 ops/sec
    Difference: +69.1%

Balance Sequential (sequential_pattern):
--------------------------------------------------
  Throughput:
    Default:     765,568 ops/sec
    DuplexOS:  1,916,720 ops/sec
    Difference: +150.4% (better)
  P99 Latency:
    Default:      4.77 ms
    DuplexOS:     4.42 ms
    Difference: -7.4% (better)
  GET Operations:
    Default:     382,784 ops/sec
    DuplexOS:    958,360 ops/sec
    Difference: +150.4%
  SET Operations:
    Default:     382,784 ops/sec
    DuplexOS:    958,360 ops/sec
    Difference: +150.4%

Balance Random (advanced_gaussian_random):
--------------------------------------------------
  Throughput:
    Default:     371,584 ops/sec
    DuplexOS:    421,719 ops/sec
    Difference: +13.5% (better)
  P99 Latency:
    Default:      9.98 ms
    DuplexOS:    11.55 ms
    Difference: +15.7% (worse)
  GET Operations:
    Default:     185,792 ops/sec
    DuplexOS:    210,860 ops/sec
    Difference: +13.5%
  SET Operations:
    Default:     185,792 ops/sec
    DuplexOS:    210,860 ops/sec
    Difference: +13.5%

====================================================================================================
OVERALL SUMMARY
====================================================================================================

Average Throughput:
  Default:   1,753,086 ops/sec
  DuplexOS:  1,882,869 ops/sec
  Difference: +7.4% (better)

Average P99 Latency:
  Default:      6.23 ms
  DuplexOS:     5.86 ms
  Difference: -6.0% (better)

Best improvement for DuplexOS: Balance Sequential (+150.4% throughput)
Worst performance for DuplexOS: Read Heavy (-22.1% throughput)

Analysis complete!
(base) root@gpu01:~/yunwei37/ai-os/workloads/redis/results# 
```