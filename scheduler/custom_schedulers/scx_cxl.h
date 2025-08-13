/* SPDX-License-Identifier: GPL-2.0 */
/*
 * CXL Bandwidth-Aware Scheduler Header
 * 
 * Common definitions for CXL scheduler BPF and userspace components
 */

#ifndef __SCX_CXL_H
#define __SCX_CXL_H

#include <stdint.h>
#include <stdbool.h>

#define MAX_CPUS 1024
#define MAX_TASKS 8192
#define MB_TO_BYTES(mb) ((mb) * 1024 * 1024)

enum task_type {
	TASK_TYPE_UNKNOWN = 0,
	TASK_TYPE_MOE_VECTORDB,
	TASK_TYPE_KWORKER,
	TASK_TYPE_REGULAR,
	TASK_TYPE_LATENCY_SENSITIVE,
	TASK_TYPE_READ_INTENSIVE,
	TASK_TYPE_WRITE_INTENSIVE,
	TASK_TYPE_BANDWIDTH_TEST,
};

enum io_pattern {
	IO_PATTERN_UNKNOWN = 0,
	IO_PATTERN_READ_HEAVY,
	IO_PATTERN_WRITE_HEAVY,
	IO_PATTERN_MIXED,
	IO_PATTERN_SEQUENTIAL,
	IO_PATTERN_RANDOM,
};

struct memory_access_pattern {
	uint64_t nr_accesses;
	uint64_t avg_access_size;
	uint64_t total_access_time;
	uint64_t last_access_time;
	uint64_t hot_regions;
	uint64_t cold_regions;
	uint32_t locality_score;
	uint32_t working_set_size;
	uint64_t read_bytes;
	uint64_t write_bytes;
	enum io_pattern io_pattern;
};

struct cxl_pmu_metrics {
	uint64_t memory_bandwidth;
	uint64_t cache_hit_rate;
	uint64_t memory_latency;
	uint64_t cxl_utilization;
	uint64_t read_bandwidth;
	uint64_t write_bandwidth;
	uint64_t last_update_time;
};

struct bandwidth_control {
	uint64_t max_read_bandwidth_mb;
	uint64_t max_write_bandwidth_mb;
	uint64_t token_bucket_size;
	uint64_t refill_interval_ns;
	bool enabled;
};

struct damon_config {
	const char *sysfs_path;
	uint64_t sample_interval_ns;
	uint32_t min_nr_regions;
	uint32_t max_nr_regions;
	bool enabled;
};

struct scheduler_features {
	bool enable_damon;
	bool enable_cxl_aware;
	bool enable_bandwidth_control;
	bool enable_vectordb_optimization;
	bool enable_kworker_promotion;
	bool verbose;
};

struct scheduler_stats {
	uint64_t total_enqueues;
	uint64_t total_dispatches;
	uint64_t vectordb_tasks;
	uint64_t bandwidth_limited_tasks;
	uint64_t damon_updates;
	uint64_t cxl_migrations;
};

#endif /* __SCX_CXL_H */