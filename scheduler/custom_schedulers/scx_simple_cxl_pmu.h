/* SPDX-License-Identifier: GPL-2.0 */
/*
 * Simple CXL PMU-aware scheduler - Header file
 *
 * Common definitions shared between BPF and userspace.
 */
#ifndef __SCX_SIMPLE_CXL_PMU_H
#define __SCX_SIMPLE_CXL_PMU_H

/* Dispatch queue IDs */
#define SHARED_DSQ 0
#define READ_DSQ   1
#define WRITE_DSQ  2

/* Task context for tracking read/write patterns */
struct task_ctx {
	bool is_reader;
	bool is_writer;
	__u64 last_update;
};

#endif /* __SCX_SIMPLE_CXL_PMU_H */