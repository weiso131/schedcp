/* SPDX-License-Identifier: (LGPL-2.1 OR BSD-2-Clause) */
#ifndef __RUNQSLOWER_H
#define __RUNQSLOWER_H

#define TASK_COMM_LEN 16

struct event {
	char task[TASK_COMM_LEN];
	char prev_task[TASK_COMM_LEN];
	__u64 delta_us;
	__s32 pid;
	__s32 prev_pid;
	__u32 cpu;
	/* Raw PMU counter values */
	__u64 pmu_counter;
	__u64 pmu_enabled;
	__u64 pmu_running;
};

#endif /* __RUNQSLOWER_H */
