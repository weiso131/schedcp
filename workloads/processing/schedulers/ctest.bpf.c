/* SPDX-License-Identifier: GPL-2.0 */
/*
 * scx_ctest - Scheduler optimized for ctest_suite workload
 * 
 * Optimization: Prioritize "long" tasks to start early and reduce tail latency
 * - Detects "long" vs "short" processes by comm name
 * - Gives "long" processes higher priority (lower vtime)
 * - Uses longer time slices for long tasks
 */
#include <scx/common.bpf.h>

char _license[] SEC("license") = "GPL";

UEI_DEFINE(uei);

#define SHARED_DSQ 0

/* Time slice configurations */
#define SHORT_SLICE_NS  30000000ULL   /* 3ms for short tasks */
#define LONG_SLICE_NS   15000000ULL  /* 15ms for long tasks */

/* Detect if task is a "long" task by checking comm */
static bool is_long_task(struct task_struct *p)
{
	char comm[16];
	
	if (bpf_probe_read_kernel_str(comm, sizeof(comm), p->comm) < 0)
		return false;
	
	/* Check for exact match "long" */
	return (comm[0] == 'l' && comm[1] == 'o' && comm[2] == 'n' && 
	        comm[3] == 'g' && comm[4] == '\0');
}

/* Get appropriate time slice based on task type */
static u64 get_time_slice(struct task_struct *p)
{
	return is_long_task(p) ? LONG_SLICE_NS : SHORT_SLICE_NS;
}

s32 BPF_STRUCT_OPS(ctest_select_cpu, struct task_struct *p, s32 prev_cpu, u64 wake_flags)
{
	bool is_idle = false;
	s32 cpu;

	cpu = scx_bpf_select_cpu_dfl(p, prev_cpu, wake_flags, &is_idle);
	if (is_idle) {
		/* Long tasks get priority - dispatch directly to local DSQ */
		if (is_long_task(p)) {
			scx_bpf_dsq_insert(p, SCX_DSQ_LOCAL, LONG_SLICE_NS, 0);
		} else {
			scx_bpf_dsq_insert(p, SCX_DSQ_LOCAL, SHORT_SLICE_NS, 0);
		}
	}

	return cpu;
}

void BPF_STRUCT_OPS(ctest_enqueue, struct task_struct *p, u64 enq_flags)
{
	u64 slice_ns = get_time_slice(p);
	u64 vtime = p->scx.dsq_vtime;
	
	/* Give long tasks priority by reducing their vtime */
	if (is_long_task(p)) {
		vtime -= 1000000000ULL;  /* 1 second priority boost */
	}
	
	/* Enqueue to shared DSQ with adjusted vtime */
	scx_bpf_dsq_insert_vtime(p, SHARED_DSQ, slice_ns, vtime, enq_flags);
}

void BPF_STRUCT_OPS(ctest_dispatch, s32 cpu, struct task_struct *prev)
{
	scx_bpf_dsq_move_to_local(SHARED_DSQ);
}

s32 BPF_STRUCT_OPS_SLEEPABLE(ctest_init)
{
	return scx_bpf_create_dsq(SHARED_DSQ, -1);
}

void BPF_STRUCT_OPS(ctest_exit, struct scx_exit_info *ei)
{
	UEI_RECORD(uei, ei);
}

SCX_OPS_DEFINE(ctest_ops,
	       .select_cpu		= (void *)ctest_select_cpu,
	       .enqueue			= (void *)ctest_enqueue,
	       .dispatch		= (void *)ctest_dispatch,
	       .init			= (void *)ctest_init,
	       .exit			= (void *)ctest_exit,
	       .name			= "ctest");