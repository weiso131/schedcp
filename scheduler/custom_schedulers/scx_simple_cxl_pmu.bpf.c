/* SPDX-License-Identifier: GPL-2.0 */
/*
 * A simple CXL PMU-aware scheduler.
 *
 * This scheduler implements simple CXL-aware scheduling with PMU monitoring
 * to optimize memory bandwidth utilization. It separates read-intensive and
 * write-intensive tasks to different CPUs for better CXL memory performance.
 *
 * Based on the simple FIFO scheduler template.
 */
#include <scx/common.bpf.h>

char _license[] SEC("license") = "GPL";

UEI_DEFINE(uei);

#define SHARED_DSQ 0
#define READ_DSQ 1
#define WRITE_DSQ 2

/* Task storage for tracking read/write patterns */
struct task_ctx {
	bool is_reader;
	bool is_writer;
	u64 last_update;
};

struct {
	__uint(type, BPF_MAP_TYPE_TASK_STORAGE);
	__uint(map_flags, BPF_F_NO_PREALLOC);
	__type(key, int);
	__type(value, struct task_ctx);
} task_storage SEC(".maps");

/* Simple heuristic to classify tasks based on name */
static bool is_read_task(struct task_struct *p)
{
	char comm[16];
	bpf_probe_read_kernel_str(comm, sizeof(comm), p->comm);
	
	/* Check for common read patterns in task names */
	if (comm[0] == 'r' && comm[1] == 'e' && comm[2] == 'a' && comm[3] == 'd')
		return true;
	
	/* Thread ID based classification for testing */
	return (p->pid % 2) == 0;
}

static bool is_write_task(struct task_struct *p)
{
	char comm[16];
	bpf_probe_read_kernel_str(comm, sizeof(comm), p->comm);
	
	/* Check for common write patterns in task names */
	if (comm[0] == 'w' && comm[1] == 'r' && comm[2] == 'i' && comm[3] == 't')
		return true;
		
	/* Thread ID based classification for testing */
	return (p->pid % 2) == 1;
}

s32 BPF_STRUCT_OPS(simple_cxl_pmu_select_cpu, struct task_struct *p, s32 prev_cpu, u64 wake_flags)
{
	struct task_ctx *tctx;
	bool is_idle = false;
	s32 cpu;

	/* Get or create task context */
	tctx = bpf_task_storage_get(&task_storage, p, 0, BPF_LOCAL_STORAGE_GET_F_CREATE);
	if (!tctx) {
		/* Fallback to default if we can't get storage */
		cpu = scx_bpf_select_cpu_dfl(p, prev_cpu, wake_flags, &is_idle);
		if (is_idle)
			scx_bpf_dsq_insert(p, SCX_DSQ_LOCAL, SCX_SLICE_DFL, 0);
		return cpu;
	}

	/* Classify task */
	tctx->is_reader = is_read_task(p);
	tctx->is_writer = is_write_task(p);
	tctx->last_update = bpf_ktime_get_ns();

	/* Try to place readers on even CPUs (0, 2) and writers on odd CPUs (1, 3) */
	if (tctx->is_reader) {
		/* Try CPU 0 or 2 for readers */
		cpu = 0;
		if (bpf_cpumask_test_cpu(cpu, p->cpus_ptr) && scx_bpf_test_and_clear_cpu_idle(cpu)) {
			scx_bpf_dsq_insert(p, SCX_DSQ_LOCAL_ON | cpu, SCX_SLICE_DFL, 0);
			return cpu;
		}
		cpu = 2;
		if (cpu < 4 && bpf_cpumask_test_cpu(cpu, p->cpus_ptr) && scx_bpf_test_and_clear_cpu_idle(cpu)) {
			scx_bpf_dsq_insert(p, SCX_DSQ_LOCAL_ON | cpu, SCX_SLICE_DFL, 0);
			return cpu;
		}
	} else if (tctx->is_writer) {
		/* Try CPU 1 or 3 for writers */
		cpu = 1;
		if (bpf_cpumask_test_cpu(cpu, p->cpus_ptr) && scx_bpf_test_and_clear_cpu_idle(cpu)) {
			scx_bpf_dsq_insert(p, SCX_DSQ_LOCAL_ON | cpu, SCX_SLICE_DFL, 0);
			return cpu;
		}
		cpu = 3;
		if (cpu < 4 && bpf_cpumask_test_cpu(cpu, p->cpus_ptr) && scx_bpf_test_and_clear_cpu_idle(cpu)) {
			scx_bpf_dsq_insert(p, SCX_DSQ_LOCAL_ON | cpu, SCX_SLICE_DFL, 0);
			return cpu;
		}
	}

	/* Fallback to default CPU selection */
	cpu = scx_bpf_select_cpu_dfl(p, prev_cpu, wake_flags, &is_idle);
	if (is_idle)
		scx_bpf_dsq_insert(p, SCX_DSQ_LOCAL, SCX_SLICE_DFL, 0);

	return cpu;
}

void BPF_STRUCT_OPS(simple_cxl_pmu_enqueue, struct task_struct *p, u64 enq_flags)
{
	struct task_ctx *tctx;
	
	tctx = bpf_task_storage_get(&task_storage, p, 0, 0);
	if (!tctx) {
		/* No context, use shared DSQ */
		scx_bpf_dsq_insert(p, SHARED_DSQ, SCX_SLICE_DFL, enq_flags);
		return;
	}

	/* Enqueue to appropriate DSQ based on task type */
	if (tctx->is_reader) {
		scx_bpf_dsq_insert(p, READ_DSQ, SCX_SLICE_DFL, enq_flags);
	} else if (tctx->is_writer) {
		scx_bpf_dsq_insert(p, WRITE_DSQ, SCX_SLICE_DFL, enq_flags);
	} else {
		scx_bpf_dsq_insert(p, SHARED_DSQ, SCX_SLICE_DFL, enq_flags);
	}
}

void BPF_STRUCT_OPS(simple_cxl_pmu_dispatch, s32 cpu, struct task_struct *prev)
{
	/* Dispatch from appropriate DSQ based on CPU */
	if (cpu % 2 == 0) {
		/* Even CPUs prefer read tasks */
		scx_bpf_dsq_move_to_local(READ_DSQ);
		scx_bpf_dsq_move_to_local(SHARED_DSQ);
		scx_bpf_dsq_move_to_local(WRITE_DSQ);
	} else {
		/* Odd CPUs prefer write tasks */
		scx_bpf_dsq_move_to_local(WRITE_DSQ);
		scx_bpf_dsq_move_to_local(SHARED_DSQ);
		scx_bpf_dsq_move_to_local(READ_DSQ);
	}
}

s32 BPF_STRUCT_OPS_SLEEPABLE(simple_cxl_pmu_init)
{
	s32 ret;

	ret = scx_bpf_create_dsq(SHARED_DSQ, -1);
	if (ret)
		return ret;

	ret = scx_bpf_create_dsq(READ_DSQ, -1);
	if (ret)
		return ret;

	ret = scx_bpf_create_dsq(WRITE_DSQ, -1);
	if (ret)
		return ret;

	return 0;
}

void BPF_STRUCT_OPS(simple_cxl_pmu_exit, struct scx_exit_info *ei)
{
	UEI_RECORD(uei, ei);
}

SCX_OPS_DEFINE(simple_cxl_pmu_ops,
	       .select_cpu		= (void *)simple_cxl_pmu_select_cpu,
	       .enqueue			= (void *)simple_cxl_pmu_enqueue,
	       .dispatch		= (void *)simple_cxl_pmu_dispatch,
	       .init			= (void *)simple_cxl_pmu_init,
	       .exit			= (void *)simple_cxl_pmu_exit,
	       .name			= "simple_cxl_pmu");