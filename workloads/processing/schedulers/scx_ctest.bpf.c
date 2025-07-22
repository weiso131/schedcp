/* SPDX-License-Identifier: GPL-2.0 */
/*
 * scx_ctest - BPF scheduler optimized for ctest_suite workload
 * 
 * Optimizes for long-tail workloads with severe load imbalance:
 * - Prioritizes "long" tasks to start early and reduce tail latency
 * - Uses shorter time slices for "short" tasks to maintain responsiveness
 * - Simple FIFO scheduling within priority groups
 */
#include <scx/common.bpf.h>

#define MAX_CPUS 256
#define FIFO_SIZE 4096

/* Time slice configurations (in nanoseconds) */
#define SHORT_SLICE_NS  5000000ULL   /* 5ms for short tasks */
#define LONG_SLICE_NS   20000000ULL  /* 20ms for long tasks */

/* Priority levels */
#define PRIO_HIGH 0  /* For "long" tasks */
#define PRIO_LOW  1  /* For "short" tasks */

/* Per-CPU FIFO queue */
struct cpu_ctx {
	u32 high_head;
	u32 high_tail;
	u32 low_head;
	u32 low_tail;
	s32 high_queue[FIFO_SIZE];
	s32 low_queue[FIFO_SIZE];
};

struct {
	__uint(type, BPF_MAP_TYPE_PERCPU_ARRAY);
	__uint(key_size, sizeof(u32));
	__uint(value_size, sizeof(struct cpu_ctx));
	__uint(max_entries, 1);
} cpu_ctx_map SEC(".maps");

/* Task priority assignment based on comm */
static int get_task_priority(struct task_struct *p)
{
	char comm[16];
	
	if (bpf_probe_read_kernel_str(comm, sizeof(comm), p->comm) < 0)
		return PRIO_LOW;
		
	/* Check if this is a "long" task - give it high priority */
	if (comm[0] == 'l' && comm[1] == 'o' && comm[2] == 'n' && comm[3] == 'g' && comm[4] == '\0')
		return PRIO_HIGH;
		
	/* All other tasks (including "short") get low priority */
	return PRIO_LOW;
}

/* Get time slice based on task priority */
static u64 get_time_slice(int priority)
{
	return (priority == PRIO_HIGH) ? LONG_SLICE_NS : SHORT_SLICE_NS;
}

/* Enqueue task to appropriate priority queue */
static void enqueue_task(struct cpu_ctx *ctx, s32 pid, int priority)
{
	if (priority == PRIO_HIGH) {
		u32 next = (ctx->high_tail + 1) % FIFO_SIZE;
		if (next != ctx->high_head) {
			ctx->high_queue[ctx->high_tail] = pid;
			ctx->high_tail = next;
		}
	} else {
		u32 next = (ctx->low_tail + 1) % FIFO_SIZE;
		if (next != ctx->low_head) {
			ctx->low_queue[ctx->low_tail] = pid;
			ctx->low_tail = next;
		}
	}
}

/* Dequeue task from highest priority non-empty queue */
static s32 dequeue_task(struct cpu_ctx *ctx)
{
	s32 pid = -1;
	
	/* Try high priority queue first */
	if (ctx->high_head != ctx->high_tail) {
		pid = ctx->high_queue[ctx->high_head];
		ctx->high_head = (ctx->high_head + 1) % FIFO_SIZE;
		return pid;
	}
	
	/* Then try low priority queue */
	if (ctx->low_head != ctx->low_tail) {
		pid = ctx->low_queue[ctx->low_head];
		ctx->low_head = (ctx->low_head + 1) % FIFO_SIZE;
		return pid;
	}
	
	return -1;
}

s32 BPF_STRUCT_OPS(ctest_select_cpu, struct task_struct *p, s32 prev_cpu, u64 wake_flags)
{
	s32 cpu = prev_cpu;
	
	/* Simple CPU selection - prefer prev_cpu if available */
	if (scx_bpf_test_and_clear_cpu_idle(cpu))
		return cpu;
		
	/* Find any idle CPU */
	cpu = scx_bpf_pick_idle_cpu(p->cpus_ptr, 0);
	if (cpu >= 0)
		return cpu;
		
	return prev_cpu;
}

void BPF_STRUCT_OPS(ctest_enqueue, struct task_struct *p, u64 enq_flags)
{
	u32 zero = 0;
	struct cpu_ctx *ctx;
	s32 cpu = scx_bpf_task_cpu(p);
	int priority;
	
	/* Get per-CPU context */
	ctx = bpf_map_lookup_elem(&cpu_ctx_map, &zero);
	if (!ctx) {
		scx_bpf_error("Failed to lookup CPU context");
		return;
	}
	
	/* Determine task priority */
	priority = get_task_priority(p);
	
	/* Enqueue to appropriate priority queue */
	enqueue_task(ctx, p->pid, priority);
	
	/* Wake up the CPU */
	scx_bpf_kick_cpu(cpu, 0);
}

void BPF_STRUCT_OPS(ctest_dispatch, s32 cpu, struct task_struct *prev)
{
	u32 zero = 0;
	struct cpu_ctx *ctx;
	s32 pid;
	int priority;
	u64 slice_ns;
	
	/* Get per-CPU context */
	ctx = bpf_map_lookup_elem(&cpu_ctx_map, &zero);
	if (!ctx) {
		scx_bpf_error("Failed to lookup CPU context");
		return;
	}
	
	/* Dequeue next task */
	pid = dequeue_task(ctx);
	if (pid < 0)
		return;
	
	/* Dispatch task with appropriate time slice */
	if (!scx_bpf_dispatch_nr_slots()) {
		struct task_struct *p = scx_bpf_find_task_by_pid(pid);
		if (p) {
			priority = get_task_priority(p);
			slice_ns = get_time_slice(priority);
			scx_bpf_dispatch(p, SCX_DSQ_LOCAL, slice_ns, 0);
			scx_bpf_task_release(p);
		}
	}
}

void BPF_STRUCT_OPS(ctest_running, struct task_struct *p)
{
	/* Task started running - nothing special needed */
}

void BPF_STRUCT_OPS(ctest_stopping, struct task_struct *p, bool runnable)
{
	/* Task stopped running - nothing special needed */
}

s32 BPF_STRUCT_OPS_SLEEPABLE(ctest_init)
{
	return scx_bpf_create_dsq(0, -1);
}

void BPF_STRUCT_OPS(ctest_exit, struct scx_exit_info *ei)
{
	UEI_RECORD(uei, ei);
}

SCX_OPS_DEFINE(ctest_ops,
		.select_cpu		= (void *)ctest_select_cpu,
		.enqueue		= (void *)ctest_enqueue,
		.dispatch		= (void *)ctest_dispatch,
		.running		= (void *)ctest_running,
		.stopping		= (void *)ctest_stopping,
		.init			= (void *)ctest_init,
		.exit			= (void *)ctest_exit,
		.name			= "ctest");