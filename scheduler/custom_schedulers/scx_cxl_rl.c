/* SPDX-License-Identifier: GPL-2.0 */
/*
 * CXL RL Scheduler - Userspace component
 * 使用强化学习策略进行CXL内存带宽优化调度
 */
#include <stdio.h>
#include <unistd.h>
#include <signal.h>
#include <assert.h>
#include <libgen.h>
#include <string.h>
#include <bpf/bpf.h>
#include <scx/common.h>
#include "scx_cxl_rl.bpf.skel.h"

/* RL States and Actions */
#define MAX_ACTIONS 8
#define MAX_STATES 8

/* Q-value entry */
struct q_entry {
    int values[MAX_ACTIONS];
};

/* CPU context */
struct cpu_ctx {
    __u32 nr_tasks;
    __u32 nr_readers;
    __u32 nr_writers;
    bool is_cxl;
};

const char help_fmt[] =
"CXL RL (Reinforcement Learning) Scheduler\n"
"\n"
"A sched_ext scheduler that uses reinforcement learning to optimize\n"
"CXL memory bandwidth allocation and task placement.\n"
"\n"
"Usage: %s [-v] [-h]\n"
"\n"
"  -v            Print libbpf debug messages\n"
"  -h            Display this help and exit\n";

static bool verbose;
static volatile int exit_req;

static int libbpf_print_fn(enum libbpf_print_level level, const char *format, va_list args)
{
	if (level == LIBBPF_DEBUG && !verbose)
		return 0;
	return vfprintf(stderr, format, args);
}

static void sigint_handler(int simple)
{
	exit_req = 1;
}

static void print_q_table(struct scx_cxl_rl *skel)
{
	struct q_entry entry;
	__u32 state;
	int ret;
	
	printf("\n=== Q-Table State ===\n");
	for (state = 0; state < 8; state++) {
		ret = bpf_map_lookup_elem(bpf_map__fd(skel->maps.q_table), 
					  &state, &entry);
		if (ret < 0)
			continue;
		
		const char *state_names[] = {
			"LOW_BW", "HIGH_BW", "READ_HEAVY", "WRITE_HEAVY",
			"BALANCED", "CONTENDED", "IDLE", "MAX"
		};
		
		printf("State %s: ", state_names[state]);
		for (int i = 0; i < 8; i++) {
			printf("%4d ", entry.values[i]);
		}
		printf("\n");
	}
}

static void print_cpu_stats(struct scx_cxl_rl *skel)
{
	struct cpu_ctx ctx;
	__u32 cpu;
	int ret;
	
	printf("\n=== CPU Statistics ===\n");
	for (cpu = 0; cpu < 4; cpu++) {
		ret = bpf_map_lookup_elem(bpf_map__fd(skel->maps.cpu_ctxs), 
					  &cpu, &ctx);
		if (ret < 0)
			continue;
		
		printf("CPU %u: tasks=%u readers=%u writers=%u %s\n",
		       cpu, ctx.nr_tasks, ctx.nr_readers, ctx.nr_writers,
		       ctx.is_cxl ? "[CXL]" : "[LOCAL]");
	}
}

int main(int argc, char **argv)
{
	struct scx_cxl_rl *skel;
	struct bpf_link *link;
	__u32 opt;
	int iter = 0;

	libbpf_set_print(libbpf_print_fn);
	signal(SIGINT, sigint_handler);
	signal(SIGTERM, sigint_handler);

	skel = SCX_OPS_OPEN(cxl_rl_ops, scx_cxl_rl);

	while ((opt = getopt(argc, argv, "vh")) != -1) {
		switch (opt) {
		case 'v':
			verbose = true;
			break;
		default:
			fprintf(stderr, help_fmt, basename(argv[0]));
			return opt != 'h';
		}
	}

	SCX_OPS_LOAD(skel, cxl_rl_ops, scx_cxl_rl, uei);
	link = SCX_OPS_ATTACH(skel, cxl_rl_ops, scx_cxl_rl);

	printf("CXL RL Scheduler started\n");
	printf("Press Ctrl-C to exit\n\n");

	while (!exit_req && !UEI_EXITED(skel, uei)) {
		if (iter % 5 == 0) {  // Print stats every 5 seconds
			print_cpu_stats(skel);
			if (verbose) {
				print_q_table(skel);
			}
		}
		fflush(stdout);
		sleep(1);
		iter++;
	}

	bpf_link__destroy(link);
	scx_cxl_rl__destroy(skel);
	
	if (UEI_EXITED(skel, uei)) {
		printf("Scheduler exited\n");
		UEI_REPORT(skel, uei);
		return 1;
	}

	return 0;
}