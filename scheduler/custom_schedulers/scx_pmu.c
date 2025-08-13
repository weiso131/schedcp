/* SPDX-License-Identifier: GPL-2.0 */
/*
 * Simplified CXL PMU-aware scheduler - Userspace component
 */
#include <stdio.h>
#include <unistd.h>
#include <signal.h>
#include <assert.h>
#include <libgen.h>
#include <string.h>
#include <bpf/bpf.h>
#include <scx/common.h>
#include "scx_pmu.bpf.skel.h"

/* Constants */
#define MAX_CPUS 64

/* Memory pattern */
struct memory_pattern {
    __u64 last_access_time;
    __u32 locality_score;
    __u32 access_count;
    bool is_reader;
};

/* Task context */
struct task_ctx {
    __u32 type;
    __u32 priority_boost;
    bool is_memory_intensive;
    __u32 thread_id;
};

/* CPU context */
struct cpu_ctx {
    __u32 active_tasks;
    __u64 last_update;
    bool is_preferred;
};

const char help_fmt[] =
"Simplified CXL PMU-aware Scheduler\n"
"\n"
"A sched_ext scheduler optimized for CXL memory systems with PMU awareness.\n"
"This scheduler separates read and write threads to different CPUs to optimize\n"
"memory bandwidth utilization.\n"
"\n"
"Task Types Supported:\n"
"  - MOE VectorDB tasks (high priority)\n"
"  - Memory bandwidth test tasks (double_bandwidth, mlc, stream)\n"
"  - Regular tasks\n"
"  - Kernel worker threads (kworker - low priority)\n"
"\n"
"CPU Allocation Strategy:\n"
"  - Even thread IDs (readers) → Even CPUs (0, 2)\n"
"  - Odd thread IDs (writers) → Odd CPUs (1, 3)\n"
"\n"
"Usage: %s [-v] [-h]\n"
"\n"
"  -v            Print libbpf debug messages\n"
"  -h            Display this help and exit\n";

static bool verbose;
static volatile int exit_req;

/* Task type names for display */
static const char *task_type_names[] = {
    [0] = "UNKNOWN",
    [1] = "MOE_VECTORDB",
    [2] = "KWORKER",
    [3] = "REGULAR",
    [4] = "BANDWIDTH"
};

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

static void print_cpu_stats(struct scx_sdt *skel)
{
	struct cpu_ctx ctx;
	int nr_cpus = libbpf_num_possible_cpus();
	int values[MAX_CPUS];
	__u32 cpu;
	int ret;
	
	printf("\n=== CPU Statistics ===\n");
	printf("CPU  Active  Preferred  Status\n");
	printf("---  ------  ---------  ------\n");
	
	for (cpu = 0; cpu < 4 && cpu < nr_cpus; cpu++) {
		// For percpu maps, we need to get all CPU values
		ret = bpf_map_lookup_elem(bpf_map__fd(skel->maps.cpu_contexts), 
					  &cpu, values);
		if (ret < 0)
			continue;
		
		// Get the value for this specific CPU
		memcpy(&ctx, &values[cpu * sizeof(struct cpu_ctx)], sizeof(struct cpu_ctx));
		
		printf("%3u  %6u  %9s  %s\n",
		       cpu, 
		       ctx.active_tasks,
		       ctx.is_preferred ? "Yes" : "No",
		       (cpu % 2 == 0) ? "[Read CPU]" : "[Write CPU]");
	}
}

static void print_memory_patterns(struct scx_sdt *skel)
{
	struct memory_pattern pattern;
	struct task_ctx tctx;
	__u32 pid;
	int ret;
	static int print_header = 1;
	
	if (print_header) {
		printf("\n=== Memory Pattern Analysis (Bandwidth Tasks) ===\n");
		printf("PID     Type         Thread  R/W     Access  Locality  Last Access\n");
		printf("-----   ----------   ------  -----   ------  --------  -----------\n");
		print_header = 0;
	}
	
	// This is a simplified version - in real implementation, you would
	// iterate through active PIDs
	for (pid = 1; pid < 32768; pid++) {
		ret = bpf_map_lookup_elem(bpf_map__fd(skel->maps.memory_patterns), 
					  &pid, &pattern);
		if (ret < 0)
			continue;
		
		// Only show bandwidth tasks with recent activity
		if (pattern.access_count > 0 && pattern.last_access_time > 0) {
			// For display purposes, we show simplified info
			printf("%-7u BANDWIDTH    %-6u  %-5s   %-6u  %-8u  %llu\n",
			       pid,
			       pid & 0xff,  // Thread ID
			       pattern.is_reader ? "Read" : "Write",
			       pattern.access_count,
			       pattern.locality_score,
			       pattern.last_access_time / 1000000); // Convert to ms
		}
	}
}

int main(int argc, char **argv)
{
	struct scx_sdt *skel;
	struct bpf_link *link;
	__u32 opt;
	int iter = 0;

	libbpf_set_print(libbpf_print_fn);
	signal(SIGINT, sigint_handler);
	signal(SIGTERM, sigint_handler);

	skel = SCX_OPS_OPEN(cxl_ops, scx_pmu);

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

	SCX_OPS_LOAD(skel, cxl_ops, scx_pmu, uei);
	link = SCX_OPS_ATTACH(skel, cxl_ops, scx_pmu);

	printf("CXL PMU-aware Scheduler started\n");
	printf("Optimizing for memory bandwidth with read/write thread separation\n");
	printf("Press Ctrl-C to exit\n\n");

	while (!exit_req && !UEI_EXITED(skel, uei)) {
		print_cpu_stats(skel);
		
		if (iter % 3 == 0 && verbose) {  // Print memory patterns every 3 seconds in verbose mode
			print_memory_patterns(skel);
		}
		
		fflush(stdout);
		sleep(1);
		iter++;
	}

	bpf_link__destroy(link);
	scx_pmu__destroy(skel);
	
	if (UEI_EXITED(skel, uei)) {
		printf("Scheduler exited\n");
		UEI_REPORT(skel, uei);
		return 1;
	}

	return 0;
}