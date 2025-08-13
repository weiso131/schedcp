/* SPDX-License-Identifier: GPL-2.0 */
/*
 * Simple CXL PMU-aware scheduler - Userspace component
 * 
 * A simplified scheduler that demonstrates CXL memory optimization
 * by separating read and write tasks to different CPUs.
 */
#include <stdio.h>
#include <unistd.h>
#include <signal.h>
#include <libgen.h>
#include <bpf/bpf.h>
#include <scx/common.h>
#include "scx_simple_cxl_pmu.bpf.skel.h"

const char help_fmt[] =
"A simple CXL PMU-aware scheduler.\n"
"\n"
"This scheduler demonstrates basic CXL memory optimization by:\n"
"  - Separating read-intensive tasks to even CPUs (0, 2)\n"
"  - Separating write-intensive tasks to odd CPUs (1, 3)\n"
"  - Using separate dispatch queues for read/write tasks\n"
"\n"
"See the top-level comment in .bpf.c for more details.\n"
"\n"
"Usage: %s [-v] [-h]\n"
"\n"
"  -v            Enable verbose output\n"
"  -h            Display this help and exit\n";

static bool verbose;
static volatile int exit_req;

static void sigint_handler(int sig)
{
	exit_req = 1;
}

static int libbpf_print_fn(enum libbpf_print_level level, const char *format, va_list args)
{
	if (level == LIBBPF_DEBUG && !verbose)
		return 0;
	return vfprintf(stderr, format, args);
}

int main(int argc, char **argv)
{
	struct scx_simple_cxl_pmu *skel;
	struct bpf_link *link;
	int opt;

	libbpf_set_print(libbpf_print_fn);
	signal(SIGINT, sigint_handler);
	signal(SIGTERM, sigint_handler);

	/* Parse command line options */
	while ((opt = getopt(argc, argv, "vh")) != -1) {
		switch (opt) {
		case 'v':
			verbose = true;
			break;
		case 'h':
		default:
			fprintf(stderr, help_fmt, basename(argv[0]));
			return opt != 'h';
		}
	}

	/* Open and load BPF program */
	skel = SCX_OPS_OPEN(simple_cxl_pmu_ops, scx_simple_cxl_pmu);
	SCX_OPS_LOAD(skel, simple_cxl_pmu_ops, scx_simple_cxl_pmu, uei);
	
	/* Attach the scheduler */
	link = SCX_OPS_ATTACH(skel, simple_cxl_pmu_ops, scx_simple_cxl_pmu);

	printf("Simple CXL PMU scheduler running. Press Ctrl-C to exit.\n");
	if (verbose) {
		printf("Configuration:\n");
		printf("  - Read tasks → Even CPUs (0, 2)\n");
		printf("  - Write tasks → Odd CPUs (1, 3)\n");
		printf("  - Task classification based on PID (even=read, odd=write)\n");
	}

	/* Main loop */
	while (!exit_req && !UEI_EXITED(skel, uei)) {
		sleep(1);
	}

	/* Cleanup */
	bpf_link__destroy(link);
	UEI_REPORT(skel, uei);
	scx_simple_cxl_pmu__destroy(skel);

	return 0;
}