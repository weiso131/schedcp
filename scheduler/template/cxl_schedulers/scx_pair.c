/* SPDX-License-Identifier: GPL-2.0 */
/*
 * Working CXL PMU-aware scheduler - Userspace component
 */
#include <stdio.h>
#include <unistd.h>
#include <signal.h>
#include <assert.h>
#include <libgen.h>
#include <string.h>
#include <bpf/bpf.h>
#include <scx/common.h>
#include "scx_pair.bpf.skel.h"

const char help_fmt[] =
"Working CXL PMU-aware Scheduler\n"
"\n"
"A minimal sched_ext scheduler that demonstrates basic functionality.\n"
"This is a simplified version designed to work reliably with current\n"
"sched_ext implementations.\n"
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

int main(int argc, char **argv)
{
	struct scx_pair *skel;
	struct bpf_link *link;
	__u32 opt;
	int iter = 0;

	libbpf_set_print(libbpf_print_fn);
	signal(SIGINT, sigint_handler);
	signal(SIGTERM, sigint_handler);

	skel = SCX_OPS_OPEN(cxl_ops, scx_pair);

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

	SCX_OPS_LOAD(skel, cxl_ops, scx_pair, uei);
	link = SCX_OPS_ATTACH(skel, cxl_ops, scx_pair);

	printf("CXL Working Scheduler started\n");
	printf("This is a minimal scheduler for testing purposes\n");
	printf("Press Ctrl-C to exit\n\n");

	while (!exit_req && !UEI_EXITED(skel, uei)) {
		if (iter % 5 == 0) {  // Print status every 5 seconds
			printf("Scheduler running... [%d seconds]\n", iter);
		}
		fflush(stdout);
		sleep(1);
		iter++;
	}

	bpf_link__destroy(link);
	scx_pair__destroy(skel);
	
	if (UEI_EXITED(skel, uei)) {
		printf("Scheduler exited\n");
		UEI_REPORT(skel, uei);
		return 1;
	}

	return 0;
}