// SPDX-License-Identifier: (LGPL-2.1 OR BSD-2-Clause)
// Copyright (c) 2019 Facebook
//
// Based on runqslower(8) from BCC by Ivan Babrou.
// 11-Feb-2020   Andrii Nakryiko   Created this.
#include <argp.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <errno.h>
#include <unistd.h>
#include <sys/syscall.h>
#include <linux/perf_event.h>
#include <bpf/libbpf.h>
#include <bpf/bpf.h>
#include "runqslower.h"
#include "runqslower.bpf.skel.h"

static volatile sig_atomic_t exiting = 0;

struct env {
	pid_t pid;
	pid_t tid;
	__u64 min_us;
	bool verbose;
} env = {
	.min_us = 10,
};

const char *argp_program_version = "runqslower 0.1";
const char *argp_program_bug_address =
	"https://github.com/iovisor/bcc/tree/master/libbpf-tools";
const char argp_program_doc[] =
"Trace high run queue latency.\n"
"\n"
"USAGE: runqslower [--help] [-p PID] [-t TID] [min_us]\n"
"\n"
"EXAMPLES:\n"
"    runqslower         # trace latency higher than 10000 us (default)\n"
"    runqslower 1000    # trace latency higher than 1000 us\n"
"    runqslower -p 123  # trace pid 123\n"
"    runqslower -t 123  # trace tid 123 (use for threads only)\n";

static const struct argp_option opts[] = {
	{ "pid", 'p', "PID", 0, "Process PID to trace", 0 },
	{ "tid", 't', "TID", 0, "Thread TID to trace", 0 },
	{ "verbose", 'v', NULL, 0, "Verbose debug output", 0 },
	{ NULL, 'h', NULL, OPTION_HIDDEN, "Show the full help", 0 },
	{},
};

static error_t parse_arg(int key, char *arg, struct argp_state *state)
{
	static int pos_args;
	int pid;
	long long min_us;

	switch (key) {
	case 'h':
		argp_state_help(state, stderr, ARGP_HELP_STD_HELP);
		break;
	case 'v':
		env.verbose = true;
		break;
	case 'p':
		errno = 0;
		pid = strtol(arg, NULL, 10);
		if (errno || pid <= 0) {
			fprintf(stderr, "Invalid PID: %s\n", arg);
			argp_usage(state);
		}
		env.pid = pid;
		break;
	case 't':
		errno = 0;
		pid = strtol(arg, NULL, 10);
		if (errno || pid <= 0) {
			fprintf(stderr, "Invalid TID: %s\n", arg);
			argp_usage(state);
		}
		env.tid = pid;
		break;
	case ARGP_KEY_ARG:
		if (pos_args++) {
			fprintf(stderr,
				"Unrecognized positional argument: %s\n", arg);
			argp_usage(state);
		}
		errno = 0;
		min_us = strtoll(arg, NULL, 10);
		if (errno || min_us <= 0) {
			fprintf(stderr, "Invalid delay (in us): %s\n", arg);
			argp_usage(state);
		}
		env.min_us = min_us;
		break;
	default:
		return ARGP_ERR_UNKNOWN;
	}
	return 0;
}

static int libbpf_print_fn(enum libbpf_print_level level, const char *format, va_list args)
{
	if (level == LIBBPF_DEBUG && !env.verbose)
		return 0;
	return vfprintf(stderr, format, args);
}

static void sig_int(int signo)
{
	exiting = 1;
}

/* Helper function to format timestamp */
static void str_timestamp(const char *fmt, char *buf, size_t sz)
{
	time_t now;
	struct tm *tm;

	time(&now);
	tm = localtime(&now);
	strftime(buf, sz, fmt, tm);
}

/* Helper function to check if BTF is available for tracepoints */
static int probe_tp_btf(const char *tp_name)
{
	/* For simplicity, we'll always use raw tracepoints */
	return 0;
}

void handle_event(void *ctx, int cpu, void *data, __u32 data_sz)
{
	struct event e;
	char ts[32];

	if (data_sz < sizeof(e)) {
		printf("Error: packet too small\n");
		return;
	}
	/* Copy data as alignment in the perf buffer isn't guaranteed. */
	memcpy(&e, data, sizeof(e));

	str_timestamp("%H:%M:%S", ts, sizeof(ts));

	printf("%-8s %-3u %-16s %-6d %14llu %-16s %-6d %14llu %14llu %14llu\n", 
	       ts, e.cpu, e.task, e.pid, e.delta_us, e.prev_task, e.prev_pid, 
	       e.pmu_counter, e.pmu_enabled, e.pmu_running);
}

void handle_lost_events(void *ctx, int cpu, __u64 lost_cnt)
{
	printf("Lost %llu events on CPU #%d!\n", lost_cnt, cpu);
}

static int open_pmu_counter(int cpu, int type, int config)
{
	struct perf_event_attr attr = {};
	int fd;

	attr.type = type;
	attr.size = sizeof(attr);
	attr.config = config;
	attr.freq = 0;
	attr.sample_period = 0;
	attr.inherit = 0;
	attr.read_format = 0;
	attr.sample_type = 0;
	attr.disabled = 0;

	fd = syscall(__NR_perf_event_open, &attr, -1, cpu, -1, 0);
	if (fd < 0) {
		fprintf(stderr, "Failed to open PMU counter on CPU %d: %s\n", 
		        cpu, strerror(errno));
	}
	return fd;
}

static int setup_pmu_counters(struct runqslower *obj)
{
	int ncpus = libbpf_num_possible_cpus();
	int cpu, fd;

	for (cpu = 0; cpu < ncpus; cpu++) {
		/* Open CPU cycles counter */
		fd = open_pmu_counter(cpu, PERF_TYPE_HARDWARE, PERF_COUNT_HW_CPU_CYCLES);
		if (fd < 0)
			continue; /* Skip if PMU not available on this CPU */
		
		/* Update BPF map with the file descriptor */
		if (bpf_map_update_elem(bpf_map__fd(obj->maps.pmu_counters), 
		                        &cpu, &fd, BPF_ANY) < 0) {
			fprintf(stderr, "Failed to update PMU map for CPU %d\n", cpu);
			close(fd);
		}
	}
	return 0;
}

int main(int argc, char **argv)
{
	static const struct argp argp = {
		.options = opts,
		.parser = parse_arg,
		.doc = argp_program_doc,
	};
	struct perf_buffer *pb = NULL;
	struct runqslower *obj;
	int err;

	err = argp_parse(&argp, argc, argv, 0, NULL, NULL);
	if (err)
		return err;

	libbpf_set_print(libbpf_print_fn);

	obj = runqslower__open();
	if (!obj) {
		fprintf(stderr, "failed to open BPF object\n");
		return 1;
	}

	/* initialize global data (filtering options) */
	obj->rodata->targ_tgid = env.pid;
	obj->rodata->targ_pid = env.tid;
	obj->rodata->min_us = env.min_us;

	if (probe_tp_btf("sched_wakeup")) {
		bpf_program__set_autoload(obj->progs.handle_sched_wakeup, false);
		bpf_program__set_autoload(obj->progs.handle_sched_wakeup_new, false);
		bpf_program__set_autoload(obj->progs.handle_sched_switch, false);
	} else {
		bpf_program__set_autoload(obj->progs.sched_wakeup, false);
		bpf_program__set_autoload(obj->progs.sched_wakeup_new, false);
		bpf_program__set_autoload(obj->progs.sched_switch, false);
	}

	err = runqslower__load(obj);
	if (err) {
		fprintf(stderr, "failed to load BPF object: %d\n", err);
		goto cleanup;
	}

	/* Setup PMU counters after loading BPF program */
	err = setup_pmu_counters(obj);
	if (err < 0) {
		fprintf(stderr, "Warning: PMU counters setup failed, continuing without PMU data\n");
		/* Continue without PMU data */
	}

	err = runqslower__attach(obj);
	if (err) {
		fprintf(stderr, "failed to attach BPF programs\n");
		goto cleanup;
	}

	printf("Tracing run queue latency higher than %llu us\n", env.min_us);
	printf("%-8s %-3s %-16s %-6s %14s %-16s %-6s %14s %14s %14s\n", 
	       "TIME", "CPU", "COMM", "TID", "LAT(us)", "PREV COMM", "PREV TID", 
	       "PMU_COUNTER", "PMU_ENABLED", "PMU_RUNNING");

	pb = perf_buffer__new(bpf_map__fd(obj->maps.events), 64,
			      handle_event, handle_lost_events, NULL, NULL);
	if (!pb) {
		err = -errno;
		fprintf(stderr, "failed to open perf buffer: %d\n", err);
		goto cleanup;
	}

	if (signal(SIGINT, sig_int) == SIG_ERR) {
		fprintf(stderr, "can't set signal handler: %s\n", strerror(errno));
		err = 1;
		goto cleanup;
	}

	while (!exiting) {
		err = perf_buffer__poll(pb, 100);
		if (err < 0 && err != -EINTR) {
			fprintf(stderr, "error polling perf buffer: %s\n", strerror(-err));
			goto cleanup;
		}
		/* reset err to return 0 if exiting */
		err = 0;
	}

cleanup:
	perf_buffer__free(pb);
	runqslower__destroy(obj);

	return err != 0;
}
