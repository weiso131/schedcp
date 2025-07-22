/* SPDX-License-Identifier: GPL-2.0 */
/*
 * General BPF scheduler loader - dynamically loads any .bpf.o file
 *
 * Copyright (c) 2022 Meta Platforms, Inc. and affiliates.
 * Copyright (c) 2022 Tejun Heo <tj@kernel.org>
 * Copyright (c) 2022 David Vernet <dvernet@meta.com>
 */
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <signal.h>
#include <libgen.h>
#include <string.h>
#include <errno.h>
#include <bpf/bpf.h>
#include <bpf/libbpf.h>
#include <scx/common.h>

const char help_fmt[] =
"A general BPF scheduler loader.\n"
"\n"
"Loads any BPF scheduler from .bpf.o file.\n"
"\n"
"Usage: %s <bpf_object_file> [-v]\n"
"\n"
"  <bpf_object_file>  Path to the .bpf.o file to load\n"
"  -v                 Print libbpf debug messages\n"
"  -h                 Display this help and exit\n";

static bool verbose;
static volatile int exit_req;

static struct {
	struct bpf_object *obj;
	struct bpf_link *link;
	char *sched_name;
} current_sched;

static int libbpf_print_fn(enum libbpf_print_level level, const char *format, va_list args)
{
	if (level == LIBBPF_DEBUG && !verbose)
		return 0;
	return vfprintf(stderr, format, args);
}

static void sigint_handler(int sig)
{
	exit_req = 1;
}

static int load_bpf_scheduler(const char *obj_path)
{
	struct bpf_object *obj;
	struct bpf_program *prog;
	struct bpf_link *link;
	int err;

	obj = bpf_object__open(obj_path);
	if (!obj) {
		fprintf(stderr, "Failed to open BPF object: %s\n", strerror(errno));
		return -1;
	}

	err = bpf_object__load(obj);
	if (err) {
		fprintf(stderr, "Failed to load BPF object: %s\n", strerror(-err));
		bpf_object__close(obj);
		return -1;
	}

	prog = bpf_object__find_program_by_name(obj, "simple_ops");
	if (!prog) {
		bpf_object__for_each_program(prog, obj) {
			const char *prog_name = bpf_program__name(prog);
			if (strstr(prog_name, "_ops")) {
				break;
			}
		}
		if (!prog) {
			fprintf(stderr, "Failed to find scheduler ops program\n");
			bpf_object__close(obj);
			return -1;
		}
	}

	link = bpf_program__attach(prog);
	if (!link) {
		fprintf(stderr, "Failed to attach BPF program: %s\n", strerror(errno));
		bpf_object__close(obj);
		return -1;
	}

	current_sched.obj = obj;
	current_sched.link = link;
	current_sched.sched_name = strdup(basename((char *)obj_path));

	printf("BPF scheduler %s loaded successfully\n", current_sched.sched_name);
	return 0;
}

int main(int argc, char **argv)
{
	const char *obj_path = NULL;
	int opt;

	libbpf_set_print(libbpf_print_fn);
	signal(SIGINT, sigint_handler);
	signal(SIGTERM, sigint_handler);

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

	if (optind >= argc) {
		fprintf(stderr, "Error: BPF object file path is required\n\n");
		fprintf(stderr, help_fmt, basename(argv[0]));
		return 1;
	}

	obj_path = argv[optind];

	if (access(obj_path, R_OK) != 0) {
		fprintf(stderr, "Error: Cannot access BPF object file: %s\n", obj_path);
		return 1;
	}

	if (load_bpf_scheduler(obj_path) < 0) {
		return 1;
	}

	printf("Press Ctrl+C to unload scheduler\n");
	while (!exit_req) {
		sleep(1);
	}

	bpf_link__destroy(current_sched.link);
	bpf_object__close(current_sched.obj);
	free(current_sched.sched_name);
	printf("Scheduler %s unloaded\n", current_sched.sched_name ? current_sched.sched_name : "unknown");
	return 0;
}