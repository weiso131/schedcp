# scx_sdt

## Overview

scx_sdt is a simple deadline-based scheduler that uses deadline tracking for task scheduling decisions.

## Description

This scheduler implements deadline-based scheduling using the sched_ext framework. It tracks task deadlines and makes scheduling decisions based on which tasks have the earliest deadlines.

## Features

- Deadline-based scheduling
- Simple implementation for educational purposes
- Demonstrates deadline tracking in BPF

## Use Case

This scheduler is useful for workloads that have deadline requirements or for learning about deadline-based scheduling algorithms.

## Production Ready?

This is primarily an example scheduler for educational purposes and may not be suitable for production use without further testing and optimization.
## Command Line Options

```
/root/yunwei37/ai-os/scheduler/sche_bin/scx_sdt: invalid option -- '-'
A simple sched_ext scheduler.

See the top-level comment in .bpf.c for more details.

Usage: scx_sdt [-f] [-v]

  -v            Print libbpf debug messages
  -h            Display this help and exit
```
