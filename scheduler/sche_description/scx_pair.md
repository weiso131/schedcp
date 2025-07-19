# scx_pair

## Overview

A sibling scheduler which ensures that tasks will only ever be co-located on a physical core if they're in the same cgroup. It illustrates how a scheduling policy could be implemented to mitigate CPU bugs, such as L1TF, and also shows how some useful kfuncs such as `scx_bpf_kick_cpu()` can be utilized.

## Description

scx_pair is a demonstration scheduler that implements strict cgroup-based core isolation on systems with SMT (Simultaneous Multi-Threading). The scheduler ensures that only tasks from the same cgroup can run simultaneously on sibling CPUs (hyperthreads) of the same physical core.

The scheduler works by:
1. **Pairing CPU Siblings**: Organizing CPUs into pairs based on their physical core relationships
2. **Cgroup Enforcement**: Ensuring both siblings in a pair only run tasks from the same cgroup
3. **Core Kicking**: Using `scx_bpf_kick_cpu()` to preempt siblings when cgroup constraints are violated

This design was created to demonstrate how sched_ext could be used to quickly implement security mitigations for hardware vulnerabilities like L1TF (L1 Terminal Fault), where data leakage between hyperthreads could occur.

## Features

- **Strict Cgroup Isolation**: Prevents tasks from different cgroups from sharing a physical core
- **SMT-Aware Scheduling**: Understands and enforces constraints on hyperthread pairs
- **Configurable CPU Stride**: Allows customization of CPU pairing strategy
- **Security Mitigation Demo**: Shows how to implement hardware vulnerability mitigations
- **BPF Feature Showcase**: Demonstrates key sched_ext capabilities like CPU kicking

## Use Cases

While this scheduler is only meant to be used to illustrate certain sched_ext features, with a bit more work (e.g. by adding some form of priority handling inside and across cgroups), it could have been used as a way to quickly mitigate L1TF before core scheduling was implemented and rolled out.

Educational purposes:
- Understanding SMT-aware scheduling
- Learning sched_ext BPF features
- Exploring security-focused scheduling policies

## Production Readiness

No - This is a demonstration scheduler intended for educational purposes and feature illustration only.


## Command Line Options

```
/root/yunwei37/ai-os/scheduler/sche_bin/scx_pair: invalid option -- '-'
A demo sched_ext core-scheduler which always makes every sibling CPU pair
execute from the same CPU cgroup.

See the top-level comment in .bpf.c for more details.

Usage: scx_pair [-S STRIDE]

  -S STRIDE     Override CPU pair stride (default: nr_cpus_ids / 2)
  -v            Print libbpf debug messages
  -h            Display this help and exit
```
