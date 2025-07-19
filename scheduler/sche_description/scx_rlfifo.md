# scx_rlfifo

This is a single user-defined scheduler used within [sched_ext](https://github.com/sched-ext/scx/tree/main), which is a Linux kernel feature which enables implementing kernel thread schedulers in BPF and dynamically loading them. [Read more about sched_ext](https://github.com/sched-ext/scx/tree/main).

## Overview

scx_rlfifo is a simple Round-Robin scheduler runs in user-space, based on the
scx_rustland_core framework.
It dequeues tasks in FIFO order and assigns dynamic time slices, preempting and
re-enqueuing tasks to achieve basic Round-Robin behavior.

## Typical Use Case

This scheduler is provided as a simple template that can be used as a baseline
to test more complex scheduling policies.

## Production Ready?

Definitely not. Using this scheduler in a production environment is not
recommended, unless there are specific requirements that necessitate a basic
FIFO scheduling approach. Even then, it's still recommended to use the kernel's
SCHED_FIFO real-time class.

## Command Line Options

```

**************************************************************************

WARNING: The purpose of scx_rlfifo is to provide a simple scheduler
implementation based on scx_rustland_core, and it is not intended for
use in production environments. If you want to run a scheduler that makes
decisions in user space, it is recommended to use *scx_rustland* instead.

Please do not open GitHub issues in the event of poor performance, or
scheduler eviction due to a runnable task timeout. However, if running this
scheduler results in a system crash or the entire system becoming unresponsive,
please open a GitHub issue.

**************************************************************************
Error: another sched_ext scheduler is already running
```
