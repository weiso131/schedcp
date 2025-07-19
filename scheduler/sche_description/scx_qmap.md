# scx_qmap

## Overview

scx_qmap is a simple five-level FIFO queue scheduler that demonstrates fundamental sched_ext features. It implements a weighted queuing policy where tasks are assigned to one of five priority queues based on their compound weight. The scheduler uses BPF_MAP_TYPE_QUEUE for task queuing and provides examples of various sched_ext capabilities.

### Key Features

- **Five-Level Priority Queuing**: Tasks are distributed across five FIFO queues (queue0-queue4) based on their weight
- **Weighted Dispatch**: Higher priority queues get more dispatch opportunities (1 from queue0, 2 from queue1, 4 from queue2, etc.)
- **Sleepable Task Storage**: Demonstrates per-task storage allocation using ops.prep_enable()
- **Core Scheduling Support**: Implements core-sched ordering using per-queue sequence numbers
- **CPU Performance Scaling**: Optional CPU performance target adjustment based on queue index
- **High Priority Boosting**: Can boost nice -20 tasks to a dedicated high-priority DSQ

### Architecture

The scheduler maintains five BPF queues implemented as an array of maps. Each CPU round-robins through these queues, dispatching more tasks from higher-indexed queues to provide weighted scheduling. Tasks are enqueued using their PIDs, and the scheduler includes comprehensive error handling and debugging features.

## Typical Use Case

scx_qmap is primarily designed for demonstration and testing of sched_ext features. It serves as an educational example showing how to:
- Implement BPF-side task queueing
- Handle CPU release events when higher priority scheduling classes take over
- Support core scheduling requirements
- Implement weighted scheduling policies

While functional, it is unlikely to be optimal for actual production workloads due to its simplified design focused on illustrating sched_ext capabilities.

## Production Ready?

No. This scheduler is explicitly designed for demonstration and testing purposes. Its simplified five-level FIFO approach, while educational, lacks the sophistication needed for production environments.

## Command Line Options

```
A simple five-level FIFO queue sched_ext scheduler.

See the top-level comment in .bpf.c for more details.

Usage: scx_qmap [-s SLICE_US] [-e COUNT] [-t COUNT] [-T COUNT] [-l COUNT] [-b COUNT]
       [-P] [-d PID] [-D LEN] [-p] [-v]

  -s SLICE_US   Override slice duration
  -e COUNT      Trigger scx_bpf_error() after COUNT enqueues
  -t COUNT      Stall every COUNT'th user thread
  -T COUNT      Stall every COUNT'th kernel thread
  -l COUNT      Trigger dispatch infinite looping after COUNT dispatches
  -b COUNT      Dispatch upto COUNT tasks together
  -P            Print out DSQ content to trace_pipe every second, use with -b
  -H            Boost nice -20 tasks in SHARED_DSQ, use with -b
  -d PID        Disallow a process from switching into SCHED_EXT (-1 for self)
  -D LEN        Set scx_exit_info.dump buffer length
  -S            Suppress qmap-specific debug dump
  -p            Switch only tasks on SCHED_EXT policy instead of all
  -v            Print libbpf debug messages
  -h            Display this help and exit
```