# scx_simple

## Overview

scx_simple is a minimal sched_ext scheduler that demonstrates the fundamental concepts of BPF-based scheduling. It offers two scheduling modes: global weighted virtual time (vtime) scheduling and FIFO scheduling. Despite its simplicity, it provides a functional scheduler that can perform reasonably well under specific conditions.

### Key Features

- **Dual Scheduling Modes**: 
  - **Weighted vtime mode** (default): Tasks are scheduled based on their virtual time, providing fairness
  - **FIFO mode**: Simple first-in-first-out scheduling for straightforward task ordering
- **Global Scheduling Queue**: All CPUs share a single scheduling queue, ensuring system-wide scheduling decisions
- **Statistics Tracking**: Monitors tasks queued to local vs global dispatch queues
- **Minimal Overhead**: Extremely lightweight implementation focusing on core scheduling functionality

### Architecture

The scheduler uses a custom dispatch queue (DSQ) with ID 0 for priority queue operations, as built-in DSQs like SCX_DSQ_GLOBAL cannot be used with vtime-based dispatching. In vtime mode, it maintains fairness by tracking each task's virtual time and preventing excessive budget accumulation during idle periods. The scheduler implements:

1. **CPU Selection**: Uses the default CPU selection logic with idle CPU detection
2. **Task Enqueueing**: Places tasks in either local DSQ (for immediate dispatch) or global shared DSQ
3. **Dispatch**: Moves tasks from the shared DSQ to local CPU queues for execution
4. **vtime Tracking**: Updates virtual time for running tasks to maintain fairness

## Typical Use Case

scx_simple is well-suited for:
- **Single-socket Systems**: Performs best on CPUs with uniform L3 cache topology
- **Educational Purposes**: Excellent for understanding sched_ext fundamentals
- **Simple Workloads**: Works well when scheduling requirements are straightforward
- **Testing and Development**: Useful as a baseline for comparing more complex schedulers

### Limitations

- **FIFO Mode Risks**: In FIFO mode, CPU-saturating threads can starve interactive tasks
- **No Preemption**: Lacks preemption mechanisms, relying on natural task completion
- **Limited NUMA Support**: Not optimized for NUMA architectures or complex cache hierarchies

## Production Ready?

This scheduler could be used in production environments with careful consideration of:
- Hardware must match the single-socket, uniform cache topology constraint
- Workload must tolerate the simplicity of the scheduling policy
- FIFO mode should be used cautiously due to potential starvation issues
- Best suited for environments where predictability is more important than optimization

## Command Line Options

```
A simple sched_ext scheduler.

See the top-level comment in .bpf.c for more details.

Usage: scx_simple [-f] [-v]

  -f            Use FIFO scheduling instead of weighted vtime scheduling
  -v            Print libbpf debug messages
  -h            Display this help and exit
```