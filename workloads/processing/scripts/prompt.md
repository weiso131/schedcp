# The prompt for generating the scheduler

## analysis

For each desc file, Check the application itself and determin what's your optimization goal, and how to optimize your scheduler to achieve that goal. plan before you start writing the scheduler. makke the file containing a natural language summary of the workload’s purpose, key performance char-acteristics and resource requirements, and explicit optimization goals and algorithm that guide the next stage.

you are a experienced eBPF developer who already know what application you are going to run, and you are designing scheduler for it, so you don't need to use special ways to detect it. you just need to make exact match for the processes name in the scheduler to make it have a simple and different policy for the application you want to run. like you can use comm, pid, ppid like that to check for application, but no need other complex approach.

You just need to update the docs. check the files in current dir and update each of them.

## make scheduler

make a scheduler special for {TEST_CASE_ID} case inside this /home/yunwei37/ai-os/workloads/processing/schedulers. read '/home/yunwei37/ai-os/workloads/processing/test_cases_parallel.json' and the README.md of the application first and start write a minimal scheduler. the loader and makefile are already there, you just need to write the scheduler.

you are a experienced eBPF developer who already know what application you are going to run, and you are designing scheduler for it, so you don't need to use special ways to detect it. you just need to make exact match for the processes in the scheduler to make it have a simple and different policy for the application you want to run. like you can use comm, pid, ppid like that to check for application, but no need other complex approach.
