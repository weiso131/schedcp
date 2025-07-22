#

copy '/home/yunwei37/ai-os/workloads/processing/schedulers/template' and 
  make a scheduler special for ctest_suite case inside this 
  '/home/yunwei37/ai-os/workloads/processing/test_cases_parallel.json'. read
   '/home/yunwei37/ai-os/workloads/processing/test_cases_parallel.json' 
  first and start write a minimal scheduler. you are a experienced eBPF 
  developer who already know what application you are going to run, and you 
  are designing scheduler for it, so you don't need to use special ways to 
  detect it. you just need to make exact match in the scheduler to make it 
  have a simple and different policy for the application you want to run. 
  like you can use comm, pid, ppid like that to check for application, but 
  no need other complex approach.