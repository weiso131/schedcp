#



## API Usage

Deteck file name

```c
#include "pid_filename_header.h"

/* Detect if task is a "long" task by checking filename */
static bool is_long_task(struct task_struct *p)
{
	pid_t pid = p->pid;
	struct filename_info *info;
	char local_filename[MAX_FILENAME_LEN];
	const char *basename;
	int last_slash = -1;
	
	/* Get filename info for this PID */
	info = bpf_map_lookup_elem(&pid_to_filename, &pid);
	if (!info)
		return false;
	
	int is_large = (info->filename[0] == '.' && info->filename[1] == '/' 
		&& info->filename[2] == 'l' && info->filename[3] == 'a' 
		&& info->filename[4] == 'r' && info->filename[5] == 'g' 
		&& info->filename[6] == 'e');
	if (is_large)
		bpf_printk("large task: %s\n", info->filename);
	return is_large;
}
```

Detect comm name

```c
static bool is_long_task(struct task_struct *p)
{
	char comm[16];
	
	if (bpf_probe_read_kernel_str(comm, sizeof(comm), p->comm) < 0)
		return false;
	
	/* Check for start with "large" */
	return (comm[0] == 'l' && comm[1] == 'a' && comm[2] == 'r' && 
	        comm[3] == 'g' && comm[4] == 'e');
}
```