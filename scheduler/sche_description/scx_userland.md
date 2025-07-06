# scx_userland


### Overview

A simple weighted vtime scheduler where all scheduling decisions take place in
user space. This is in contrast to Rusty, where load balancing lives in user
space, but scheduling decisions are still made in the kernel.

### Typical Use Case

There are many advantages to writing schedulers in user space. For example, you
can use a debugger, you can write the scheduler in Rust, and you can use data
structures bundled with your favorite library.

On the other hand, user space scheduling can be hard to get right. You can
potentially deadlock due to not scheduling a task that's required for the
scheduler itself to make forward progress (though the sched_ext watchdog will
protect the system by unloading your scheduler after a timeout if that
happens). You also have to bootstrap some communication protocol between the
kernel and user space.

A more robust solution to this would be building a user space scheduling
framework that abstracts much of this complexity away from you.

### Production Ready?

No. This scheduler uses an ordered list for vtime scheduling, and is strictly
less performant than just using something like `scx_simple`. It is purely
meant to illustrate that it's possible to build a user space scheduler on
top of sched_ext.
