build:
	make -C mcp/new_sched
	make -C scheduler
	cd mcp && cargo build --release
	cd autotune && cargo build --release
