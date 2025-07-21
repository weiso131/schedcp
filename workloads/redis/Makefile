.PHONY: all clone build configure benchmark clean install deps

# Redis configuration
REDIS_REPO = https://github.com/redis/redis.git
REDIS_DIR = redis
REDIS_BRANCH = 7.2
BUILD_THREADS = $(shell nproc)

all: deps clone build configure

deps:
	@echo "Installing Redis dependencies..."
	sudo apt-get update
	sudo apt-get install -y build-essential tcl pkg-config libsystemd-dev

clone:
	@echo "Cloning Redis repository..."
	if [ ! -d $(REDIS_DIR) ]; then \
		git clone --depth=1 --branch=$(REDIS_BRANCH) $(REDIS_REPO) $(REDIS_DIR); \
	fi

build: clone
	@echo "Building Redis..."
	cd $(REDIS_DIR) && make -j$(BUILD_THREADS) BUILD_TLS=yes

configure:
	@echo "Configuring Redis..."
	cp redis.conf $(REDIS_DIR)/redis.conf

install: build
	@echo "Installing Redis..."
	cd $(REDIS_DIR) && sudo make install

start:
	@echo "Starting Redis server..."
	cd $(REDIS_DIR) && ./src/redis-server redis.conf &
	sleep 2

stop:
	@echo "Stopping Redis server..."
	cd $(REDIS_DIR) && ./src/redis-cli shutdown || true
	pkill -f redis-server || true

benchmark: start
	@echo "Running Redis benchmark..."
	python3 redis_benchmark.py

quick-benchmark: start
	@echo "Running quick Redis benchmark..."
	cd $(REDIS_DIR) && ./src/redis-benchmark -t set,get -n 100000 -c 50 -P 16

clean:
	@echo "Cleaning up..."
	cd $(REDIS_DIR) && make clean || true
	rm -rf $(REDIS_DIR)

test: build
	@echo "Running Redis tests..."
	cd $(REDIS_DIR) && make test