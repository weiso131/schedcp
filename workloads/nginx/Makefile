.PHONY: all clone build configure benchmark clean install deps wrk2

# Nginx configuration
NGINX_REPO = https://github.com/nginx/nginx.git
NGINX_DIR = nginx
NGINX_BRANCH = master
WRK2_REPO = https://github.com/giltene/wrk2.git
WRK2_DIR = wrk2
BUILD_THREADS = $(shell nproc)

all: deps clone-all build configure

deps:
	@echo "Installing Nginx dependencies..."
	sudo apt-get update
	sudo apt-get install -y build-essential zlib1g-dev libpcre3-dev libssl-dev \
		libgd-dev libxml2-dev libxslt1-dev libgeoip-dev lua5.1 liblua5.1-0-dev

clone-all: clone-nginx clone-wrk2

clone-nginx:
	@echo "Cloning Nginx repository..."
	if [ ! -d $(NGINX_DIR) ]; then \
		git clone --depth=1 --branch=$(NGINX_BRANCH) $(NGINX_REPO) $(NGINX_DIR); \
	fi

clone-wrk2:
	@echo "Cloning wrk2 repository..."
	if [ ! -d $(WRK2_DIR) ]; then \
		git clone --depth=1 $(WRK2_REPO) $(WRK2_DIR); \
	fi

build: clone-all build-nginx build-wrk2

build-nginx: clone-nginx
	@echo "Building Nginx..."
	cd $(NGINX_DIR) && ./auto/configure \
		--prefix=/tmp/nginx \
		--sbin-path=/tmp/nginx/nginx \
		--conf-path=/tmp/nginx/nginx.conf \
		--pid-path=/tmp/nginx/nginx.pid \
		--lock-path=/tmp/nginx/nginx.lock \
		--error-log-path=/dev/null \
		--http-log-path=/dev/null \
		--with-http_ssl_module \
		--with-http_v2_module \
		--with-http_realip_module \
		--with-http_gzip_module \
		--with-http_stub_status_module \
		--with-threads \
		--with-file-aio
	cd $(NGINX_DIR) && make -j$(BUILD_THREADS)

build-wrk2: clone-wrk2
	@echo "Building wrk2..."
	cd $(WRK2_DIR) && make -j$(BUILD_THREADS)

configure:
	@echo "Configuring Nginx..."
	mkdir -p /tmp/nginx/html
	echo "<html><body>Hello World</body></html>" > /tmp/nginx/html/index.html
	cp nginx.conf /tmp/nginx/nginx.conf

install: build
	@echo "Installing Nginx..."
	cd $(NGINX_DIR) && make install

start:
	@echo "Starting Nginx server..."
	/tmp/nginx/nginx -c /tmp/nginx/nginx.conf
	sleep 2

stop:
	@echo "Stopping Nginx server..."
	/tmp/nginx/nginx -s quit 2>/dev/null || true
	pkill -f nginx || true

benchmark: start
	@echo "Running Nginx benchmark with wrk2..."
	python3 nginx_benchmark.py

quick-benchmark: start
	@echo "Running quick Nginx benchmark..."
	./$(WRK2_DIR)/wrk -t4 -c100 -d30s -R2000 http://127.0.0.1:8080/

clean:
	@echo "Cleaning up..."
	cd $(NGINX_DIR) && make clean || true
	cd $(WRK2_DIR) && make clean || true
	rm -rf /tmp/nginx
	rm -rf $(NGINX_DIR) $(WRK2_DIR)

test: build
	@echo "Testing Nginx configuration..."
	/tmp/nginx/nginx -t -c /tmp/nginx/nginx.conf