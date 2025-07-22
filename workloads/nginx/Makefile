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
		lua5.1 liblua5.1-0-dev python3-psutil

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
		--prefix=$(PWD)/nginx-install \
		--sbin-path=$(PWD)/nginx-install/nginx \
		--conf-path=$(PWD)/nginx-install/nginx.conf \
		--pid-path=$(PWD)/nginx-install/nginx.pid \
		--lock-path=$(PWD)/nginx-install/nginx.lock \
		--error-log-path=/dev/null \
		--http-log-path=/dev/null \
		--with-http_ssl_module \
		--with-http_v2_module \
		--with-http_realip_module \
		--with-http_stub_status_module \
		--with-threads \
		--with-file-aio
	cd $(NGINX_DIR) && make -j$(BUILD_THREADS)
	cd $(NGINX_DIR) && make install
	@echo "Copying nginx binary to local directory..."
	cp $(PWD)/nginx-install/nginx ./nginx-bin
	cp $(PWD)/nginx-install/mime.types ./nginx-mime.types

build-wrk2: clone-wrk2
	@echo "Building wrk2..."
	cd $(WRK2_DIR) && make -j$(BUILD_THREADS)

configure:
	@echo "Configuring Nginx..."
	mkdir -p ./html
	echo "<html><body>Hello World</body></html>" > ./html/index.html
	sed 's|include /tmp/nginx/mime.types;|include $(PWD)/nginx-mime.types;|g' nginx.conf > nginx-local.conf
	sed -i 's|root /tmp/nginx/html;|root $(PWD)/html;|g' nginx-local.conf
	sed -i 's|pid /tmp/nginx/nginx.pid;|pid $(PWD)/nginx.pid;|g' nginx-local.conf

install: build configure
	@echo "Nginx built and configured locally"

start: install
	@echo "Starting Nginx server..."
	./nginx-bin -c $(PWD)/nginx-local.conf
	sleep 2

stop:
	@echo "Stopping Nginx server..."
	./nginx-bin -s quit -c $(PWD)/nginx-local.conf 2>/dev/null || true
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
	rm -rf nginx-install html nginx-bin nginx-mime.types nginx-local.conf nginx.pid results
	rm -rf $(NGINX_DIR) $(WRK2_DIR)

test: install
	@echo "Testing Nginx configuration..."
	./nginx-bin -t -c $(PWD)/nginx-local.conf