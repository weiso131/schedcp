.PHONY: all build test clean run model

all: build

build:
	cargo build --release

test:
	cargo test

clean:
	cargo clean
	rm -f model.pb model.tflite
	rm -rf model/

model:
	cd src/model_dir && python3 create_model.py
	cd src/model_dir && python3 transfer.py

run: build
	./target/release/ml-scheduler

install-deps:
	pip3 install tensorflow numpy

help:
	@echo "ML Scheduler Build Targets:"
	@echo "  make build       - Build the ML scheduler"
	@echo "  make test        - Run tests"
	@echo "  make clean       - Clean build artifacts"
	@echo "  make model       - Generate sample ML model"
	@echo "  make run         - Run the ML scheduler"
	@echo "  make install-deps - Install Python dependencies for model training"