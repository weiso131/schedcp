# Makefile for PyTorch UVM Allocator

CUDA_PATH ?= /usr/local/cuda
CC = gcc
CFLAGS = -shared -fPIC -O3
INCLUDES = -I$(CUDA_PATH)/include
LDFLAGS = -L$(CUDA_PATH)/lib64 -lcudart

TARGET = uvm_allocator.so
SOURCE = uvm_allocator.c

.PHONY: all clean test

all: $(TARGET)

$(TARGET): $(SOURCE)
	$(CC) $(CFLAGS) $(INCLUDES) $(SOURCE) $(LDFLAGS) -o $(TARGET)
	@echo "Built $(TARGET) successfully"

clean:
	rm -f $(TARGET)
	@echo "Cleaned build artifacts"

test: $(TARGET)
	@echo "Running UVM allocator tests..."
	source .venv/bin/activate && python pytorch_uvm_allocator.py

help:
	@echo "PyTorch UVM Allocator Makefile"
	@echo ""
	@echo "Targets:"
	@echo "  all    - Build the UVM allocator shared library (default)"
	@echo "  clean  - Remove built artifacts"
	@echo "  test   - Build and run tests"
	@echo "  help   - Show this help message"
