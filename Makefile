# Makefile for building and running the sample CUDA program
NVCC ?= nvcc
SRC := sample/vector_add.cu
BIN := sample/vector_add
CFLAGS := -g -G

.PHONY: all build run clean

all: build

build:
	@echo "Building $(SRC) -> $(BIN)"
	$(NVCC) $(CFLAGS) $(SRC) -o $(BIN)

run: build
	@echo "Running $(BIN)"
	$(BIN)

clean:
	@echo "Cleaning"
	rm -f $(BIN)
