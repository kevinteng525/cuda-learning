CXX := clang++
SRC := main.cpp
BIN := main
CXXFLAGS := -std=c++17 -g -O0

.PHONY: all build run clean

all: build

build:
	@echo "Building $(SRC) -> $(BIN)"
	$(CXX) $(CXXFLAGS) $(SRC) -o $(BIN)

run: build
	@echo "Running $(BIN)"
	./$(BIN)

clean:
	rm -f $(BIN)
