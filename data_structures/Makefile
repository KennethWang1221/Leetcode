# Makefile for Leetcode C++ Solutions

# Compiler and flags
CXX = g++
CXXFLAGS = -std=c++17 -Wall -Wextra -O2

# Build directory
BUILD_DIR = build

# Default target
all: build_all

# Find all C++ source files in current directory
SOURCES = $(wildcard *.cpp)
EXECUTABLES = $(SOURCES:%.cpp=$(BUILD_DIR)/%)

# Build all executables
build_all: $(BUILD_DIR) $(EXECUTABLES)

# Create build directory
$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

# Generic rule to build any executable from its .cpp file
$(BUILD_DIR)/%: %.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -o $@ $<

# Specific target for Merge Intervals
merge_intervals: $(BUILD_DIR)/Leetcode_0056_Merge_Intervals
	./$(BUILD_DIR)/Leetcode_0056_Merge_Intervals

# Run a specific program (usage: make run TARGET=Leetcode_0056_Merge_Intervals)
run: $(BUILD_DIR)/$(TARGET)
	./$(BUILD_DIR)/$(TARGET)

# Compile and run a specific file (usage: make test FILE=Leetcode_0056_Merge_Intervals.cpp)
test: $(BUILD_DIR)/$(basename $(FILE))
	./$(BUILD_DIR)/$(basename $(FILE))

# Quick compile and run for the most recent file
quick: $(BUILD_DIR)/$(basename $(notdir $(shell ls -t *.cpp | head -1)))
	./$(BUILD_DIR)/$(basename $(notdir $(shell ls -t *.cpp | head -1)))

# Clean build files
clean:
	rm -rf $(BUILD_DIR)

# Clean and rebuild
rebuild: clean build_all

# Show help
help:
	@echo "Available targets:"
	@echo "  all              - Build all C++ files"
	@echo "  merge_intervals  - Build and run Leetcode_0056_Merge_Intervals"
	@echo "  run TARGET=name  - Build and run specific target"
	@echo "  test FILE=name.cpp - Build and run specific file"
	@echo "  quick            - Build and run most recent C++ file"
	@echo "  clean            - Remove all executables"
	@echo "  rebuild          - Clean and rebuild all"
	@echo "  help             - Show this help message"
	@echo ""
	@echo "Examples:"
	@echo "  make merge_intervals"
	@echo "  make run TARGET=Leetcode_0056_Merge_Intervals"
	@echo "  make test FILE=Leetcode_0056_Merge_Intervals.cpp"

.PHONY: all clean rebuild help run test quick merge_intervals build_all