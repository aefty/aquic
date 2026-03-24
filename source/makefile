# Compiler
CC = g++
CFLAGS = -O3 -Wall -std=c++23 -fPIC -Xpreprocessor -fopenmp -I/opt/homebrew/opt/libomp/include \
        -L/opt/homebrew/opt/libomp/lib -lomp

PYTHON = python3.12

# Detect platform
UNAME_S := $(shell uname -s)

BOOST_INC = -I/opt/homebrew/opt/boost/include
EIGEN_INC = -I/opt/homebrew/opt/eigen/include/eigen3

# Include Pybind11 and Python
PYBIND11_INCLUDE = $(shell ${PYTHON} -m pybind11 --includes)
PYTHON_INCLUDE  = $(shell ${PYTHON}-config --includes)
PYTHON_LIB      = $(shell ${PYTHON}-config --prefix)/lib

# MAC
PYTHON_LDFLAGS = $(shell ${PYTHON}-config --ldflags)
LDFLAGS = -llapack -lblas -L$(PYTHON_LIB) -l${PYTHON}
SHARED_FLAG = -dynamiclib

# Output shared library
TARGET = quic_pybind$(shell ${PYTHON}-config --extension-suffix)

# Install location: one level up
INSTALL_DIR = ..

# Source files
SRCS = quic_pybind.cpp QUIC.cpp

# Object files
OBJS = $(SRCS:.cpp=.o)

# Default target
all: $(TARGET)

# Compile the shared library
$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) -shared -o $@ $(OBJS) $(LDFLAGS) $(PYTHON_LDFLAGS)

# Compile source files into object files
%.o: %.cpp
	@echo "\n"
	@echo "############################################"
	@echo "PYTHON   Version:   " $(shell $(PYTHON) --version)
	@echo "BOOST    Location:  " $(BOOST_INC)
	@echo "EIGEN    Location:  " $(EIGEN_INC)
	@echo "PYBIND11 Location:  " $(PYBIND11_INCLUDE)
	@echo "############################################"
	@echo "\n"
	$(CC) $(CFLAGS) $(PYBIND11_INCLUDE) $(PYTHON_INCLUDE) $(BOOST_INC) $(EIGEN_INC) -c $< -o $@

# Install: move the built shared library one level up
install: $(TARGET)
	mv -f $(TARGET) $(INSTALL_DIR)/$(TARGET)

# Clean up object files and shared library
clean:
	rm -f $(OBJS) $(TARGET)

# Phony targets
.PHONY: all install clean