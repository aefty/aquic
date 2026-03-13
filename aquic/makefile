# module load gettext

# Compiler
CC = g++
CFLAGS = -O3 -Wall -std=c++23 -fPIC -Xpreprocessor -fopenmp -I/opt/homebrew/opt/libomp/include \
        -L/opt/homebrew/opt/libomp/lib -lomp 

PYTHON=python3.12

# Detect platform
UNAME_S := $(shell uname -s)

BOOST_INC = -I/opt/homebrew/opt/boost/include
EIGEN_INC = -I/opt/homebrew/opt/eigen/include/eigen3

# Include Pybind11 and Python
PYBIND11_INCLUDE = $(shell ${PYTHON} -m pybind11 --includes)
PYTHON_INCLUDE = $(shell ${PYTHON}-config --includes)
PYTHON_LIB = $(shell ${PYTHON}-config --prefix)/lib


# Linker flags for LAPACK, BLAS, and Python
#PYTHON_LDFLAGS = $(shell python3-config --ldflags | sed 's/-lintl//g')
#LDFLAGS = -llapack -lblas -L$(PYTHON_LIB) -lpython3.11
#SHARED_FLAG = -shared

# MAC
PYTHON_LDFLAGS = $(shell ${PYTHON}-config --ldflags)
LDFLAGS = -llapack -lblas -L$(PYTHON_LIB) -l${PYTHON}
SHARED_FLAG = -dynamiclib



# Output shared library
TARGET = quic_pybind$(shell ${PYTHON}-config --extension-suffix)

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
	@echo "XXXX" $(PYBIND11_INCLUDE)
	$(CC) $(CFLAGS) $(PYBIND11_INCLUDE) $(PYTHON_INCLUDE) $(BOOST_INC) $(EIGEN_INC) -c $< -o $@

# Clean up object files and shared library
clean:
	rm -f $(OBJS) $(TARGET)

# Phony targets (not real files)
.PHONY: all clean