

CC = 			mpicxx

CUC =			nvcc

INC_FLAGS = 	-I/home/andres/code/smartsim/Project/libraries/SmartRedis/install/include 

LD_FLAGS =  	-L/home/andres/code/smartsim/Project/libraries/SmartRedis/install/lib \
				-L/home/andres/Software/cuda/cuda_toolkit_12.3/install/toolkit/lib64

CXXFLAGS =		-std=c++17 $(INC_FLAGS)

LIBS = 			-lsmartredis -lcudart

C_FILES =		main.c \
				c_communicator.c

CPP_FILES =		c_client.cpp \
				c_configoptions.cpp \
				c_dataset.cpp \
				c_error.cpp \
				c_logcontext.cpp \
				c_logger.cpp

CU_FILES =		cuda_test.cu

C_OBJS = 		main.o \
				c_communicator.o

CPP_OBJS = 		c_client.o \
				c_configoptions.o \
				c_dataset.o \
				c_error.o \
				c_logcontext.o \
				c_logger.o

CU_OBJS = 		cuda_test.o

EXECUTABLE =	program

# Default target
all: $(C_OBJS) $(CPP_OBJS) $(CU_OBJS)
	$(CC) $(LD_FLAGS) -o $(EXECUTABLE) $(C_OBJS) $(CPP_OBJS) $(CU_OBJS) $(LIBS)

# To compile C files
%.o: %.c
	$(CC) $(CXXFLAGS) $< -c -o $@

# To compile C++ files
%.o: %.cpp
	$(CC) $(CXXFLAGS) $< -c -o $@

# To compile cu files
%.o: %.cu
	$(CUC) $< -c -o $@

# Clean up
clean:
	rm -f $(C_OBJS) $(CPP_OBJS) $(EXECUTABLE)