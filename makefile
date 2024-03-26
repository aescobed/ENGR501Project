# module load openmpi/5.0.2 smartRedis/1.3.10
# redis-server ./c6378_redis.conf
# redis-server ./c6379_redis.conf
# redis-server ./py_redis.conf
# redis-cli shutdown nosave
# sudo lsof -i :6379
# redis-cli ping
# /etc/init.d/redis-server stop

# redis-cli --cluster create 127.0.0.1:6378 127.0.0.1:6379 127.0.0.1:6380 			#--cluster-replicas 1
# redis-cli -p 6379 -c
# cluster slots

# For ever primary create 1 replica

CC = 			mpicxx


INC_FLAGS = 	-I/home/andres/code/smartsim/Project/libraries/SmartRedis/install/include 

LD_FLAGS =  	-L/home/andres/code/smartsim/Project/libraries/SmartRedis/install/lib

CXXFLAGS =		-std=c++17 $(INC_FLAGS)

LIBS = 			-lsmartredis

C_FILES =		c_program.c

CPP_FILES =		c_client.cpp \
				c_configoptions.cpp \
				c_dataset.cpp \
				c_error.cpp \
				c_logcontext.cpp \
				c_logger.cpp

C_OBJS = 		c_program.o 

CPP_OBJS = 		c_client.o \
				c_configoptions.o \
				c_dataset.o \
				c_error.o \
				c_logcontext.o \
				c_logger.o

EXECUTABLE =	program

# Default target
all: $(C_OBJS) $(CPP_OBJS)
	$(CC) $(LD_FLAGS) -o $(EXECUTABLE) $(C_OBJS) $(CPP_OBJS) $(LIBS)

# To compile C files
%.o: %.c
	$(CC) $(CXXFLAGS) $< -c -o $@

# To compile C++ files
%.o: %.cpp
	$(CC) $(CXXFLAGS) $< -c -o $@

# Clean up
clean:
	rm -f $(C_OBJS) $(CPP_OBJS) $(EXECUTABLE)