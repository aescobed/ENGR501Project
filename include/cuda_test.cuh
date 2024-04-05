
#include "c_communicator.h"

extern "C" {
int* callKernel(int w_size, int comm, int arraySize, int* hA, int* hB, int* hC, tensorSender* MLParameters); 
}