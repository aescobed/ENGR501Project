#include <iostream>
#include <mpi.h>
#include <math.h>

#include "include/cuda_test.cuh"
#include "include/c_communicator.h"

int main (int argc, char** argv)
{
    MPI_Init(NULL, NULL);

    // Get number of MPI processes
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    // Get process rank
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    void* client;
    struct tensorSender ts;

    if(world_rank!=0)
    {
        client = StartClient();
        ts = CreateTensorSender(3, client);
        DeleteTensorSender(ts);
        EndClient(client);
    }



    const int arraySize = 10;


    // Allocate memory on the CPU for arrays
    int* hostArrayA = (int*)malloc(arraySize * sizeof(int));
    int* hostArrayB = (int*)malloc(arraySize * sizeof(int));
    int* hostArrayC = (int*)malloc(arraySize * sizeof(int));

    // Initialize arrays on the CPU
    for (int i = 0; i < arraySize; ++i) {
        hostArrayA[i] = i;
        hostArrayB[i] = i * 10;
    }

    // Amount of times each GPU thread does the same task
    const int repeat = 2;

    hostArrayC = callKernel(world_size, world_rank, arraySize, hostArrayA, hostArrayB, hostArrayC, repeat);
    

    MPI_Barrier(MPI_COMM_WORLD);

    // Everyone send to rank 0
    if(world_rank!=0)
        MPI_Send(hostArrayC + (int)(world_rank * ceil(arraySize/(double) world_size)), ceil(arraySize/(double) world_size), MPI_INT, 0, 0, MPI_COMM_WORLD);

    if(world_rank==0) {
        
        for(int i=1; i < world_size; i++)
            //TODO fix possible out of bounds
            MPI_Recv(hostArrayC + (int)(i* ceil(arraySize/(double) world_size)), ceil(arraySize/(double) world_size), MPI_INT, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            
        for (int i = 0; i < arraySize; i++) {
            std::cout << hostArrayA[i] << " + " << hostArrayB[i] << " = " << hostArrayC[i] << std::endl;
        }    
        
        std::cout << "Program finished with " << world_size << " processes\n";

    }
    // Free CPU memory
    free(hostArrayA);
    free(hostArrayB);
    free(hostArrayC);



    MPI_Finalize();

    return 0;
}