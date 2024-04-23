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
    struct tensorSender params;

    if(world_rank==0)
    {

 

        client = StartClient();
        params = CreateTensorSender(3, client);

        SetTensorParameterValue(&params, 0, 1);
        SetTensorParameterValue(&params, 1, 1);
        SetTensorParameterValue(&params, 2, 1);

        SetTensorOutputValue(&params, 0, 1);
        SetTensorOutputValue(&params, 1, 1);
        SetTensorOutputValue(&params, 2, 1);


        GetTensor(&params);

        printf("value === %f", params.tensor[0]);

    }




    const int arraySize = 1000;


    // Allocate memory on the CPU for arrays
    int* hostArrayA = (int*)malloc(arraySize * sizeof(int));
    int* hostArrayB = (int*)malloc(arraySize * sizeof(int));
    int* hostArrayC = (int*)malloc(arraySize * sizeof(int));

    // Initialize arrays on the CPU
    for (int i = 0; i < arraySize; ++i) {
        hostArrayA[i] = i;
        hostArrayB[i] = i * 10;
    }

    double all_gpu_timer;
    if(world_rank==0)
    {  
        all_gpu_timer = MPI_Wtime();
    }

    hostArrayC = callKernel(world_size, world_rank, arraySize, hostArrayA, hostArrayB, hostArrayC, &params);
    
    if(world_rank==0)
    {  
        all_gpu_timer = MPI_Wtime() - all_gpu_timer;
    }

    MPI_Barrier(MPI_COMM_WORLD);

    // Everyone send to rank 0
    if(world_rank!=0)
        MPI_Send(hostArrayC + (int)(world_rank * ceil(arraySize/(double) world_size)), ceil(arraySize/(double) world_size), MPI_INT, 0, 0, MPI_COMM_WORLD);

    if(world_rank==0) {
        
        for(int i=1; i < world_size; i++)
            //TODO fix possible out of bounds
            MPI_Recv(hostArrayC + (int)(i* ceil(arraySize/(double) world_size)), ceil(arraySize/(double) world_size), MPI_INT, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            
        for (int i = 0; i < arraySize; i++) {
            //std::cout << hostArrayA[i] << " + " << hostArrayB[i] << " = " << hostArrayC[i] << std::endl;
        }    
        
        std::cout << "Program finished with " << world_size << " processes\n";

    }
    // Free CPU memory
    free(hostArrayA);
    free(hostArrayB);
    free(hostArrayC);


    if(world_rank==0)
    {
        SetTensorParameterValue(&params, 0, params.tensor[0]);
        SetTensorParameterValue(&params, 1, params.tensor[1]);
        SetTensorParameterValue(&params, 2, params.tensor[2]);

        SetTensorOutputValue(&params, 0, all_gpu_timer);
        SetTensorOutputValue(&params, 1, 1);
        SetTensorOutputValue(&params, 2, 1);


        SendTensor(&params);

       
        //DeleteTensorSender(&params);

        
        EndClient(client);
    }

    MPI_Finalize();

    return 0;
}