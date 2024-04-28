#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include <mpi.h>

#include "include/cuda_test.cuh"


// Kernel function to add two arrays element-wise
__global__ void addArrays(int* a, int* b, int* c, int size, int repeat) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (int j = 0; j < 10; j++)
    {
    if (idx < size) 
        for(int i=0; i < repeat; i++)
            if (idx + i*size < size*repeat)
                //TODO: fix possible out of bounds
                c[idx + i*size] = a[idx + i*size] + b[idx + i*size];
    }    
    
    
}


extern "C"
{

int* callKernel(int w_size, int comm, int arraySize, int* hA, int* hB, int* hC, tensorSender* MLParameters) 
{

    int repeat = (int)MLParameters->tensor[1];

    int over = comm * (int)ceil(arraySize/(double) w_size) + (int)ceil(arraySize/(double) w_size) - arraySize;

    // Allocate memory on the GPU for arrays
    int* deviceArrayA;
    int* deviceArrayB;
    int* deviceArrayC;


    if(over > 0)
    {

        cudaMalloc((void**)&deviceArrayA, (ceil(arraySize/ (double)w_size) - over) * sizeof(int)); // res = (a + (b - 1)) / b; - For getting ceil
        cudaMalloc((void**)&deviceArrayB, (ceil(arraySize/ (double)w_size) - over) * sizeof(int));
        cudaMalloc((void**)&deviceArrayC, (ceil(arraySize/ (double)w_size) - over) * sizeof(int));

        // Copy data from CPU to GPU
        cudaMemcpy(deviceArrayA, hA + comm * (int)ceil(arraySize/(double) w_size), (ceil(arraySize/ (double)w_size) - over) * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(deviceArrayB, hB + comm * (int)ceil(arraySize/(double) w_size), (ceil(arraySize/ (double)w_size) - over) * sizeof(int), cudaMemcpyHostToDevice);

        // Define grid and block sizes for CUDA
        int threadsPerBlock = (int)MLParameters->tensor[0];
        int blocksPerGrid = ((ceil((arraySize/(double) w_size- over)/(double)repeat) - 1) / threadsPerBlock) + 1;  //(N-1)/nthreads + 1




        double gpu_timer;
        if(comm==0)
        {  
            gpu_timer = MPI_Wtime();
        }

        for (int i = 0; i < 1000; i++)
        {
        // Launch the CUDA kernel
        addArrays<<<blocksPerGrid, threadsPerBlock>>>(deviceArrayA, deviceArrayB, deviceArrayC, ceil((arraySize/ (double) w_size - over)/ (double) repeat), repeat);
        }

        if(comm==0)
        {  
            gpu_timer = MPI_Wtime() - gpu_timer;
            SetTensorOutputValue(MLParameters, 0, gpu_timer);
        }


        // Copy result from GPU to CPU
        cudaMemcpy(hC + comm * (int)ceil(arraySize/ (double)w_size), deviceArrayC, (ceil(arraySize/ (double)w_size) - over) * sizeof(int), cudaMemcpyDeviceToHost);
        

        // Print the result
        /*
        for (int i = comm * (int)ceil(arraySize/(double) w_size); i < (comm+1) * (int)ceil(arraySize/(double) w_size) - over; i++) {
            std::cout << "comm " << comm << ": " << hA[i] << " + " << hB[i] << " = " << hC[i] << std::endl;
        }
        */

    }
    else
    {
        cudaMalloc((void**)&deviceArrayA, ceil(arraySize/ (double)w_size) * sizeof(int)); // res = (a + (b - 1)) / b; - For getting ceil
        cudaMalloc((void**)&deviceArrayB, ceil(arraySize/ (double)w_size) * sizeof(int));
        cudaMalloc((void**)&deviceArrayC, ceil(arraySize/ (double)w_size) * sizeof(int));

        // Copy data from CPU to GPU
        cudaMemcpy(deviceArrayA, hA + comm * (int)ceil(arraySize/(double) w_size), ceil(arraySize/(double) w_size) * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(deviceArrayB, hB + comm * (int)ceil(arraySize/(double) w_size), ceil(arraySize/(double) w_size) * sizeof(int), cudaMemcpyHostToDevice);

        // Define grid and block sizes for CUDA
        int threadsPerBlock = (int)MLParameters->tensor[0];
        int blocksPerGrid = (ceil(arraySize/(double)w_size/(double)repeat) - 1) / threadsPerBlock + 1;  //(N-1)/nthreads + 1


        double gpu_timer;
        if(comm==0)
        {  
            gpu_timer = MPI_Wtime();
        }

        for (int i = 0; i < 1000; i++)
        {
        // Launch the CUDA kernel
        addArrays<<<blocksPerGrid, threadsPerBlock>>>(deviceArrayA, deviceArrayB, deviceArrayC, ceil(arraySize/ (double) w_size/ (double) repeat), repeat);
        }

        if(comm==0)
        {  
            gpu_timer = MPI_Wtime() - gpu_timer;
            SetTensorOutputValue(MLParameters, 0, gpu_timer);
        }

        // Copy result from GPU to CPU
        cudaMemcpy(hC + comm * (int)ceil(arraySize/ (double)w_size), deviceArrayC, ceil(arraySize/ (double) w_size) * sizeof(int), cudaMemcpyDeviceToHost);


        // Print the result
        /*
        for (int i = comm * (int)ceil(arraySize/(double) w_size); i < (comm+1) * (int)ceil(arraySize/(double) w_size); i++) {
                std::cout << "comm " << comm << ": " << hA[i] << " + " << hB[i] << " = " << hC[i] << std::endl;
        }
        */

    }
    

    // Free GPU memory
    cudaFree(deviceArrayA);
    cudaFree(deviceArrayB);
    cudaFree(deviceArrayC);

    return hC;

}

}
