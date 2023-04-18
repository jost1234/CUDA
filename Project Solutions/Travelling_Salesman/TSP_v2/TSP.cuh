#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cooperative_groups.h>
#include <iostream>

#include "Defines.h"

// TSP : Travelling Salesman Problem

/// DIAGNOSTIC FUNCTIONS

// Diagnostic printing of a matrix
__host__ __device__ void print(DATATYPE* A, size_t size) {
    for (int ii = 0; ii < size; ii++) {
        for (int jj = 0; jj < size; jj++)
            printf("%.2f ", A[ii * size + jj]);
        printf("\n");
    }
    printf("\n");
}

// ceil function used to calculate block count
__device__ __host__ int my_ceil(int osztando, int oszto) {
    if (!(osztando % oszto)) return osztando / oszto;
    else return	osztando / oszto + 1;
}

// Inicializes a random seed for different threads
__global__ void setup_kernel(curandState* state, unsigned long seed);

// Diagnostic function for printing given sequence
__device__ __host__ double sequencePrint(int* Route, double* Dist, size_t size);

namespace TSP {
    /// CUDA LAUNCH AND KERNEL FUNCTIONS

    // Main CUDA function
    cudaError_t TSP_Ant_CUDA(TSP_AntCUDA_ParamTypedef h_params);

    inline bool inputCheck(TSP_AntCUDA_ParamTypedef h_params);

    // 1 block sized kernel
    __global__ void TSP_AntKernel_1Block(
        TSP_AntKernel_ParamTypedef params,
        TSP_AntKernel_Config_ParamTypedef configParams);

    // Multiblock sized kernel
    __global__ void VRP_AntKernel_multiBlock(
        TSP_AntKernel_ParamTypedef params,
        TSP_AntKernel_Config_ParamTypedef configParams,
        TSP_AntKernel_Global_ParamTypedef globalParams);

    // Frees device memory
    void Free_device_memory(
        TSP_AntKernel_ParamTypedef params, 
        TSP_AntKernel_Global_ParamTypedef globalParams);

}