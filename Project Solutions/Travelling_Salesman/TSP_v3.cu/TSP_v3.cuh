#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cooperative_groups.h>
#include <iostream>

#include "Defines.cuh"

// TSP : Travelling Salesman Problem

/// DIAGNOSTIC FUNCTIONS

// Diagnostic printing of a matrix
__host__ __device__ void print(float* A, size_t size) {
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



namespace TSP {
    /// CUDA LAUNCH AND KERNEL FUNCTIONS

    // Main CUDA function
    cudaError_t TSP_Ant_CUDA(CUDA_Main_ParamTypedef h_params);

    inline bool inputCheck(CUDA_Main_ParamTypedef* ph_params);

    // Inicializes a random seed for each different threads
    __global__ void setup_kernel(curandState* state, unsigned long seed);

    // Diagnostic function for printing given sequence
    __device__ __host__ float sequencePrint(int* Route, float* Dist, int size);

    // 1 block sized kernel
    __global__ void Kernel(
        Kernel_ParamTypedef params,
        Kernel_ConfigParamTypedef configParams
    );

    // Multiblock sized kernel
    __global__ void Kernel_multiBlock(
        Kernel_ParamTypedef params,
        Kernel_ConfigParamTypedef configParams,
        Kernel_GlobalParamTypedef globalParams);

    // Frees device memory
    void Free_device_memory(
        Kernel_ParamTypedef params);

    // Gets initial value of Route arrays
    __device__ void initAntRoute(
        Kernel_ParamTypedef* pkernelParams,
        int antIndex
    );

    // Generates a random sequence of numbers between 0 and (size - 1) starting with 0
    // secondVertex: Variable used for giving an arbitrary second vertex
    //      0 < secondvertex < size : valid input (condition = 1)
    //      else: invalid input, no mandatory second vertex (condition = 0)
    __device__ void generateRandomSolution(
        Kernel_ParamTypedef* pkernelParams,
        int antIndex,
        int secondVertex
    );

    // Returns bool value of whether newParam is already listed in the route
    // Special care for node 0, which can be in the route [maxVehicles] times.
    __device__ bool alreadyListed(
        Kernel_ParamTypedef* pkernelParams,
        int antIndex,
        int idx,    // serial number of node in route
        int newParam
    );

    // Returns the length of the given route
    // Returns -1 if route not possible (for example has dead end)
    __device__ float antRouteLength(TSP_AntKernel_ParamTypedef* pkernelParams, int antIndex);

    // Represents az ant who follows other ants' pheromones
    // Generates a route with Roulette wheel method given the values of the Pheromone matrix
    __device__ void followPheromones(
        Kernel_ParamTypedef* pkernelParams,
        int antIndex,
        int maxTryNumber
    );

    __device__ void evaluateSolution(
        Kernel_ParamTypedef* pkernelParams,
        int antIndex,
        float multiplConstant,
        float* minRes,
        float rewardMultiplier,
        int repNumber
    );

    // Auxilary function for greedy sequence
    // Return the highest vertex index not yet chosen
    /// row : row of previous route element (decides, which row to watch in the function)
    __device__ int maxInIdxRow(Kernel_ParamTypedef* pkernelParams, int row, int idx);

    // Generates a sequnce using greedy algorithm
    // Always chooses the highest possible value for the next vertex
    __device__ void greedySequence(Kernel_ParamTypedef* pkernelParams);

    // Copies a route into the answer vector
    __device__ void copyAntRoute(Kernel_ParamTypedef* pkernelParams, int antIndex);

    // Validates the output vector
    __device__ bool validRoute(Kernel_ParamTypedef* pkernelParams);
}