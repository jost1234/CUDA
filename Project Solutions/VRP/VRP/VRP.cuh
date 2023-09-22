#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cooperative_groups.h>
#include <iostream>

#include "Defines.h"

// VRP : Vehicle Routing Problem
// Adott egy 0. állomáson (warehouse) k db teherautó, és N-1 db célpont. 
// Cél: Minden pontra eljutni, miközben a maximális út minél rövidebb legyen



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

// Diagnostic function for printing given sequence
__device__ __host__ double sequencePrint(int* Route, DATATYPE* Dist, int size, int routeSize);

// ceil function used to calculate block count
__device__ __host__ int my_ceil(int osztando, int oszto) {
    if (!(osztando % oszto)) return osztando / oszto;
    else return	osztando / oszto + 1;
}

/// CUDA LAUNCH AND KERNEL FUNCTIONS

// Main CUDA function
cudaError_t VRP_Ant_CUDA(VRP_AntCUDA_ParamTypedef params);

// Inicializes a random seed for different threads
__global__ void setup_kernel(curandState* state, unsigned long seed);

// 1 block sized kernel
__global__ void VRP_AntKernel_1Block(
    VRP_AntKernel_ParamTypedef params, 
    VRP_AntKernel_Config_ParamTypedef configParams);

// Multiblock sized kernel
__global__ void VRP_AntKernel_multiBlock(
    VRP_AntKernel_ParamTypedef params, 
    VRP_AntKernel_Config_ParamTypedef configParams, 
    VRP_AntKernel_Global_ParamTypedef globalParams);

// Frees device memory
void Free_device_memory(
    VRP_AntKernel_ParamTypedef params, 
    VRP_AntKernel_Global_ParamTypedef globalParams);

// Gets initial value of Route arrays
__device__ void initAntRoute(
    VRP_AntKernel_ParamTypedef kernelParams,
    unsigned int antIndex
);

// Generates a random sequence of numbers between 0 and (size - 1) starting with 0
__device__ void generateRandomSolution(
    VRP_AntKernel_ParamTypedef kernelParams,
    unsigned int antIndex
);

// Returns bool value of whether newParam is already listed in the route
// Special care for node 0, which can be in the route [maxVehicles] times.
__device__ bool alreadyListed(
    int* antRoute,  // Too few argument to worth giving out the whole kernel struct
    int size,
    int maxVehicles,
    int antIndex,
    int idx,
    int newParam
);

// Returns the length of the given route
// Returns -1 if route not possible (for example has dead end)
__device__ DATATYPE antRouteLength(VRP_AntKernel_ParamTypedef kernelParams, int antIndex);

// Represents az ant who follows other ants' pheromones
// Generates a route with Roulette wheel method given the values of the Pheromone matrix
__device__ void followPheromones(
    VRP_AntKernel_ParamTypedef kernelParams,
    unsigned int antIndex,
    int maxTryNumber
);

__device__ void evaluateSolution(
    VRP_AntKernel_ParamTypedef kernelParams,
    int antIndex,
    DATATYPE multiplConstant,
    DATATYPE* minRes,
    DATATYPE rewardMultiplier,
    int repNumber
);

__host__ __device__ inline int RouteSize(int size, int maxVehicles)
{
    return size + maxVehicles - 1;
};

// Auxilary function for greedy sequence
// Return the highest vertex index not yet chosen
/// row : row of previous route element (decides, which row to watch in the function)
__device__ int maxInIdxRow(VRP_AntKernel_ParamTypedef kernelParams, int row, int idx);

// Generates a sequnce using greedy algorithm
// Always chooses the highest possible value for the next vertex
__device__ void greedySequence(VRP_AntKernel_ParamTypedef kernelParams);