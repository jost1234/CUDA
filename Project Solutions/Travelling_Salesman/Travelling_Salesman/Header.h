﻿#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cooperative_groups.h>
#include <iostream>

// Thread block size
#define BLOCK_SIZE 1024


///
/// Control panel
///

// Number of threads = number of ants
const int ants = 1024;

// Repetition constants
#define REPETITIONS 20
#define RANDOM_GENERATIONS 20
#define FOLLOWER_GENERATIONS 500

// Pheromon matrix constants
#define ALPHA 0.75  // Reduction ratio of previous pheromon value
#define REWARD_MULTIPLIER 100   // Rewart multiplier after finding a shortest path until then
#define INITIAL_PHEROMONE_VALUE 1000    // Initial value of elements in the Pheromone matrix


#define SERIALMAXTRIES 1    // Number of serial processes (for debug purposes)


// Diagnostic printing of a matrix
__host__ __device__ void print(double* A, size_t size) {
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

// Main CUDA function
cudaError_t AntCUDA(double* h_Dist, int* h_Route, double* h_Pheromone, bool* h_FoundRoute, unsigned int antNum, size_t size);

// Frees device memory
void Free_device_memory(double* d_Dist, double* d_Pheromone, int* d_Route, bool* d_FoundRoute, int* antRoute, bool* d_invalidInput, bool* d_isolatedVertex, double* d_averageDist);

// Inicializes a random seed for different threads
__global__ void setup_kernel(curandState* state, unsigned long seed);

// 1 block sized kernel
__global__ void AntKernel_1Block(
    double* Dist,       // Cost function input
    double* Pheromone,
    int* Route,         // Sequence output
    bool* FoundRoute,   // Existence output
    size_t size,        // Number of graph vertices
    int* antRoute,      // Temp array
    int antNum,         // Number of ants
    curandState* state  // CURAND random state
);

// Multiblock sized kernel
__global__ void AntKernel_multiBlock(
    double* Dist,       // Cost function input
    double* Pheromone,
    int* Route,         // Sequence output
    bool* FoundRoute,   // Existence output
    size_t size,        // Number of graph vertices
    int* antRoute,      // Temp array
    int antNum,         // Number of ants
    curandState* state, // CURAND random state

    // Pseudo global variables
    bool* invalidInput,     // Variable used to detecting invalid input
    bool* isolatedVertex,   // Variable used to detecting isolated vertex (for optimization purposes)
    double* averageDist     
);

// Generates a random sequence of numbers between 0 and (size - 1) starting with 0
// secondVertex: Variable used for giving an arbitrary second vertex
//      0 < secondvertex < size : valid input (condition = 1)
//      else: invalid input, no mandatory second vertex (condition = 0)
__device__ void generateRandomSolution(int* antRoute, unsigned int antIndex, int secondVertex, double* Dist, size_t size, curandState* state);

// Auxiliary function for generating random sequence
__device__ bool alreadyListed(int* antRoute, int antIndex, size_t size, int idx, int newParam);

// Returns the length of the given route
// Return -1 if route has dead end
__device__ double antRouteLength(double* Dist, int* antRoute, int antIndex, size_t size);

// Evaluates the given solution: modifies Pheromone matrix more if shorter path found
__device__ void evaluateSolution(double* Dist, double* Pheromone, int* antRoute, int antIndex, size_t size, double multiplConstant, int repNumber = 1);

// Represent az ant who follows other ant' pheromones
// Generates a route with Roulette wheel method given the values of the Pheromone matrix
__device__ void followPheromones(const double* Pheromone, int* antRoute, int antIndex, size_t size, curandState* state);

// Diagnostic function for printing given sequence
__device__ __host__ double sequencePrint(int* Route, double* Dist, size_t size);


// Generates a sequnce using greedy algorithm
// Always chooses the highest possible value for the next vertex
__device__ void greedySequence(const double* Pheromone, int* antRoute, size_t size);