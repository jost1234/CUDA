﻿// Special CUDA API headers
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "device_launch_parameters.h"
#include <cooperative_groups.h>
#include "curand.h"
#include "curand_kernel.h"

// Custom header containing Control Panel
#include "VRP_v2.cuh"

// General purpose headers
#include <iostream>
#include <stdbool.h>
#include <stdlib.h>
#include <assert.h>
#include <float.h>

// Cooperative groups namespace for block and grid sync
using namespace cooperative_groups;

// Main function
int main(int argc, char* argv[])
{
    // Variables used for reading from txt file
    FILE* pfile;    // File pointer
    int fileNameIdx;
    bool foundDistFile = false;   // Error handling
    int size;    // Number of graph vertices
    int maxVehicles; // Maximum number of vehicles in warehouse
    int i;  // Iterator
    srand(time(0)); // Need seeds for random solutions

    // Processing command line arguments
    for (i = 1; i < argc; ++i)
    {
        /// Distance file: REQUIRED
        // Command Line Syntax: ... --dist [file_name]
        if ((strcmp(argv[i], "-d") == 0) || (strcmp(argv[i], "--dist") == 0))
        {
            pfile = fopen(argv[++i], "r");
            if (pfile == NULL) {
                fprintf(stderr, "Unable to open file \"%s\"", argv[i]);
                return -1;
            }
            fileNameIdx = i;
            printf("Opening file \"%s\"!\n", argv[fileNameIdx]);
            foundDistFile = true;
        }

        /// Number of threads: OPTIONAL (default: 1024)
        // Command Line Syntax: ... --ants [number of ants]
        else if ((strcmp(argv[i], "--a") == 0) || (strcmp(argv[i], "--ants") == 0))
        {
            if (sscanf(argv[++i], "%d", &ants) != 1) {
                fprintf(stderr, "Unable to read ant number!\n");
            }
            else {
                printf("Given ant number : %d\n", ants);
            }
        }
    }

    // Checking required elements
    if (!foundDistFile)
    {
        fprintf(stderr, "Please give a file in command line arguments to set the Distance Matrix!\n");
        fprintf(stderr, "Command Line Syntax:\n\t--dist [data_file].txt\n");
        fprintf(stderr, "File Syntax:\n\t[Number of Nodes]\n\tdist11, dist12, ...\n\tdist21 ... \n");
        return -1;
    }

    // File syntax : 1st row must contain graph size in decimal
    // Following rows: graph edge values separated with comma (,)
    if (fscanf_s(pfile, "%d \n", &size) == 0) {
        fprintf(stderr, "Unable to read Size!\n Make sure you have the right file syntax!\n");
        fprintf(stderr, "File Syntax:\n\t[Number of Nodes]\n\tdist11, dist12, ...\n\tdist21 ... \n");
        fclose(pfile);
        return -1;
    }

    // Distance matrix
    // Store type: adjacency matrix format
    float* Dist = (float*)calloc(size * size, sizeof(float));

    // Reading distance values from dist file
    for (int ii = 0; ii < size; ++ii) {
        float temp;

        for (int jj = 0; jj < size; ++jj) {
            if (fscanf_s(pfile, "%f", &temp) == 0) {
                fprintf(stderr, "Error reading file \"%s\" distance(%d,%d)\n", argv[fileNameIdx], ii, jj);
                fclose(pfile);
                return -1;
            }
            Dist[ii * size + jj] = temp;
        }
        fscanf_s(pfile, "\n");
    }

    // File syntax : row after dist values must contain maximum vehicle count in decimal
    if (fscanf_s(pfile, "No of trucks: %d", &maxVehicles) == 0) {
        fprintf(stderr, "Unable to read Maximum Vehicle Number!\n Make sure you have the right file syntax!\n");
        fprintf(stderr, "File Syntax:\n\t[Number of Vehicles Available]\n\t[Number of Nodes]\n\tdist11, dist12, ...\n\tdist21 ... \n");
        fclose(pfile);
        return -1;
    }

    // Closing data file
    printf("Closing file \"%s\"!\n", argv[fileNameIdx]);
    if (fclose(pfile) != 0) {
        fprintf(stderr, "Unable to close file \"%s\"!\n", argv[fileNameIdx]);
        return -1;
    }

    // Printing Matrix
    printf("Maximum number of vehicles: %d\n", maxVehicles);
    printf("Given Dist matrix:\n");
    print(Dist, size);

    // Host Variables

    // Route: [0 ... 1st Route ... 0 ... 2nd Route ... ... Last Route ... ( Last 0 not stored)]

    VRP::CUDA_Main_ParamTypedef params;
    params.antNum = ants;
    params.size = size;
    params.maxVehicles = maxVehicles;
    params.Dist = Dist;
    params.Pheromone = (float*)malloc(size * VRP::RouteSize(size, maxVehicles) * sizeof(float));
    params.route = (int*)malloc(VRP::RouteSize(size, maxVehicles) * sizeof(int));

    printf("Vehicle Route Problem with Ant Colony Algorithm\n");
    VRP::CUDA_main(params);

    free(params.Dist);
    free(params.Pheromone);
    free(params.route);
    return 0;
}

namespace VRP {

    // Global variables for multi grid Kernel
    __device__ Kernel_GlobalParamTypedef globalParams;

    // Host function for CUDA
    cudaError_t CUDA_main(CUDA_Main_ParamTypedef h_params)
    {
        cudaError_t cudaStatus;
        // Local variables
        int maxVehicles = h_params.maxVehicles; // Maximum number of vehicles in warehouse
        int size = h_params.size;    // Number of graph vertices
        int antNum = h_params.antNum;    // Number of Ants (= threads)

        // Invalid inputs
        if (!inputGood(&h_params)) {
            fprintf(stderr, "Invalid Input values!\n");
            return cudaError_t::cudaErrorInvalidConfiguration;
        }

        // Choosing GPU, may be nessesary in a multi-GPU system
        cudaStatus = cudaSetDevice(0);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaSetDevice failed! Do you have a CUDA-capable GPU installed?\n");
            return cudaStatus;
        }

        // Calculates the number of Grid blocks to execute
        // Number of threads = number of ants
        int BlockNum = 1;
        if (antNum > BLOCK_SIZE) {
            BlockNum = my_ceil(antNum, BLOCK_SIZE);
            antNum = BlockNum * BLOCK_SIZE; // For better usage of parallel threads
        }

        // Device pointers
        Kernel_ParamTypedef d_kernelParams;
        d_kernelParams.Dist = NULL;
        d_kernelParams.Pheromone = NULL;
        d_kernelParams.route = NULL;
        d_kernelParams.state = NULL;
        d_kernelParams.antNum = antNum;
        d_kernelParams.size = size;
        d_kernelParams.maxVehicles = maxVehicles;
        d_kernelParams.routeSize = RouteSize(size, maxVehicles);

        // Config parameters
        Kernel_ConfigParamTypedef d_configParams;
        d_configParams.Rho = RHO;
        d_configParams.Follower_Generations = FOLLOWER_GENERATIONS;
        d_configParams.Initial_Pheromone_Value = INITIAL_PHEROMONE_VALUE;
        d_configParams.maxTryNumber = RouteSize(size, maxVehicles);
        d_configParams.Random_Generations = RANDOM_GENERATIONS;
        d_configParams.Repetitions = REPETITIONS;
        d_configParams.Reward_Multiplier = REWARD_MULTIPLIER;

        // Size of device malloc
        size_t Dist_bytes = size * size * sizeof(float);
        size_t Pheromone_bytes = size * RouteSize(size, maxVehicles) * sizeof(float);
        // We need memory for multiple Routes
        size_t route_bytes = RouteSize(size, maxVehicles) * sizeof(int);

        size_t antRoute_bytes = antNum * route_bytes;   // Allocating working memory for all threads
        size_t state_bytes = antNum * sizeof(curandState);

        // Dist
        cudaStatus = cudaMalloc((void**)&d_kernelParams.Dist, Dist_bytes);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "d_Dist cudaMalloc failed!\n");
            Free_device_memory(d_kernelParams);
            return cudaStatus;
        }
        // Pheromone
        cudaStatus = cudaMalloc((void**)&d_kernelParams.Pheromone, Pheromone_bytes);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "d_Pheromone cudaMalloc failed!\n");
            Free_device_memory(d_kernelParams);
            return cudaStatus;
        }
        // route
        cudaStatus = cudaMalloc((void**)&d_kernelParams.route, route_bytes);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "d_Route cudaMalloc failed!\n");
            Free_device_memory(d_kernelParams);
            return cudaStatus;
        }
        // antRoute : auxiliary array
        cudaStatus = cudaMalloc((void**)&d_kernelParams.antRoute, antRoute_bytes);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "antRoute cudaMalloc failed!\n");
            Free_device_memory(d_kernelParams);
            return cudaStatus;
        }
        // state : CUDA supported random seeds for threads
        cudaStatus = cudaMalloc(&d_kernelParams.state, state_bytes);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "state cudaMalloc failed!\n");
            Free_device_memory(d_kernelParams);
            return cudaStatus;
        }

        // Copying Dist data : Host -> Device
        cudaStatus = cudaMemcpy(d_kernelParams.Dist, h_params.Dist, Dist_bytes, cudaMemcpyHostToDevice);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "Dist cudaMemcpy failed!\n");
            Free_device_memory(d_kernelParams);
            return cudaStatus;
        }

        printf("Called function with %d Block", BlockNum);
        if (BlockNum == 1)
            printf(": \n");
        else
            printf("s: \n");
        int threadPerBlock = (antNum > BLOCK_SIZE) ? BLOCK_SIZE : antNum;

        // setup seeds

        setup_kernel <<< BlockNum, threadPerBlock >>> (d_kernelParams.state, time(NULL) * rand());

        // Kernel call

        float min = FLT_MAX;
        float sum = 0.0f;
        int foundCount = 0;

        for (int iter = 0; iter < SERIALMAXTRIES; iter++)
        {
            printf("\nAttempt #%d ||\n", iter);

            if (BlockNum == 1) {
                Kernel_1Block <<< 1, threadPerBlock >>> (d_kernelParams, d_configParams);
            }
            else
            {
                // During Kernel call it's important to use cudaLaunchCooperativeKernel CUDA runtime launch API
                // or its CUDA driver equivalent instead of the <<<...>>> syntax

                // Sets supportsCoopLaunch=1 if the operation is supported on device 0
                // Only compute capability 6.0 or higher!
                int dev = 0;
                int supportsCoopLaunch = 0;
                cudaDeviceGetAttribute(&supportsCoopLaunch, cudaDevAttrCooperativeLaunch, dev);
                if (supportsCoopLaunch != 1)
                {
                    fprintf(stderr, "Cooperative Launch is not supported on this machine configuration.");
                    Free_device_memory(d_kernelParams);
                    return cudaStatus;
                }

                // Call arguments
                void* kernelArgs[] = { &d_kernelParams, &d_configParams };

                cudaLaunchCooperativeKernel((void*)Kernel_multiBlock, BlockNum, BLOCK_SIZE, kernelArgs);
            }

            // Error handling during Kernel execution
            cudaStatus = cudaGetLastError();
            if (cudaStatus != cudaSuccess) {
                fprintf(stderr, "AntKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
                Free_device_memory(d_kernelParams);
                return cudaStatus;
            }

            // cudaDeviceSynchronize waits for the kernel to finish
            cudaStatus = cudaDeviceSynchronize();
            if (cudaStatus != cudaSuccess) {
                fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching antKernel!\n%s\n", cudaStatus, cudaGetErrorString(cudaStatus));
                Free_device_memory(d_kernelParams);
                return cudaStatus;
            }

            // Copying processed route data : Device -> Host
            cudaStatus = cudaMemcpy(h_params.route, d_kernelParams.route, route_bytes, cudaMemcpyDeviceToHost);
            if (cudaStatus != cudaSuccess) {
                fprintf(stderr, "route dev->host cudaMemcpy failed!");
                // Frees GPU device memory
                Free_device_memory(d_kernelParams);
                return cudaStatus;
            }

            float _length = sequencePrint(h_params.route, h_params.Dist, size, RouteSize(size, maxVehicles));
            if (_length > 0) {
                foundCount++;
                sum += _length;
                if (_length < min)
                    min = _length;
            }


        }
        printf("\nSummary:\nAverage length: %.2f\n", sum / foundCount);
        printf("Minimal length: %.2f\n", min);

        // Frees GPU device memory
        Free_device_memory(d_kernelParams);
        return cudaStatus;
    }

    // Testing input for main CUDA function
    // Returns true if input data syntax is good
    // Disclaimer: Only tests NULL property of pointers, does not 100% guarantee perfect data
    __host__ inline bool inputGood(CUDA_Main_ParamTypedef* params) {
        return (
            32 <= params->antNum &&        // At least 32 threads (for GPU usage)
            2 <= params->size &&          // At least 2 nodes
            1 <= params->maxVehicles &&   // At least 1 vehicle
            NULL != params->Dist &&
            NULL != params->Pheromone &&
            NULL != params->route);
    }

    // Inicializes a random seed for each different threads
    __global__ void setup_kernel(curandState* state, unsigned long seed)
    {
        int id = blockIdx.x * blockDim.x + threadIdx.x;
        curand_init(seed, id, id, &state[id]);
    }

    // Frees device memory with cudaFree if pointer is not NULL
    // Important: all initial values of device pointers MUST BE NULL
    void Free_device_memory(Kernel_ParamTypedef params) {
        // Temporary device data structures
        if (NULL != params.Dist) cudaFree(params.Dist);
        if (NULL != params.antRoute) cudaFree(params.antRoute);
        if (NULL != params.Pheromone) cudaFree(params.Pheromone);
        if (NULL != params.route) cudaFree(params.route);
        if (NULL != params.state) cudaFree(params.state);
    }

    // Testing input for main CUDA function
    // Returns true if input data syntax is good
    // Disclaimer: Only tests NULL property of pointers, does not 100% guarantee perfect data
    __device__ inline bool inputGood(Kernel_ParamTypedef* params) {
        return (
            32 <= params->antNum &&    // At least 32 threads (for GPU usage)
            2 <= params->size &&      // At least 2 nodes
            1 <= params->maxVehicles &&   // At least 1 vehicle
            NULL != params->Dist &&
            NULL != params->Pheromone &&
            NULL != params->route);
    }

    // Diagnostic function for printing given sequence
    __device__ __host__ float sequencePrint(int* route, float* Dist, int size, int routeSize) {
        if (
            2 > size ||
            2 > routeSize ||
            NULL == route ||
            NULL == Dist) {
            printf("Invalid input!\n");
            return -1;
        }

        float l = 0;
        int vehicleCntr = 0, i = 0;

        // Check for dead end
        while (i < routeSize)
        {
            int src = route[i];
            int dst = route[(i + 1) % routeSize];
            assert(src > -1 && src < size&& dst > -1 && dst < size);
            if (Dist[src * size + dst] < 0)
            {
                printf("Route not found!\n");
                return -1;
            }
            i++;
        }

        i = 0;
        printf("Vehicle #0 : ");
        while (i < routeSize) {
            int src = route[i];
            int dst = route[(i + 1) % routeSize];

            // End of route for a vehicle
            if (dst == 0) {
                if (src == 0)
                    printf("Unused\nVehicle #%d : ", ++vehicleCntr);
                else if (routeSize - 1 != i) {
                    printf("%d (%.0f) 0\nVehicle #%d : ", src, Dist[src * size + dst], ++vehicleCntr);
                }
                else {
                    printf("%d (%.0f) 0\n", src, Dist[src * size + dst]);
                }
            }
            else {
                // Next element of Route 
                printf("%d (%.0f) ", src, Dist[src * size + dst]);
            }
            l += Dist[src * size + dst];
            i++;
        }
        printf(" Total length : %.2f\n ", l);
        return l;
    }

    // 1 block sized kernel
    __global__ void Kernel_1Block(
        Kernel_ParamTypedef params,
        Kernel_ConfigParamTypedef configParams
    )
    {
        // Dist (i,j) means the distance from node i to node j
        // If no edge drawn between them: Dist(i,j) = -1 (expected syntax)
        thread_block block = this_thread_block();

        int antIndex = blockIdx.x * blockDim.x + threadIdx.x;  // Ant index 0 - (antNum-1)
        int tr = block.thread_rank();   // elvileg ugyanaz mint az előző sor
        // Ott használom, ahol ez az átláthatóságot segíti

        if (antIndex >= params.antNum || blockIdx.x > 0)     // Defense against overaddressing
            return;

        // Shared variables between threads in the same block
        __shared__ bool invalidInput;       // Variable used to detecting invalid input
        __shared__ bool isolatedVertex;     // Variable used to detecting isolated vertex (for optimization purposes)
        __shared__ float averageDist;    // Average edge distance
        __shared__ float multiplicationConst;
        __shared__ int size;                // Local Copy of argument parameter
        __shared__ int maxVehicles;

        // Initialization of temporary variables
        invalidInput = false;
        isolatedVertex = false;
        averageDist = 0.0f;
        multiplicationConst = 0.0f;
        size = params.size; // Need to be written too many times
        maxVehicles = params.maxVehicles;
        params.routeSize = RouteSize(size, maxVehicles);
        globalParams.minRes = FLT_MAX;

        // Invalidate route vector
        for (int i = 0; i < size; i++)
            params.route[i] = 0;

        // Input check
        if (antIndex == 0 && !inputGood(&params)) {
            invalidInput = true;
            printf("Invalid Input\n");
        }
        block.sync();



        // Pheromone matrix initialization
        if (antIndex == 0)
        {
            bool foundNeighboor = false;    // Checking if any of the nodes are isolated
            int i, j;
            for (i = 0; i < size; i++) {
                for (j = 0; j < size; j++) {
                    // Initializing Pheromone graph (anti - unitmatrix, all main diagonal elements are 0)
                    // 0 Pheromone value if no edge drawn
                    // Initial Pheromone value is of consideration in the Control panel
                    if (i < size && ((i == j) || (params.Dist[i * size + j] < 0)))
                        params.Pheromone[i * size + j] = 0.0f;
                    else
                        params.Pheromone[i * size + j] = configParams.Initial_Pheromone_Value;

                    // Error handling 
                    // Check if there are invalid given elements 
                    // Valid input if: positive OR -1 OR 0 (only if i=j)
                    if (i != j && params.Dist[i * size + j] <= 0
                        && params.Dist[i * size + j] != -1)
                    {
                        printf("Dist(%d,%d) incorrect!\n", i, j);
                        invalidInput = true;
                        break;
                    }
                    if (!foundNeighboor && params.Dist[i * size + j] > 0) {
                        // Has neighboor therefore not isolated
                        foundNeighboor = true;
                    }
                }
                if (!foundNeighboor) { // Did not have any neighboors => wrong model of TSP
                    printf("Vertex %d isolated!\n", i);
                    isolatedVertex = true;
                }
            }
            /// The warehouse is simulated
            /// as k nodes in the same spot ==> TSP Reduction
            for (; i < params.routeSize; i++)
            {
                for (j = 0; j < size; j++)
                {
                    params.Pheromone[i * size + j] = configParams.Initial_Pheromone_Value;
                }
            }
        }

        block.sync();


        if (invalidInput || isolatedVertex)   // Invalid input, so no point of continuing
            return;                           // Case of isolated node means no route exists

        // Case of only 2 nodes: handle quickly in 1 thread
        if (size == 2) {
            if (tr == 0) {
                if (params.Dist[0 * size + 1] > 0 && params.Dist[1 * size + 0] > 0) {    // Route exists
                    params.route[0] = 0;    // Route = [0 1 0...0]
                    params.route[1] = 1;
                    for (int ii = 0; ii < params.maxVehicles - 1; ++ii)
                        params.route[size + ii] = 0; // Only one vehicle needed
                }
            }
            block.sync();
            return;
        }



        // Calculating average distance
        if (antIndex == 0) {
            float sum = 0.0f;   // Sum of edge values
            int numPos = 0;     // Number of edges
            for (int i = 0; i < size; i++) {
                for (int j = 0; j < size; j++)
                {
                    float edge = params.Dist[i * size + j];
                    if (edge > 0)
                    {
                        sum += edge;
                        numPos++;
                    }
                }
            }
            averageDist = sum / numPos * size;

        }
        block.sync();


        // Default values for routes
        initAntRoute(&params, antIndex);
        block.sync();

        // Ants travelling to all directions
        for (int repNumber = 0; repNumber < configParams.Repetitions; repNumber++)
        {

            if (antIndex == 0)
                multiplicationConst = averageDist / configParams.Rho * 5;
            block.sync();
            /*block.sync();
            if (antIndex == 0)
            {
                printf("\nPh2:\n");
                print(params.Pheromone, params.size, params.routeSize);
                printf("\nDist2:\n");
                print(params.Dist, size);
            }
            block.sync();*/
            // Numerous random guesses
            for (int j = 0; j < configParams.Random_Generations; j++)
            {

                generateRandomSolution(&params, antIndex);
                evaluateSolution(&params, antIndex, multiplicationConst, configParams.Reward_Multiplier, repNumber);
                block.sync();
            }

            if (antIndex == 0)
                multiplicationConst *= 2;
            block.sync();

            // Lots of ants following pheromone of previous ants
            for (int gen = 0; gen < configParams.Follower_Generations; gen++) {

                // Reducing previous pheromon values by value RHO (modifiable in the Control Panel)
                if (antIndex == 0) {
                    for (int i = 0; i < params.routeSize; i++) {
                        for (int j = 0; j < size; j++)
                            params.Pheromone[i * size + j] *= configParams.Rho;
                    }
                }
                block.sync();

                // New ants following pheromone of previous ants
                followPheromones(&params, antIndex, configParams.maxTryNumber);
                block.sync();
                evaluateSolution(&params, antIndex, multiplicationConst, configParams.Reward_Multiplier, repNumber);
                block.sync();
            }
        }
        // Removing unwanted threads
        if (antIndex != 0)
            return;

        // Choosing path with greedy algorithm if we dont have a valid answer
        if (!validRoute(&params)) {
            // Mostly occurs when it did not find any routes, but we also prepare for corrupted memory
            printf("Need to find route in greedy mode!\n");
            greedySequence(&params);
        }
    }

    // Multiblock sized kernel
    __global__ void Kernel_multiBlock(
        Kernel_ParamTypedef params,
        Kernel_ConfigParamTypedef configParams)
    {
        // Dist (i,j) means the distance from vertex i to vertex j
        // If no edge drawn between them: Dist(i,j) = -1 (expected syntax)
        grid_group grid = this_grid();
        if (!grid.is_valid())
            return;
        grid.sync();
        int antIndex = blockIdx.x * blockDim.x + threadIdx.x;  // ant index
        grid.sync();

        __shared__ int size;                // Local Copy of argument parameter

        // Initialization of temporary variables
        __shared__ float multiplicationConst;
        multiplicationConst = 0.0f;
        globalParams.invalidInput = false;
        globalParams.isolatedVertex = false;
        globalParams.averageDist = 0.0f;
        size = params.size; // Need to be written too many times
        params.routeSize = RouteSize(params.size, params.maxVehicles);
        globalParams.minRes = FLT_MAX;

        // Invalidate route vector
        for (int i = 0; i < params.size; i++)
            params.route[i] = 0;

        // Input check
        if (antIndex == 0 && !inputGood(&params)) {
            globalParams.invalidInput = true;
            printf("Invalid Input\n");
        }
        grid.sync();

        // Pheromone matrix initialization
        /*if (threadIdx.x == 0)
        {*/
        bool foundNeighboor = false;    // Checking if any of the nodes are isolated
        int i, j;
        for (i = 0; i < params.size; i++) {
            for (j = 0; j < params.size; j++) {
                // Initializing Pheromone graph (anti - unitmatrix, all main diagonal elements are 0)
                // 0 Pheromone value if no edge drawn
                // Initial Pheromone value is of consideration in the Control panel
                if ((i == j) || (params.Dist[i * params.size + j] < 0))
                    params.Pheromone[i * params.size + j] = 0.0f;
                else
                    params.Pheromone[i * params.size + j] = configParams.Initial_Pheromone_Value;

                // Error handling 
                // Check if there are invalid given elements 
                // Valid input if: positive OR -1 OR 0 (only if i=j)
                if (i != j && params.Dist[i * params.size + j] <= 0
                    && params.Dist[i * params.size + j] != -1)
                {
                    printf("Dist(%d,%d) incorrect!\n", i, j);
                    globalParams.invalidInput = true;
                    break;
                }
                if (!foundNeighboor && params.Dist[i * params.size + j] > 0) {
                    // Has neighboor therefore not isolated
                    foundNeighboor = true;
                }
            }
            if (!foundNeighboor) { // Did not have any neighboors => wrong model of TSP
                printf("Vertex %d isolated!\n", i);
                globalParams.isolatedVertex = true;
            }
        }
        /// The warehouse is simulated
        /// as k nodes in the same spot ==> TSP Reduction
        for (; i < params.routeSize; i++)
        {
            for (j = 0; j < params.size; j++)
            {
                params.Pheromone[i * params.size + j] = configParams.Initial_Pheromone_Value;
            }
        }
        //}
        grid.sync();

        if (globalParams.invalidInput || globalParams.isolatedVertex) {   // Invalid input, so no point of continuing
            return;                             // Case of isolated node means no route exists
        }

        // Case of only 2 nodes: handle quickly in 1 thread
        if (params.size == 2) {
            if (antIndex == 0) {
                if (params.Dist[0 * params.size + 1] > 0 && params.Dist[1 * params.size + 0] > 0) {    // Route exists
                    params.route[0] = 0;    // Route = [0 1 0...0]
                    params.route[1] = 1;
                    for (int ii = 0; ii < params.maxVehicles - 1; ++ii)
                        params.route[params.size + ii] = 0; // Only one vehicle needed
                }
            }
            grid.sync();
            return;
        }

        // Left: Connected(?) graph with at least 3 nodes
        // Calculating average distance

        if (antIndex == 0)
        {
            float sum = 0.0f;   // sum of edge values
            int numpos = 0;  // number of edges
            float edge;  // temp variable
            for (int i = 0; i < params.size; i++) {
                for (int j = 0; j < params.size; j++)
                {
                    edge = params.Dist[i * params.size + j];
                    if (edge > 0)
                    {
                        sum += edge;
                        numpos++;
                    }
                }
            }
            globalParams.averageDist = sum / numpos * params.routeSize;
        }
        grid.sync();

        // Initializing ant Routes 
        initAntRoute(&params, antIndex);
        grid.sync();

        // Ants travelling to all directions
        for (int repNumber = 0; repNumber < configParams.Repetitions; repNumber++)
        {
            grid.sync();
            multiplicationConst = globalParams.averageDist / configParams.Rho * 5.0f;
            grid.sync();

            // Numerous random guess
            for (int j = 0; j < configParams.Random_Generations; j++)
            {
                generateRandomSolution(&params, antIndex);
                grid.sync();
                // Evaluating the given solution: modifies Pheromone matrix more if shorter path found
                evaluateSolution(&params, antIndex, multiplicationConst, configParams.Reward_Multiplier, repNumber);
                grid.sync();
            }
            

            multiplicationConst *= 2;
            grid.sync();

            // Lots of ants following pheromone of previous ants
            for (int gen = 0; gen < configParams.Follower_Generations; gen++)
            {
                // Reducing previous pheromone values by value RHO (modifiable in the Control Panel)
                if (antIndex == 0) {
                    for (int i = 0; i < params.routeSize; i++) {
                        for (int j = 0; j < params.size; j++)
                            params.Pheromone[i * params.size + j] *= configParams.Rho;
                    }
                }
                grid.sync();

                // New ants following pheromone of previous ants
                followPheromones(&params, antIndex, configParams.maxTryNumber);
                grid.sync();
                evaluateSolution(&params, antIndex, multiplicationConst, configParams.Reward_Multiplier, repNumber);
                grid.sync();
            }
        }

        if (antIndex == 0) {
            // Choosing path with greedy algorithm if we dont have a valid answer
            if (!validRoute(&params)) {
                //printf("Need to find route in greedy mode!\n");
                greedySequence(&params);
            }
        }
    }

    // Gets initial value of Route arrays
    __device__ void initAntRoute(
        Kernel_ParamTypedef* pkernelParams,
        int antIndex
    )
    {
        // Route init [0, 1, 2 ... size-1, 0, 0 ... 0]
        // Optimizing array addressing
        int* antRouteOffset = pkernelParams->antRoute +
            antIndex * pkernelParams->size;


        for (int idx1 = 0; idx1 < pkernelParams->size; ++idx1) {
            antRouteOffset[idx1] = idx1;
        }
        for (int idx2 = 0; idx2 < pkernelParams->maxVehicles - 1; ++idx2) {
            antRouteOffset[pkernelParams->size + idx2] = 0;
        }
    }

    // Generates a random sequence of numbers between 0 and (size - 1) starting with 0
    // secondVertex: Variable used for giving an arbitrary second vertex
    //      0 < secondvertex < size : valid input (condition = 1)
    //      else: invalid input, no mandatory second vertex (condition = 0)
    __device__ void generateRandomSolution(
        Kernel_ParamTypedef* pkernelParams,
        int antIndex
    )
    {
        int* antRouteOffset = pkernelParams->antRoute
            + antIndex * pkernelParams->routeSize;   // Optimizing array addressing
        // Expected to start in node 0 (in normal use this is already set)
        antRouteOffset[0] = 0;
        // Route init [0, 1, 2 ... size-1, 0, 0 ... 0]
        int min_rand_int = 1, max_rand_int = pkernelParams->routeSize - 1;

        // routeSize-1 times random swap in the sequence, to shuffle the edges
        for (int idx = min_rand_int; idx < pkernelParams->routeSize; idx++)
        {
            float myrandf;
            int myrand;

            myrandf = curand_uniform(&pkernelParams->state[antIndex]);  // RND Number between 0 and 1
            myrandf *= (max_rand_int - min_rand_int + 0.999999f);
            myrandf += min_rand_int;
            myrand = (int)truncf(myrandf);

            assert(myrand <= max_rand_int);
            assert(myrand >= min_rand_int);

            int temp = antRouteOffset[idx];
            antRouteOffset[idx] = antRouteOffset[myrand];
            antRouteOffset[myrand] = temp;
        }
        /*if (antIndex == 0) {
            printf("Generated random sequence:\n ");
            sequencePrint(pkernelParams->antRoute, pkernelParams->Dist, pkernelParams->size, pkernelParams->routeSize);
        }*/
    }

    // Returns bool value of whether newParam is already listed in the route
    // Special care for node 0, which can be in the route [maxVehicles] times.
    // antindex = -1 means we are meant to look for the route vector
    __device__ bool alreadyListed(
        Kernel_ParamTypedef* pkernelParams,
        int antIndex,
        int idx,    // serial number of node in route
        int newParam
    )
    {
        assert(idx < pkernelParams->routeSize);
        if (idx >= pkernelParams->routeSize)
            return true;    // Rather make infinite cycle than overaddressing
        // Count, how many vehicles are being used (0-s in the Route)
        int vehicleCntr = 0;
        int temp;
        int* antRouteOffset = pkernelParams->antRoute
            + antIndex * pkernelParams->routeSize;   // Optimizing array addressing

        // Special care for -1: watching route vector
        if (antIndex == -1)
            antRouteOffset = pkernelParams->route;


        if (newParam == 0)
        {
            for (int i = 0; i < idx; ++i)
            {
                temp = antRouteOffset[i];
                if (temp == 0)
                    vehicleCntr++;
            }
            return vehicleCntr >= pkernelParams->maxVehicles;
            // Already listed only if max amount of vehicles are still used
        }
        // Regular node
        for (int i = 0; i < idx; ++i)   // Compare with all previous route nodes
            if (newParam == antRouteOffset[i])
                return true;    // Matching previous node
        // No match found
        return false;

    }

    // Returns the sum length of the given route of trucks
    // Returns -1 if route not possible (for example has dead end)
    __device__ float antRouteLength(Kernel_ParamTypedef* pkernelParams, int antIndex)
    {
        int* antRouteOffset = pkernelParams->antRoute
            + antIndex * pkernelParams->routeSize;   // Optimizing array addressing

        // Special care for -1: watching route vector
        if (antIndex == -1)
            antRouteOffset = pkernelParams->route;

        float length = 0;  // Return value
        int src, dst;

        int vehicleIdx = 0;

        for (int i = 0; i < pkernelParams->routeSize; ++i) {
            src = antRouteOffset[i];
            dst = antRouteOffset[(i + 1) % pkernelParams->routeSize];   // Next node
            if (src == 0)
                vehicleIdx++;

            float edgeLength = pkernelParams->Dist[src * pkernelParams->size + dst];
            if (edgeLength < 0) {
                return -1;
            }
            else {
                length += edgeLength;
            }
        }

        if (length == 0)    // Defending other functions from faulty solutions
            return -1;
        if (vehicleIdx != pkernelParams->maxVehicles)
            return -1;
        
        assert(length != 0);
        return length;
    }

    // Represents az ant who follows other ants' pheromones
    // Generates a route with Roulette wheel method given the values of the Pheromone matrix
    __device__ void followPheromones(
        Kernel_ParamTypedef* pkernelParams,
        int antIndex,
        int maxTryNumber
    )
    {
        int* antRouteOffset = pkernelParams->antRoute
            + antIndex * pkernelParams->routeSize;   // Optimizing array addressing

        curandState* statePtr = &(pkernelParams->state[antIndex]);
        // Expected to start in node 0
        antRouteOffset[0] = 0;

        // Somewhat more difficult than TSP
        int vehicleIdx = 0;
        int workingRow; // when the last node was 0, we have to watch the correct row in Pheromone matrix
        float sumPheromone = 0.0f;  // Weighted Roulette wheel: first we calculate the sum of weights
        for (int i = 0; i < pkernelParams->size; i++)
            sumPheromone += pkernelParams->Pheromone[i];


        // Starting from 2nd element of the Route
        for (int i = 1; i < pkernelParams->routeSize; ++i) {
            int source = antRouteOffset[i - 1];   // Previous node
            int newParam;   // Variable for new route element
            bool foundVertexByRoulette = false;

            workingRow = correctRow(pkernelParams->size, vehicleIdx, source);
            if (source == 0)
                vehicleIdx++;
            assert(vehicleIdx <= pkernelParams->maxVehicles);
            assert(workingRow < pkernelParams->routeSize);
            for (int j = 0; j < maxTryNumber && !foundVertexByRoulette; j++)
            {
                // RND Number between 0 and sumPheromone
                float myrandflt = curand_uniform(statePtr) * sumPheromone;
                float temp = pkernelParams->Pheromone[workingRow * pkernelParams->size + 0]; // Used to store the matrix values

                for (newParam = 0; newParam < pkernelParams->size - 1; newParam++)
                {
                    if (myrandflt < temp)   // If newparam == size-1 then no other node to choose
                        break;
                    temp += pkernelParams->Pheromone[workingRow * pkernelParams->size + newParam + 1];
                } // If not already listed then adding to the sequence
                foundVertexByRoulette = !alreadyListed(pkernelParams, antIndex, i, newParam);
            }
            if (!foundVertexByRoulette)
            {
                // Next vertex choosen by equal chances
                do {
                    float newfloat = curand_uniform(&pkernelParams->state[antIndex]);  // RND Number between 0 and 1
                    newfloat *= (pkernelParams->size - 1) + 0.999999f;  // Transforming into the needed range
                    newParam = (int)truncf(newfloat);
                } while (alreadyListed(pkernelParams, antIndex, i, newParam));
            }
            // last the new vertex
            antRouteOffset[i] = newParam;
        }
    }

    // If the last node was 0 in route, we have to calculate the row index
    // we need
    // Else we need the row of the sourceNode
    __device__ inline int correctRow(int size, int vehicleIdx, int sourceNode)
    {
        assert(sourceNode < size);
        if (sourceNode != 0)
            return sourceNode;
        if (vehicleIdx == 0)
            return 0;
        return size + vehicleIdx - 1;
    }


    // Manipulating the pheromone values according to the given solution
    // The longer the route is, the smaller amount we are adding
    // Sets the route vector if we found a best yet solution
    __device__ void evaluateSolution(
        Kernel_ParamTypedef* pkernelParams,
        int antIndex,
        float multiplConstant,
        float rewardMultiplier,
        int repNumber
    )
    {
        float length = antRouteLength(pkernelParams, antIndex);
        assert(length != 0);
        float additive = multiplConstant / length; // The longer the route is, the smaller amount we are adding
        if (length < globalParams.minRes && length > 0) { // Rewarding the ant with the best yet route
            // printf("New min found: %f, rep: %d\n", length, repNumber);
            copyAntRoute(pkernelParams, antIndex);
            globalParams.minRes = length;
            if (repNumber > 2)
                additive *= rewardMultiplier * (repNumber + 1) * (repNumber + 1);
        }
        /*if (antIndex == 0) {
            printf("kukucsfv\n");
        }*/

        // Route valid if length > 0
        if (length > 0)
        {
            int* antRouteOffset = pkernelParams->antRoute + antIndex * pkernelParams->routeSize;   // Optimizing array addressing
            int vehicleIdx = 0;
            for (int i = 0; i < pkernelParams->routeSize; i++)
            {
                int src = antRouteOffset[i];
                int workingRow; // when the last node was 0, we have to watch the correct row in Pheromone matrix
                if (src == 0)
                {
                    workingRow = correctRow(pkernelParams->size, vehicleIdx++);
                    assert(vehicleIdx <= pkernelParams->maxVehicles);
                    assert(workingRow < pkernelParams->routeSize);
                }
                else
                {
                    workingRow = src;
                }
                int dst = antRouteOffset[(i + 1) % pkernelParams->routeSize];
                
                if (workingRow > pkernelParams->routeSize)
                    printf("ujjujj %d_%d_%d\n",workingRow,dst,vehicleIdx);
                
                float* ptr = &(pkernelParams->Pheromone[workingRow * pkernelParams->size + dst]);

                atomicAdd(ptr, additive);
            }
        }
    }

    // Auxilary function for greedy sequence
    // Returns the highest vertex index not yet chosen
    /// row : row of previous route element (decides, which row to watch in the function)
    __device__ int maxInIdxRow(Kernel_ParamTypedef* pkernelParams, int row, int idx) {
        int maxidx = -1;
        float max = -1;
        assert(row < pkernelParams->routeSize);

        for (int i = 0; i < pkernelParams->size; i++)
        {
            // Go through the row elements to find the highest
            float observed = pkernelParams->Pheromone[row * pkernelParams->size + i];
            if (observed > max && !alreadyListed(pkernelParams, -1, idx, i))
            {
                max = observed;
                maxidx = i;
            }
        }
        //printf("%d. vertex with value of %.2f : %d\n", idx, max, maxidx);
        return maxidx;
    }

    // Generates a sequnce using greedy algorithm
    // Always chooses the highest possible value for the next vertex
    __device__ void greedySequence(Kernel_ParamTypedef* pkernelParams)
    {
        // Need to count which vehicle is active
        int vehicleIdx = 0;
        pkernelParams->route[0] = 0;
        for (int i = 1; i < pkernelParams->routeSize; i++)
        {
            int node = pkernelParams->route[i] = maxInIdxRow(pkernelParams, correctRow(pkernelParams->size, vehicleIdx, pkernelParams->route[i - 1]), i);
            if (node == 0)
                vehicleIdx++;
            assert(node != -1);
        }
    }

    // Copies a route into the answer vector
    __device__ void copyAntRoute(Kernel_ParamTypedef* pkernelParams, int antIndex) {
        // Optimizing array addressing
        int* antRouteOffset = pkernelParams->antRoute + antIndex * pkernelParams->routeSize;
        for (int i = 1; i < pkernelParams->routeSize; i++)
            pkernelParams->route[i] = antRouteOffset[i];
    }

    // Validates the output vector
    // return true if route syntax is correct and possible
    __device__ bool validRoute(Kernel_ParamTypedef* pkernelParams) 
    {
        // Last minute correction
        pkernelParams->route[0] = 0;

        printf("Raw data: \n");
        for (int iter = 0; iter < pkernelParams->routeSize; iter++) {
            printf("%d ", pkernelParams->route[iter]);
        }
        printf("\n");

        if (pkernelParams->route[0] != 0)
        {
            return false;
        }
        // 0 must be maxVehicles times
        if (pkernelParams->maxVehicles != nodeCount(pkernelParams, 0))
            return false;
        for (int i = 1; i < pkernelParams->size; i++) {
            if (!routeContain(pkernelParams, i))
            {
                return false;
            }
        }

        // Testing length
        return (antRouteLength(pkernelParams, -1) > 0);
    }

    // How many times does the given node appear in the sequence 
    __device__ int nodeCount(Kernel_ParamTypedef* pkernelParams, int node) {
        int count = 0;
        for (int i = 0; i < pkernelParams->routeSize; i++)
        {
            if (pkernelParams->route[i] == node)
                count++;
        }
        return count;
    }

    // Finds a value in the route vector
    __device__ bool routeContain(Kernel_ParamTypedef* pkernelParams, int value)
    {
        for (int i = 1; i < pkernelParams->routeSize; i++)
            if (pkernelParams->route[i] == value)
                return true;
        return false;
    }
}