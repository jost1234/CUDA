
// Special CUDA API headers
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cooperative_groups.h>
#include "curand.h"
#include "curand_kernel.h"

// Custom header containing Control Panel
#include "TSP_v2.cuh"

// General purpose headers
#include <iostream>
#include <stdbool.h>
#include <stdlib.h>
#include <assert.h>
#include <float.h>

// Cooperative groups namespace for block and grid
using namespace cooperative_groups;


// Main function
int main(int argc, char* argv[])
{
    // Variables used for reading from txt file
    FILE* pfile;    // File pointer
    int fileNameIdx;
    bool foundFile = false;   // Error handling
    bool foundRoute;
    int size;    // Number of graph vertices

    int i;  // Iterator
    srand(time(0));
    for (i = 1; i < argc; ++i) {  // Processing command line arguments
        // Command Line Syntax: ... --dist [file_name]
        if ((strcmp(argv[i], "-d") == 0) || (strcmp(argv[i], "--dist") == 0)) {
            pfile = fopen(argv[++i], "r");
            if (pfile == NULL) {
                fprintf(stderr, "Unable to open file \"%s\"", argv[i]);
                return -1;
            }
            fileNameIdx = i;
            printf("Opening file \"%s\"!\n", argv[fileNameIdx]);
            foundFile = true;
        }
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
    if (!foundFile) {
        fprintf(stderr, "Please give a file in command line arguments to set the Distance Matrix!\n");
        fprintf(stderr, "Command Line Syntax:\n\t--dist [data_file].txt\n");
        fprintf(stderr, "File Syntax:\n\t[Number of Nodes]\n\tdist11, dist12, ...\n\tdist21 ... \n");
        return -1;
    }

    // File syntax : 1st row must contain graph size in decimal => compatible with TSP
    // Following rows: graph edge values separated with comma (,)
    if (fscanf_s(pfile, "%d \n", &size) == 0) {
        fprintf(stderr, "Unable to read Size!\n Make sure you have the right file syntax!\n");
        fprintf(stderr, "File Syntax:\n\t[Number of Nodes]\n\tdist11, dist12, ...\n\tdist21 ... \n");
        fclose(pfile);
        return -1;
    }
    // Distance matrix
    // Store type: adjacency matrix format
    DATATYPE* Dist = (DATATYPE*)calloc(size * size, sizeof(DATATYPE));

    // Reading distance values
    for (int ii = 0; ii < size; ++ii) {
        double temp;

        for (int jj = 0; jj < size; ++jj) {
            if (fscanf_s(pfile, "%lf", &temp) == 0) {
                fprintf(stderr, "Error reading file \"%s\" distance(%d,%d)\n", argv[fileNameIdx], ii, jj);
                fclose(pfile);
                return -1;
            }
            Dist[ii * size + jj] = temp;
        }
        fscanf_s(pfile, "\n");
    }

    // Closing data file
    printf("Closing file \"%s\"!\n", argv[fileNameIdx]);
    if (fclose(pfile) != 0) {
        fprintf(stderr, "Unable to close file \"%s\"!\n", argv[fileNameIdx]);
        return -1;
    }

    // Printing Matrix
    printf("Given Dist matrix:\n");
    print(Dist, size);

    // Host Variables

    TSP::TSP_AntCUDA_ParamTypedef params;
    params.foundRoute = &foundRoute;
    params.antNum = ants;   // Macro
    params.size = size;
    params.Dist = Dist;
    params.Pheromone = (DATATYPE*)malloc(size * size * sizeof(DATATYPE));
    params.route = (int*)malloc(size * sizeof(int));

    printf("Travelling Salesman Problem with Ant Colony Algorithm\n");
    TSP::TSP_Ant_CUDA(params);


    free(params.Dist);
    free(params.Pheromone);
    free(params.route);
    return 0;
}

namespace TSP {

    // Main CUDA function
    cudaError_t TSP_Ant_CUDA(TSP_AntCUDA_ParamTypedef h_params)
    {
        cudaError_t cudaStatus;

        // Local variables
        int size = h_params.size;    // Number of graph vertices
        int antNum = h_params.antNum;    // Number of Ants (= threads) 

        if (!inputCheck(&h_params)) {
            fprintf(stderr, "Invalid Input values!\n");
            return cudaError_t::cudaErrorInvalidConfiguration;
        }

        // Calculates the number of Grid blocks to execute
        // Number of threads = number of ants
        int BlockNum = 1;
        if (antNum > BLOCK_SIZE) {
            BlockNum = my_ceil(antNum, BLOCK_SIZE);
            antNum = BlockNum * BLOCK_SIZE; // For better usage of parallel threads
        }

        // Choosing GPU, may be nessesary in a multi-GPU system
        cudaStatus = cudaSetDevice(0);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaSetDevice failed! Do you have a CUDA-capable GPU installed?\n");
            return cudaStatus;
        }

        // Device pointers
        TSP_AntKernel_ParamTypedef d_kernelParams;
        d_kernelParams.Dist = NULL;
        d_kernelParams.foundRoute = NULL;
        d_kernelParams.Pheromone = NULL;
        d_kernelParams.route = NULL;
        d_kernelParams.state = NULL;
        d_kernelParams.antNum = antNum;
        d_kernelParams.size = size;
        d_kernelParams.state = NULL;

        // Config parameters
        TSP_AntKernel_Config_ParamTypedef d_configParams;
        d_configParams.Rho = RHO;
        d_configParams.Follower_Generations = FOLLOWER_GENERATIONS;
        d_configParams.Initial_Pheromone_Value = INITIAL_PHEROMONE_VALUE;
        d_configParams.maxTryNumber = size;
        d_configParams.Random_Generations = RANDOM_GENERATIONS;
        d_configParams.Repetitions = REPETITIONS;
        d_configParams.Reward_Multiplier = REWARD_MULTIPLIER;

        // Global variables in case of Multi Block execution
        TSP_AntKernel_Global_ParamTypedef d_globalParams;
        d_globalParams.averageDist = 0.0f;
        d_globalParams.invalidInput = false;
        d_globalParams.isolatedVertex = false;

        // Size of device malloc
        size_t Dist_bytes = size * size * sizeof(DATATYPE);
        size_t route_bytes = size * sizeof(int);
        size_t foundRoute_bytes = sizeof(bool); // May be optimized, only for better transparency
        size_t antRoute_bytes = antNum * size * sizeof(int);
        size_t state_bytes = antNum * sizeof(curandState);
        // CUDA Malloc

        // Dist
        cudaStatus = cudaMalloc((void**)&d_kernelParams.Dist, Dist_bytes);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "d_Dist cudaMalloc failed!\n");
            Free_device_memory(d_kernelParams);
            return cudaStatus;
        }
        // Pheromone
        cudaStatus = cudaMalloc((void**)&d_kernelParams.Pheromone, Dist_bytes);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "d_Pheromone cudaMalloc failed!\n");
            Free_device_memory(d_kernelParams);
            return cudaStatus;
        }
        // Route
        cudaStatus = cudaMalloc((void**)&d_kernelParams.route, route_bytes);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "d_Route cudaMalloc failed!\n");
            Free_device_memory(d_kernelParams);
            return cudaStatus;
        }
        // FoundRoute : flag
        cudaStatus = cudaMalloc((void**)&d_kernelParams.foundRoute, foundRoute_bytes);
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
        // CUDA Random Seeds
        cudaStatus = cudaMalloc(&d_kernelParams.state, state_bytes);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMalloc failed!\n");
            Free_device_memory(d_kernelParams);
            return cudaStatus;
        }

        // Copying data : Host -> Device
        cudaStatus = cudaMemcpy(d_kernelParams.Dist, h_params.Dist, Dist_bytes, cudaMemcpyHostToDevice);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "Dist cudaMemcpy failed!\n");
            Free_device_memory(d_kernelParams);
            return cudaStatus;
        }

        h_params.route[0] = 0; // Route[0] = 0, means we are starting in vertex 0 
        cudaStatus = cudaMemcpy(d_kernelParams.route, h_params.route, route_bytes, cudaMemcpyHostToDevice);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "Route cudaMemcpy failed!\n");
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

        setup_kernel << < BlockNum, threadPerBlock >> > (d_kernelParams.state, time(NULL) * rand());

        // Kernel call

        double min = DBL_MAX;
        double sum = 0.0;

        for (int iter = 0; iter < SERIALMAXTRIES; iter++)
        {
            printf("Attempt #%d ||\n", iter);

            if (BlockNum == 1)
            {
                TSP_AntKernel_1Block << < BlockNum, threadPerBlock >> > (d_kernelParams, d_configParams);
            }
            else {
                // During Kernel call it's important to use cudaLaunchCooperativeKernel CUDA runtime launch API
            // or its CUDA driver equivalent instead of the <<<...>>> syntax

            // Sets supportsCoopLaunch=1 if the operation is supported on device 0
            // Only compute capability 6.0 or higher!
                int dev = 0;
                int supportsCoopLaunch = 0;
                cudaDeviceGetAttribute(&supportsCoopLaunch, cudaDevAttrCooperativeLaunch, dev);
                if (supportsCoopLaunch != 1) {
                    fprintf(stderr, "Cooperative Launch is not supported on this machine configuration.");
                    Free_device_memory(d_kernelParams);
                    return cudaStatus;
                }

                // Call arguments
                void* kernelArgs[] = { &d_kernelParams, &d_configParams, &d_globalParams };

                cudaLaunchCooperativeKernel((void*)TSP_AntKernel_multiBlock, BlockNum, BLOCK_SIZE, kernelArgs);
            }

            // Error handling during Kernel execution
            cudaStatus = cudaGetLastError();
            if (cudaStatus != cudaSuccess) {
                fprintf(stderr, "AntKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
                // Frees GPU device memory
                Free_device_memory(d_kernelParams);
                return cudaStatus;
            }

            // cudaDeviceSynchronize waits for the kernel to finish
            cudaStatus = cudaDeviceSynchronize();
            if (cudaStatus != cudaSuccess) {
                fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching antKernel!\n", cudaStatus);
                // Frees GPU device memory
                Free_device_memory(d_kernelParams);
                return cudaStatus;
            }

            // Copying processed data from GPU device
            cudaStatus = cudaMemcpy(h_params.route, d_kernelParams.route, route_bytes, cudaMemcpyDeviceToHost);
            if (cudaStatus != cudaSuccess) {
                fprintf(stderr, "route dev->host cudaMemcpy failed!");
                // Frees GPU device memory
                Free_device_memory(d_kernelParams);
                return cudaStatus;
            }
            cudaStatus = cudaMemcpy(h_params.foundRoute, d_kernelParams.foundRoute, sizeof(bool), cudaMemcpyDeviceToHost);
            if (cudaStatus != cudaSuccess) {
                fprintf(stderr, "foundRoute flag dev->host cudaMemcpy failed!");
                // Frees GPU device memory
                Free_device_memory(d_kernelParams);
                return cudaStatus;
            }
            //cudaStatus = cudaMemcpy(h_params.Pheromone, d_kernelParams.Pheromone, Dist_bytes, cudaMemcpyDeviceToHost);
            //if (cudaStatus != cudaSuccess) {
            //    fprintf(stderr, "Pheromone dev->host cudaMemcpy failed!");
            //    // Frees GPU device memory
            //    Free_device_memory(d_kernelParams);
            //    return cudaStatus;
            //}

            if (*(h_params.foundRoute)) {
                DATATYPE _length = sequencePrint(h_params.route, h_params.Dist, size);
                sum += _length;
                if (_length < min)
                    min = _length;
            }
            else {
                printf("Route not found!\n\n");
            }
        }
        printf("Average length: %.2f\n", sum / SERIALMAXTRIES);
        printf("Minimal length: %.2f", min);

        // Frees GPU device memory
        Free_device_memory(d_kernelParams);
        return cudaStatus;
    }

    // Testing input for main CUDA function
    // Returns true if input data syntax is good
    // Disclaimer: Only tests NULL property of pointers, does not 100% guarantee perfect data
    inline bool inputCheck(TSP_AntCUDA_ParamTypedef* ph_params) {
        return (1 < ph_params->antNum &&
            NULL != ph_params->Dist &&
            NULL != ph_params->foundRoute &&
            NULL != ph_params->Pheromone &&
            NULL != ph_params->route &&
            1 < ph_params->size);
    }

    // Frees device memory with cudaFree if pointer is not NULL
    // Important: all initial values of device pointers MUST BE NULL
    void Free_device_memory(TSP_AntKernel_ParamTypedef params) {
        // Temporary device data structures
        if (NULL != params.Dist) cudaFree(params.Dist);
        if (NULL != params.antRoute) cudaFree(params.antRoute);
        if (NULL != params.Pheromone) cudaFree(params.Pheromone);
        if (NULL != params.route) cudaFree(params.route);
        if (NULL != params.state) cudaFree(params.state);
    }

    __device__ __host__ DATATYPE sequencePrint(int* route, DATATYPE* Dist, int size) {
        if (NULL == route || NULL == Dist || 2 > size) {
            printf("Invalid input of sequencePrint!\n");
            return -1;
        }
        bool deadEnd = false;

        printf("Sequence : ");
        double l = 0;
        for (int i = 0; i < size; ++i)
        {
            int src = route[i];
            int dst = route[(i + 1) % size];
            printf("%d ", src);
            if (Dist[src * size + dst] < 0)
                deadEnd = true;
            l += Dist[src * size + dst];
        }
        printf("%d\n", route[0]);
        if (deadEnd) {
            printf("Route not possible!\n");
            return -1;
        }

        printf("Total length : %.2f\n", l);
        return l;
    }

    // 1 block sized kernel
    __global__ void TSP_AntKernel_1Block(
        TSP_AntKernel_ParamTypedef params,
        TSP_AntKernel_Config_ParamTypedef configParams
    )
    {
        // Dist (i,j) means the distance from vertex i to vertex j
        // If no edge drawn between them: Dist(i,j) = -1 (expected syntax)
        thread_block block = this_thread_block();

        int antIndex = threadIdx.x;  // Ant index 0 - (antNum-1)

        if (antIndex >= params.antNum || blockIdx.x > 0)     // Defense against overaddressing
            return;

        // Shared variables betwenn threads in the same block
        __shared__ bool invalidInput;       // Variable used to detecting invalid input
        __shared__ bool isolatedVertex;     // Variable used to detecting isolated vertex (for optimization purposes)
        __shared__ DATATYPE averageDist;    // Average edge distance
        __shared__ DATATYPE multiplicationConst;
        __shared__ DATATYPE minRes;         // Minimal found Route distance
        __shared__ int size;                // Local Copy of argument parameter

        // Initialization with thread 0
        if (antIndex == 0) {
            invalidInput = false;
            isolatedVertex = false;
            averageDist = 0.0f;

            size = params.size; // Needs to be written too many times
            *params.foundRoute = false;
            minRes = FLT_MAX;

            // Invalidate route vector
            for (int i = 0; i < size; i++)
                params.route[i] = 0;
        }

        // Input check
        if ((2 > params.antNum ||
            NULL == params.Dist ||
            NULL == params.foundRoute ||
            NULL == params.Pheromone ||
            NULL == params.route ||
            2 > params.size)) {
            if (antIndex == 0)
                printf("Invalid Input\n");
            return;
        }

        block.sync();

        if (antIndex == 0)
        {
            bool foundNeighboor = false;    // Checking if any of the nodes are isolated
            int i, j;
            for (i = 0; i < size; i++) {
                for (j = 0; j < size; j++) {
                    // Initializing Pheromone graph (anti - unitmatrix, all main diagonal elements are 0)
                    // 0 Pheromone value if no edge drawn
                    // Initial Pheromone value is of consideration in the Control panel
                    if ((i == j) || (params.Dist[i * size + j] < 0))
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
        }

        block.sync();

        if (invalidInput || isolatedVertex) {   // Invalid input, so no point of continuing
            return;                             // Case of isolated node means no route exists
        }

        // Case of only 2 nodes: handle quickly in 1 thread
        if (size == 2) {
            if (antIndex == 0) {
                if (params.Dist[0 * size + 1] > 0 && params.Dist[1 * size + 0] > 0) {    // Route exists
                    *params.foundRoute = true;
                    params.route[0] = 0;    // Route = [0 1]
                    params.route[1] = 1;
                }
            }
            block.sync();
            return;
        }

        // Left: Connected graph with at least 3 nodes
        // Calculating average distance
        if (antIndex == 0) {
            DATATYPE sum = 0.0f;   // Sum of edge values
            int numPos = 0;     // Number of edges
            for (int i = 0; i < size; i++) {
                for (int j = 0; j < size; j++)
                {
                    DATATYPE edge = params.Dist[i * size + j];
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

        // Initializing ant Routes 
        initAntRoute(&params, antIndex);
        block.sync();

        // Ants travelling to all directions
        for (int repNumber = 0; repNumber < configParams.Repetitions; repNumber++)
        {
            if (antIndex == 0)
                multiplicationConst = averageDist / configParams.Rho * 5;
            block.sync();

            // Trying for every possible second vertices
            for (int secondVertex = 1; secondVertex < size; secondVertex++)
            {
                generateRandomSolution(&params, antIndex, secondVertex);
                // Evaluating the given solution: modifies Pheromone matrix more if shorter path found
                evaluateSolution(&params, antIndex, multiplicationConst, &minRes, configParams.Reward_Multiplier, repNumber);
                block.sync();
            }

            if (antIndex == 0)
                multiplicationConst *= 2;
            block.sync();

            // Numerous random guess
            for (int j = 0; j < configParams.Random_Generations; j++)
            {
                // Seconvertex = -1 means no prescribed second vertex
                generateRandomSolution(&params, antIndex, -1);
                block.sync();

                // Evaluating the given solution: modifies Pheromone matrix more if shorter path found
                evaluateSolution(&params, antIndex, multiplicationConst, &minRes, configParams.Reward_Multiplier, repNumber);
                block.sync();
            }

            // Lots of ants following pheromone of previous ants
            for (int gen = 0; gen < configParams.Follower_Generations; gen++)
            {
                // Reducing previous pheromon values by value RHO (modifiable in the Control Panel)
                if (antIndex == 0) {
                    for (int i = 0; i < size; i++) {
                        for (int j = 0; j < size; j++)
                            params.Pheromone[i * size + j] *= configParams.Rho;
                    }
                }
                block.sync();

                // new ants following pheromone of previous ants
                followPheromones(&params, antIndex, configParams.maxTryNumber);
                block.sync();
                // Evaluating the given solution: modifies Pheromone matrix more if shorter path found
                evaluateSolution(&params, antIndex, multiplicationConst, &minRes, configParams.Reward_Multiplier, repNumber);
                block.sync();
            }
        }
        // After that we only need one ant (thread)
        if (antIndex != 0)
            return;

        // Choosing path with greedy algorithm if we dont have a valid answer
        if (!validRoute(&params)) {
            //printf("Need to find route in greedy mode!\n");
            greedySequence(&params);
        }

        block.sync();   // We found a route if given length is greater than zero
        *params.foundRoute = (antRouteLength(&params, 0) > 0);
    }

    // Multiblock sized kernel
    __global__ void TSP_AntKernel_multiBlock(
        TSP_AntKernel_ParamTypedef params,
        TSP_AntKernel_Config_ParamTypedef configParams,
        TSP_AntKernel_Global_ParamTypedef globalParams
    )
    {
        // Dist (i,j) means the distance from vertex i to vertex j
        // If no edge drawn between them: Dist(i,j) = -1 (expected syntax)
        grid_group grid = this_grid();
        if (!grid.is_valid())
            return;
        grid.sync();
        int antIndex = blockIdx.x * blockDim.x + threadIdx.x;  // ant index

        grid.sync();

        // Initialization with thread 0
        if (antIndex == 0) {
            globalParams.invalidInput = false;
            globalParams.isolatedVertex = false;
            globalParams.averageDist = 0.0f;
            globalParams.multiplicationConst = 0.0f;

            *params.foundRoute = false;
            globalParams.minRes = FLT_MAX;

            // Invalidate route vector
            for (int i = 0; i < params.size; i++)
                params.route[i] = 0;

        }

        // Input check
        if ((2 > params.antNum ||
            NULL == params.Dist ||
            NULL == params.foundRoute ||
            NULL == params.Pheromone ||
            NULL == params.route ||
            2 > params.size))
        {
            if (antIndex == 0)
                printf("Invalid Input values!\n");
            return;
        }

        grid.sync();

        if (antIndex == 0)
        {
            bool foundNeighboor = false;    // Checking if any of the nodes are isolated
            int i, j;
            for (i = 0; i < params.size; i++) {
                for (j = 0; j < params.size; j++) {
                    // Initializing Pheromone graph (anti - unitmatrix, all main diagonal elements are 0)
                    // 0 Pheromone value if no edge drawn
                    // Initial Pheromone value is of consideration in the Control panel
                    if ((i == j) || (params.Dist[i * params.size + j] < 0))
                        params.Pheromone[i * params.size + j] = 0;
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
        }
        grid.sync();

        if (globalParams.invalidInput || globalParams.isolatedVertex) {   // Invalid input, so no point of continuing
            return;                             // Case of isolated node means no route exists
        }

        // Case of only 2 nodes: handle quickly in 1 thread
        if (params.size == 2) {
            if (antIndex == 0) {
                if (params.Dist[0 * params.size + 1] > 0 && params.Dist[1 * params.size + 0] > 0)
                {    // Route exists
                    *params.foundRoute = true;
                    params.route[0] = 0;    // Route = [0 1]
                    params.route[1] = 1;
                }
            }
            grid.sync();
            return;
        }

        // Left: Connected graph with at least 3 nodes
        // Calculating average distance
        if (antIndex == 0)
        {
            DATATYPE sum = 0.0f;   // Sum of edge values
            int numPos = 0;     // Number of edges
            for (int i = 0; i < params.size; i++) {
                for (int j = 0; j < params.size; j++)
                {
                    DATATYPE edge = params.Dist[i * params.size + j];
                    if (edge > 0)
                    {
                        sum += edge;
                        numPos++;
                    }
                }
            }
            globalParams.averageDist = sum / numPos * params.size;
        }
        grid.sync();

        // Initializing ant Routes 
        initAntRoute(&params, antIndex);
        grid.sync();

        // Ants travelling to all directions
        for (int repNumber = 0; repNumber < configParams.Repetitions; repNumber++)
        {
            if (antIndex == 0) {
                globalParams.multiplicationConst = globalParams.averageDist / configParams.Rho * 5.0f;
            }
            grid.sync();

            // Trying for every possible second vertices
            for (int secondVertex = 1; secondVertex < params.size; secondVertex++)
            {
                generateRandomSolution(&params, antIndex, secondVertex);

                // Evaluating the given solution: modifies Pheromone matrix more if shorter path found
                evaluateSolution(&params, antIndex, globalParams.multiplicationConst, &globalParams.minRes, configParams.Reward_Multiplier, repNumber);
                grid.sync();
            }

            // Numerous random guess
            for (int j = 0; j < configParams.Random_Generations; j++)
            {
                // Seconvertex = -1 means no prescribed second vertex
                generateRandomSolution(&params, antIndex, -1);
                grid.sync();

                // Evaluating the given solution: modifies Pheromone matrix more if shorter path found
                evaluateSolution(&params, antIndex, globalParams.multiplicationConst, &globalParams.minRes, configParams.Reward_Multiplier, repNumber);
                grid.sync();
            }

            if (antIndex == 0)
            {
                globalParams.multiplicationConst *= 2;
                //print(params.Pheromone, params.size);
            }
            grid.sync();

            // Lots of ants following pheromone of previous ants
            for (int gen = 0; gen < configParams.Follower_Generations; gen++)
            {
                // Reducing previous pheromone values by value RHO (modifiable in the Control Panel)
                if (antIndex == 0) {
                    for (int i = 0; i < params.size; i++)
                        for (int j = 0; j < params.size; j++)
                        {
                            params.Pheromone[i * params.size + j] *= configParams.Rho;
                        }
                }
                grid.sync();

                // new ants following pheromone of previous ants
                followPheromones(&params, antIndex, configParams.maxTryNumber);
                grid.sync();

                // Evaluating the given solution: modifies Pheromone matrix more if shorter path found
                evaluateSolution(&params, antIndex, globalParams.multiplicationConst, &globalParams.minRes, configParams.Reward_Multiplier, repNumber);
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

        grid.sync();   // We found a route if given length is greater than zero

        *params.foundRoute = (antRouteLength(&params, 0) > 0);
    }

    // Inicializes a random seed for different threads
    __global__ void setup_kernel(curandState* state, unsigned long seed)
    {
        int id = blockIdx.x * blockDim.x + threadIdx.x;
        curand_init(seed, id, id, &state[id]);
    }

    // Gets initial value of Route arrays
    __device__ void initAntRoute(
        TSP_AntKernel_ParamTypedef* pkernelParams,
        int antIndex
    )
    {
        // Route init [0, 1, 2 ... size-1]
        int* antRouteOffset = pkernelParams->antRoute + antIndex * pkernelParams->size;   // Optimizing array addressing
        for (int idx = 0; idx < pkernelParams->size; idx++) {
            antRouteOffset[idx] = idx;
        }
    }

    inline __device__ bool isValidSecondVertex(int secondVertex, int size) {
        return (secondVertex > 0 && secondVertex < size);
    }

    // Generates a random sequence of numbers between 0 and (size - 1) starting with 0
    // secondVertex: Variable used for giving an arbitrary second vertex
    //      0 < secondvertex < size : valid input (condition = 1)
    //      else: invalid input, no mandatory second vertex (condition = 0)
    __device__ void generateRandomSolution(
        TSP_AntKernel_ParamTypedef* pkernelParams,
        int antIndex,
        int secondVertex
    )
    {

        int* antRouteOffset = pkernelParams->antRoute + antIndex * pkernelParams->size;   // Optimizing array addressing
        // Expected to start in node 0 (in normal use this is already set, but for safety it's here)
        antRouteOffset[0] = 0;

        int min_rand_int = 1, max_rand_int = pkernelParams->size - 1;
        if (isValidSecondVertex(secondVertex, pkernelParams->size)) {
            min_rand_int = 2;
            int secVertexidx;
            // Find secondvertex in route
            for (secVertexidx = 0; secVertexidx < pkernelParams->size && antRouteOffset[secVertexidx] != secondVertex; ++secVertexidx);
            if (secVertexidx == pkernelParams->size)   // Could not find it, something went wrong, so we must order back the sequence
            {
                // If everything is correct, we may never enter here,
                // but in case so, we reconfigure the antRoute to default
                printf("Error occured while generating random sequence: second vertex (%d) lost!\n", secondVertex);
                for (int idx = 2; idx < pkernelParams->size; idx++)
                    antRouteOffset[idx] = idx;

                antRouteOffset[1] = secondVertex;
                antRouteOffset[secondVertex] = 1;
            }
            else   // Second vertex found
            {
                antRouteOffset[secVertexidx] = antRouteOffset[1];
                antRouteOffset[1] = secondVertex;
            }
        }

        // n db random swap in the sequence, to shuffle the edges
        // executing [size] times random swaps
        // min_rand_int means the lower limit for the swap range
        // -> if there is an exact 2.vertex, then only the (3. - size.) vertex sequence needs to be changed
        for (int idx = min_rand_int; idx < pkernelParams->size; idx++)
        {
            float myrandf;
            int myrand;

            myrandf = curand_uniform(&pkernelParams->state[antIndex]);  // RND Number between 0 and 1
            myrandf *= (max_rand_int - min_rand_int + 0.999999);
            myrandf += min_rand_int;
            myrand = (int)truncf(myrandf);

            assert(myrand <= max_rand_int);
            assert(myrand >= min_rand_int);

            int temp = antRouteOffset[idx];
            antRouteOffset[idx] = antRouteOffset[myrand];
            antRouteOffset[myrand] = temp;
        }


    }

    // Returns bool value of whether newParam is already listed in the route
    // Special care for node 0, which can be in the route [maxVehicles] times.
    __device__ bool alreadyListed(
        int* antRoute,  // Too few argument to worth giving out the whole kernel struct
        int size,
        int antIndex,
        int idx,
        int newParam
    )
    {
        if (idx >= size)
            return true;    // Rather make infinite cycle than overaddressing


        for (int i = 0; i < idx; ++i)
            if (newParam == antRoute[antIndex * size + i])
                return true;
        return false;
    }

    // Returns the length of the given route
    // Returns -1 if route not possible (for example has dead end)
    __device__ DATATYPE antRouteLength(TSP_AntKernel_ParamTypedef* pkernelParams, int antIndex)
    {
        int* antRouteOffset = pkernelParams->antRoute + antIndex * pkernelParams->size;   // Optimizing array addressing
        DATATYPE length = 0;  // Return value
        int src, dst;

        for (int i = 0; i < pkernelParams->size; ++i)
        {
            src = antRouteOffset[i];
            dst = antRouteOffset[(i + 1) % pkernelParams->size];   // Next node

            DATATYPE edgeLength = pkernelParams->Dist[src * pkernelParams->size + dst];
            if (edgeLength < 0) {
                return -1;
            }
            else {
                length += edgeLength;
            }
        }
        assert(length != 0);
        return length;
    }

    // Represents az ant who follows other ants' pheromones
    // Generates a route with Roulette wheel method given the values of the Pheromone matrix
    __device__ void followPheromones(
        TSP_AntKernel_ParamTypedef* pkernelParams,
        int antIndex,
        int maxTryNumber
    )
    {
        int* antRouteOffset = pkernelParams->antRoute + antIndex * pkernelParams->size;   // Optimizing array addressing
        curandState* statePtr = &(pkernelParams->state[antIndex]);
        // Expected to start in node 0
        antRouteOffset[0] = 0;

        DATATYPE sumPheromone = 0.0f;  // Weighted Roulette wheel: first we calculate the sum of weights

        for (int i = 0; i < pkernelParams->size; i++)
            sumPheromone += pkernelParams->Pheromone[i];

        // Starting from 2nd element of the Route
        for (int i = 1; i < pkernelParams->size; ++i)
        {
            int source = antRouteOffset[i - 1];   // Previous node
            int newParam;   // Variable for new route element
            bool foundVertexByRoulette = false;
            for (int j = 0; j < maxTryNumber && !foundVertexByRoulette; j++)
            {
                // RND Number between 0 and sumPheromone
                DATATYPE myrandflt = curand_uniform(statePtr) * sumPheromone;
                DATATYPE temp = pkernelParams->Pheromone[source * pkernelParams->size + 0]; // Used to store the matrix values

                for (newParam = 0; newParam < pkernelParams->size - 1; newParam++)
                {
                    if (myrandflt < temp)   // If newparam == size-1 then no other node to choose
                        break;
                    temp += pkernelParams->Pheromone[source * pkernelParams->size + newParam + 1];
                }   // If not already listed then adding to the sequence
                foundVertexByRoulette = !alreadyListed(pkernelParams->antRoute,
                    pkernelParams->size,
                    antIndex, i, newParam);
            }
            if (!foundVertexByRoulette)
            {
                // Next vertex choosen by equal chances
                do {
                    float newfloat = curand_uniform(statePtr);      // RND Number between 0 and 1
                    newfloat *= (pkernelParams->size - 1) + 0.999999f;  // Transforming into the needed range
                    newParam = (int)truncf(newfloat);
                } while (alreadyListed(pkernelParams->antRoute,
                    pkernelParams->size,
                    antIndex, i, newParam));
            }
            // At last the new vertex
            antRouteOffset[i] = newParam;
        }

        //if (antIndex == 32) {
        //    sequencePrint(antRouteOffset, pkernelParams->Dist, pkernelParams->size);
        //}

    }

    __device__ void evaluateSolution(
        TSP_AntKernel_ParamTypedef* pkernelParams,
        int antIndex,
        DATATYPE multiplConstant,
        DATATYPE* minRes,
        DATATYPE rewardMultiplier,
        int repNumber
    )
    {
        DATATYPE length = antRouteLength(pkernelParams, antIndex);
        assert(length != 0);
        DATATYPE additive = multiplConstant / length; // The longer the route is, the smaller amount we are adding

        //if(antIndex == 0)
        //    printf("%f\n", additive);

        if (length < *minRes && length > 0.0f) {    // Rewarding the ant with the best yet route
            //printf("New min found: %f, rep: %d\n", length, repNumber);
            *minRes = length;
            copyAntRoute(pkernelParams, antIndex);

            if (repNumber > 2)
            {
                additive *= rewardMultiplier * (repNumber + 1) * (repNumber + 1);
            }

        }

        // Route valid if length > 0
        if (length > 0)
        {

            int* antRouteOffset = pkernelParams->antRoute + antIndex * pkernelParams->size;   // Optimizing array addressing
            for (int i = 0; i < pkernelParams->size; i++)
            {
                int src = antRouteOffset[i];
                int dst = antRouteOffset[(i + 1) % pkernelParams->size];
                DATATYPE* ptr = &(pkernelParams->Pheromone[src * pkernelParams->size + dst]);
                
                //if (i == 0 && antIndex == 0)
                //    printf("%f\n", additive);
                atomicAdd(ptr, additive);

                /*for (int i = 0; i < pkernelParams->antNum; i++) {
                    if(i == antIndex)
                        *ptr += additive;
                }*/

            }
        }
    }

    // Auxilary function for greedy sequence
    // Return the highest vertex index not yet chosen
    /// row : row of previous route element (decides, which row to watch in the function)
    __device__ int maxInIdxRow(TSP_AntKernel_ParamTypedef* pkernelParams, int row, int idx) {
        int maxidx = -1;
        DATATYPE max = 0;
        for (int i = 0; i < pkernelParams->size; i++) {
            // Go through the row elements to find the highest
            DATATYPE observed = pkernelParams->Pheromone[row * pkernelParams->size + i];
            if (observed > max && !alreadyListed(pkernelParams->route,
                pkernelParams->size,
                0, idx, i))
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
    __device__ void greedySequence(TSP_AntKernel_ParamTypedef* pkernelParams)
    {
        pkernelParams->route[0] = 0;
        for (int i = 1; i < pkernelParams->size; i++)
        {
            pkernelParams->route[i] = maxInIdxRow(pkernelParams, pkernelParams->route[i - 1], i);
        }
    }

    // Copies a route into the answer vector
    __device__ void copyAntRoute(TSP_AntKernel_ParamTypedef* pkernelParams, int antIndex) {
        // Optimizing array addressing
        int* antRouteOffset = pkernelParams->antRoute + antIndex * pkernelParams->size;
        for (int i = 1; i < pkernelParams->size; i++)
            pkernelParams->route[i] = antRouteOffset[i];
    }

    // Finds a value in the route vector
    __device__ bool routeContain(TSP_AntKernel_ParamTypedef* pkernelParams, int value)
    {
        for (int i = 1; i < pkernelParams->size; i++)
            if (pkernelParams->route[i] == value)
                return true;
        return false;
    }

    // Validates the output vector
    __device__ bool validRoute(TSP_AntKernel_ParamTypedef* pkernelParams) {
        if (pkernelParams->route[0] != 0)
        {
            return false;
        }

        for (int i = 1; i < pkernelParams->size; i++)
        {
            if (!routeContain(pkernelParams, i))
            {
                return false;
            }
        }

        return true;
    }
}