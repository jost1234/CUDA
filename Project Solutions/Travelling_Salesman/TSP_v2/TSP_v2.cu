
// Special CUDA API headers
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cooperative_groups.h>
#include "curand.h"
#include "curand_kernel.h"

// Custom header containing Control Panel
#include "TSP.cuh"

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
    FILE* Dist_file;    // File pointer
    int Dist_filename_Idx;
    bool foundDistMatrix = false;   // Error handling
    bool foundRoute;
    int size;    // Number of graph vertices

    int i;  // Iterator
    srand(time(0));
    for (i = 1; i < argc; ++i)  // Processing command line arguments
        // Command Line Syntax: ... --dist [file_name]
        if ((strcmp(argv[i], "-d") == 0) || (strcmp(argv[i], "--dist") == 0)) {
            Dist_file = fopen(argv[++i], "r");
            if (Dist_file == NULL) {
                fprintf(stderr, "Unable to open file \"%s\"", argv[i]);
                return -1;
            }
            Dist_filename_Idx = i;
            printf("Opening file \"%s\"!\n", argv[Dist_filename_Idx]);
            foundDistMatrix = true;
        }
    if (!foundDistMatrix) {
        fprintf(stderr, "Please give a file in command line arguments to set the Distance Matrix!\n");
        fprintf(stderr, "Command Line Syntax:\n\t--dist [data_file].txt\n");
        fprintf(stderr, "File Syntax:\n\t[Number of Vehicles Available]\n\t[Number of Nodes]\n\tdist11, dist12, ...\n\tdist21 ... \n");
        return -1;
    }

    // File syntax : 1st row must contain graph size in decimal => compatible with TSP
    // Following rows: graph edge values separated with comma (,)
    if (fscanf_s(Dist_file, "%d \n", &size) == 0) {
        fprintf(stderr, "Unable to read Size!\n Make sure you have the right file syntax!\n");
        fprintf(stderr, "File Syntax:\n\t[Number of Vehicles Available]\n\t[Number of Nodes]\n\tdist11, dist12, ...\n\tdist21 ... \n");
        fclose(Dist_file);
        return -1;
    }
    // Distance matrix
    // Store type: adjacency matrix format
    DATATYPE* Dist = (DATATYPE*)calloc(size * size, sizeof(DATATYPE));

    // Reading distance values
    for (int ii = 0; ii < size; ++ii) {
        DATATYPE temp;

        for (int jj = 0; jj < size; ++jj) {
            if (fscanf_s(Dist_file, "%lf", &temp) == 0) {
                fprintf(stderr, "Error reading file \"%s\" distance(%d,%d)\n", argv[Dist_filename_Idx], ii, jj);
                fclose(Dist_file);
                return -1;
            }
            Dist[ii * size + jj] = temp;
        }
        fscanf_s(Dist_file, "\n");
    }

    // File syntax : row after dist values must contain maximum vehicle count in decimal
    if (fscanf_s(Dist_file, "%d \n", &maxVehicles) == 0) {
        fprintf(stderr, "Unable to read Maximum Vehicle Number!\n Make sure you have the right file syntax!\n");
        fprintf(stderr, "File Syntax:\n\t[Number of Vehicles Available]\n\t[Number of Nodes]\n\tdist11, dist12, ...\n\tdist21 ... \n");
        fclose(Dist_file);
        return -1;
    }

    // Closing data file
    printf("Closing file \"%s\"!\n", argv[Dist_filename_Idx]);
    if (fclose(Dist_file) != 0) {
        fprintf(stderr, "Unable to close file \"%s\"!\n", argv[Dist_filename_Idx]);
        return -1;
    }

    // Printing Matrix
    printf("Maximum number of vehicles: %zu\n", maxVehicles);
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

    printf("Vehicle Routing Problem with Ant Colony Algorithm\n");
    TSP::TSP_Ant_CUDA(params);

    return 0;
}

namespace TSP {

    // Main CUDA function
    cudaError_t TSP_Ant_CUDA(TSP_AntCUDA_ParamTypedef h_params)
    {
        cudaError_t cudaStatus;

        // Local variables
        int size = h_params.size;    // Number of graph vertices
        unsigned int antNum = h_params.antNum;    // Number of Ants (= threads) 

        if (!inputCheck(h_params)) {
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
            fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?\n");
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
        d_configParams.Alpha = ALPHA;
        d_configParams.Follower_Generations = FOLLOWER_GENERATIONS;
        d_configParams.Initial_Pheromone_Value = INITIAL_PHEROMONE_VALUE;
        d_configParams.maxTryNumber = size;
        d_configParams.Random_Generations = RANDOM_GENERATIONS;
        d_configParams.Repetitions = REPETITIONS;
        d_configParams.Reward_Multiplier = REWARD_MULTIPLIER;

        // Global variables in case of Multi Block execution
        TSP_AntKernel_Global_ParamTypedef d_globalParams;
        d_globalParams.averageDist = NULL;
        d_globalParams.invalidInput = NULL;
        d_globalParams.isolatedVertex = NULL;
    
        // Size of device malloc
        size_t Dist_bytes = size * size * sizeof(double);
        size_t route_bytes = size * sizeof(int);
        size_t foundRoute_bytes = sizeof(bool); // May be optimized, only for better transparency
        size_t antRoute_bytes = antNum * size * sizeof(int);
        size_t state_bytes = antNum * sizeof(curandState);
        // CUDA Malloc

        // Dist
        cudaStatus = cudaMalloc((void**)&d_kernelParams.Dist, Dist_bytes);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "d_Dist cudaMalloc failed!\n");
            Free_device_memory(d_kernelParams, d_globalParams);
            return cudaStatus;
        }
        // Pheromone
        cudaStatus = cudaMalloc((void**)&d_kernelParams.Pheromone, Dist_bytes);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "d_Pheromone cudaMalloc failed!\n");
            Free_device_memory(d_kernelParams, d_globalParams);
            return cudaStatus;

        }
        // Route
        cudaStatus = cudaMalloc((void**)&d_kernelParams.route, route_bytes);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "d_Route cudaMalloc failed!\n");
            Free_device_memory(d_kernelParams, d_globalParams);
            return cudaStatus;

        }
        // FoundRoute : flag
        cudaStatus = cudaMalloc((void**)&d_kernelParams.foundRoute, foundRoute_bytes);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "d_Route cudaMalloc failed!\n");
            Free_device_memory(d_kernelParams, d_globalParams);
            return cudaStatus;

        }
        // antRoute : auxiliary array
        cudaStatus = cudaMalloc((void**)&d_kernelParams.antRoute, antRoute_bytes);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "antRoute cudaMalloc failed!\n");
            Free_device_memory(d_kernelParams, d_globalParams);
            return cudaStatus;
        }
        // CUDA Random Seeds
        cudaStatus = cudaMalloc(&d_kernelParams.state, state_bytes);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMalloc failed!\n");
            Free_device_memory(d_kernelParams, d_globalParams);
            return cudaStatus;
        }

        // Global Variables only needed during multi block execution
        if (BlockNum > 1) {
            cudaStatus = cudaMalloc((void**)&d_globalParams.invalidInput, sizeof(bool));
            if (cudaStatus != cudaSuccess) {
                fprintf(stderr, "cudaMalloc failed!\n");
                Free_device_memory(d_kernelParams, d_globalParams);
                return cudaStatus;

            }
            cudaStatus = cudaMalloc((void**)&d_globalParams.isolatedVertex, sizeof(bool));
            if (cudaStatus != cudaSuccess) {
                fprintf(stderr, "cudaMalloc failed!\n");
                Free_device_memory(d_kernelParams, d_globalParams);
                return cudaStatus;

            }
            cudaStatus = cudaMalloc((void**)&d_globalParams.averageDist, sizeof(DATATYPE));
            if (cudaStatus != cudaSuccess) {
                fprintf(stderr, "cudaMalloc failed!\n");
                Free_device_memory(d_kernelParams, d_globalParams);
                return cudaStatus;
            }
        }

        // Copying data : Host -> Device
        cudaStatus = cudaMemcpy(d_kernelParams.Dist, h_params.Dist, Dist_bytes, cudaMemcpyHostToDevice);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "Dist cudaMemcpy failed!\n");
            Free_device_memory(d_kernelParams, d_globalParams);
            return cudaStatus;
        }

        h_params.route[0] = 0; // Route[0] = 0, means we are starting in vertex 0 
        cudaStatus = cudaMemcpy(d_kernelParams.route, h_params.route, route_bytes, cudaMemcpyHostToDevice);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "Route cudaMemcpy failed!\n");
            Free_device_memory(d_kernelParams, d_globalParams);
            return cudaStatus;
        }

        printf("Called function with %d Block", BlockNum);
        if (BlockNum == 1)
            printf(": \n");
        else
            printf("s: \n");
        int threadPerBlock = (antNum > BLOCK_SIZE) ? BLOCK_SIZE : antNum;
        
    }

    inline bool inputCheck(TSP_AntCUDA_ParamTypedef h_params) {
        return (1 > h_params.antNum ||
            NULL == h_params.Dist ||
            NULL == h_params.foundRoute ||
            NULL == h_params.Pheromone ||
            NULL == h_params.route ||
            2 > h_params.size);
    }

    // Frees device memory with cudaFree if pointer is not NULL
    // Important: all initial values of device pointers MUST BE NULL
    void Free_device_memory(TSP_AntKernel_ParamTypedef params, TSP_AntKernel_Global_ParamTypedef globalParams) {
        // Temporary device data structures
        if (NULL != params.Dist) cudaFree(params.Dist);
        if (NULL != params.antRoute) cudaFree(params.antRoute);
        if (NULL != params.Pheromone) cudaFree(params.Pheromone);
        if (NULL != params.route) cudaFree(params.route);
        if (NULL != params.state) cudaFree(params.state);

        // Incidental global variables
        if (NULL != globalParams.invalidInput) cudaFree(globalParams.invalidInput);
        if (NULL != globalParams.isolatedVertex) cudaFree(globalParams.isolatedVertex);
        if (NULL != globalParams.averageDist) cudaFree(globalParams.averageDist);
    }


}