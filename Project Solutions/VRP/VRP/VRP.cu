
// Special CUDA API headers
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cooperative_groups.h>
#include "curand.h"
#include "curand_kernel.h"

// Custom header containing Control Panel
#include "VRP.cuh"

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
    int maxVehicles; // Maximum number of vehicles in warehouse
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
    printf("Maximum number of vehicles: %zu\n",maxVehicles);
    printf("Given Dist matrix:\n");
    print(Dist, size);

    // Host Variables
    
    // Route: [0 ... 1st Route ... 0 ... 2nd Route ... ... Last Route ... ( Last 0 not stored)]

    VRP_AntCUDA_ParamTypedef params;
    params.foundRoute = &foundRoute;
    params.antNum = ants;   // Macro
    params.size = size;
    params.maxVehicles = maxVehicles;
    params.Dist = Dist;
    params.Pheromone = (DATATYPE*)malloc(size * RouteSize(size, maxVehicles) * sizeof(DATATYPE));
    params.route = (int*)malloc( RouteSize(size,maxVehicles) * sizeof(int));

    printf("Vehicle Routing Problem with Ant Colony Algorithm\n");
    VRP_Ant_CUDA(params);

    return 0;
}

// Main CUDA function
cudaError_t VRP_Ant_CUDA(VRP_AntCUDA_ParamTypedef h_params) {
    cudaError_t cudaStatus;

    // Local variables
    int maxVehicles = h_params.maxVehicles; // Maximum number of vehicles in warehouse
    int size = h_params.size;    // Number of graph vertices
    unsigned int antNum = h_params.antNum;    // Number of Ants (= threads)

    // Invalid inputs
    if (size < 2 || antNum < 2 || maxVehicles < 1) {
        fprintf(stderr, "Incorrect input values! Check antNum, size of maxVehicles!\n");
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
    VRP_AntKernel_ParamTypedef d_kernelParams;
    d_kernelParams.Dist = NULL;
    d_kernelParams.foundRoute = NULL;
    d_kernelParams.Pheromone = NULL;
    d_kernelParams.route = NULL;
    d_kernelParams.state = NULL;
    d_kernelParams.antNum = antNum;
    d_kernelParams.size = size;
    d_kernelParams.maxVehicles = maxVehicles;
    d_kernelParams.state = NULL;

    // Config parameters
    VRP_AntKernel_Config_ParamTypedef d_configParams;
    d_configParams.Alpha = ALPHA;
    d_configParams.Follower_Generations = FOLLOWER_GENERATIONS;
    d_configParams.Initial_Pheromone_Value = INITIAL_PHEROMONE_VALUE;
    d_configParams.maxTryNumber = size;
    d_configParams.Random_Generations = RANDOM_GENERATIONS;
    d_configParams.Repetitions = REPETITIONS;
    d_configParams.Reward_Multiplier = REWARD_MULTIPLIER;

    // Global variables in case of Multi Block execution
    VRP_AntKernel_Global_ParamTypedef d_globalParams;
    d_globalParams.averageDist = NULL;
    d_globalParams.invalidInput = NULL;
    d_globalParams.isolatedVertex = NULL;

    // Size of device malloc
    size_t Dist_bytes = size * size * sizeof(DATATYPE);
    size_t Pheromone_bytes = size * RouteSize(size, maxVehicles) * sizeof(DATATYPE);
        // We need memory for multiple Routes
    size_t Route_bytes = RouteSize(size, maxVehicles) * sizeof(int);  
    size_t FoundRoute_bytes = sizeof(bool); // May be optimized, only for better transparency

    size_t antRoute_bytes = antNum * Route_bytes;   // Allocating working memory for all threads
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
    cudaStatus = cudaMalloc((void**)&d_kernelParams.route, Route_bytes);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "d_Route cudaMalloc failed!\n");
        Free_device_memory(d_kernelParams, d_globalParams);
        return cudaStatus;

    }
    // FoundRoute : flag
    cudaStatus = cudaMalloc((void**)&d_kernelParams.foundRoute, FoundRoute_bytes);
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
    // state : random seeds for threads
    cudaStatus = cudaMalloc((void**)&d_kernelParams.state, state_bytes);
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
    cudaStatus = cudaMemcpy(d_kernelParams.route, h_params.route, Route_bytes, cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "route cudaMemcpy failed!\n");
        Free_device_memory(d_kernelParams, d_globalParams);
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

    double min = DBL_MAX;
    double sum = 0.0;

    for (int iter = 0; iter < SERIALMAXTRIES; iter++)
    {
        printf("Attempt #%d ||\n ", iter);

        if (BlockNum == 1) {
            VRP_AntKernel_1Block <<< 1, threadPerBlock >>> (d_kernelParams, d_configParams);
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
                Free_device_memory(d_kernelParams, d_globalParams);
                return cudaStatus;
            }

            // Call arguments
            void* kernelArgs[] = { &d_kernelParams, &d_configParams, &d_globalParams };

            cudaLaunchCooperativeKernel((void*)VRP_AntKernel_multiBlock, BlockNum, threadPerBlock, kernelArgs);
        }


        // Error handling during Kernel execution
        cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "AntKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
            // Frees GPU device memory
            Free_device_memory(d_kernelParams, d_globalParams);
            return cudaStatus;
        }

        // cudaDeviceSynchronize vár arra, hogy befejeződjön a kernel, utána visszatér
        cudaStatus = cudaDeviceSynchronize();
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching antKernel!\n", cudaStatus);
            // Frees GPU device memory
            Free_device_memory(d_kernelParams, d_globalParams);
            return cudaStatus;
        }

        // Copying processed data from GPU device
        cudaStatus = cudaMemcpy(h_params.route, d_kernelParams.route, Route_bytes, cudaMemcpyDeviceToHost);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "route dev->host cudaMemcpy failed!");
            // Frees GPU device memory
            Free_device_memory(d_kernelParams, d_globalParams);
            return cudaStatus;
        }
        cudaStatus = cudaMemcpy(h_params.foundRoute, d_kernelParams.foundRoute, sizeof(bool), cudaMemcpyDeviceToHost);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "foundRoute flag dev->host cudaMemcpy failed!");
            // Frees GPU device memory
            Free_device_memory(d_kernelParams, d_globalParams);
            return cudaStatus;
        }
        cudaStatus = cudaMemcpy(h_params.Pheromone, d_kernelParams.Pheromone, Dist_bytes, cudaMemcpyDeviceToHost);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "Pheromone dev->host cudaMemcpy failed!");
            // Frees GPU device memory
            Free_device_memory(d_kernelParams, d_globalParams);
            return cudaStatus;
        }

        if (*(h_params.foundRoute)) {
            DATATYPE _length = sequencePrint(h_params.route, h_params.Dist, size, RouteSize(size, maxVehicles));
            sum += _length;
            if (_length < min)
                min = _length;
        }
        else {
            printf("Route not found!\n\n");
        }
    }

    // Frees GPU device memory
    Free_device_memory(d_kernelParams, d_globalParams);
    return cudaStatus;
}



// Frees device memory with cudaFree if pointer is not NULL
// Important: all initial values of device pointers MUST BE NULL
void Free_device_memory(VRP_AntKernel_ParamTypedef params, VRP_AntKernel_Global_ParamTypedef globalParams) {
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

__device__ __host__ double sequencePrint(int* Route, DATATYPE* Dist, int size, int routeSize) {
    if (routeSize == 0 || NULL == Route || NULL == Dist)
    {
        printf("Invalid input!");
        return -1;
    }
    printf("Sequence :\n");
    
    DATATYPE l = 0;
    int vehicleCntr = 0, i=0;

    printf("Vehicle #0 : ");
    while (i < routeSize) {
        int src = Route[i];
        int dst = Route[(i + 1) % routeSize];
        
        // Egy út vége
        if (dst == 0) {
            if (src == 0)
                printf("Unused\nVehicle #%d : ", ++vehicleCntr);
            else if(routeSize -1 != i) {
                printf("%d (%.0f) 0\nVehicle #%d : ",src, Dist[src * size + dst], ++vehicleCntr);
            }
            else {
                printf("%d (%.0f) 0\n",src, Dist[src * size + dst]);
            }
        }
        else{
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
__global__ void VRP_AntKernel_1Block(
    VRP_AntKernel_ParamTypedef params,
    VRP_AntKernel_Config_ParamTypedef configParams
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


    

    // Shared variables betwenn threads in the same block
    __shared__ bool invalidInput;       // Variable used to detecting invalid input
    __shared__ bool isolatedVertex;     // Variable used to detecting isolated vertex (for optimization purposes)
    __shared__ DATATYPE averageDist;    // Average edge distance
    __shared__ DATATYPE minRes;         // Minimal found Route distance
    __shared__ int size;             // Local Copy of argument parameter
    // Initialization with thread 0
    if (tr == 0) {
        invalidInput = false;
        isolatedVertex = false;
        averageDist = 0.0;

        size = params.size; // Needs to be written too many times
        *params.foundRoute = false;
        minRes = DBL_MAX;

    }       

    // Managing when size=0 or size=1
    if (size < 2 || params.maxVehicles < 1)   // No graph, meaningless input
        return;

    block.sync();

    if (tr == 0)
    {
        bool foundNeighboor = false;    // Checking if any of the nodes are isolated
        int ii, jj;
        for (ii = 0; ii < size; ii++) {
            for (jj = 0; jj < size; jj++) {
                // Initializing Pheromone graph (anti - unitmatrix, all main diagonal elements are 0)
                // 0 Pheromone value if no edge drawn
                // Initial Pheromone value is of consideration in the Control panel
                if ( (ii == jj) || (params.Dist[ii * size + jj] < 0) )
                    params.Pheromone[ii * size + jj] = 0;
                else
                    params.Pheromone[ii * size + jj] = configParams.Initial_Pheromone_Value;

                // Error handling 
                // Check if there are invalid given elements 
                // Valid input if: positive OR -1 OR 0 (only if i=j)
                if (ii != jj && params.Dist[ii * size + jj] <= 0 && params.Dist[ii * size + jj] != -1) 
                {
                    printf("Dist(%d,%d) incorrect!\n", ii, jj);
                    invalidInput = true;
                    break;
                }
                if (!foundNeighboor && params.Dist[ii * size + jj] > 0) {
                    // Has neighboor therefore not isolated
                    foundNeighboor = true;
                }
            }
            if (!foundNeighboor) { // Did not have any neighboors => wrong model of VRP
                printf("Vertex %d isolated!\n", ii);
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
        if (tr == 0) {
            if (params.Dist[0 * size + 1] > 0 && params.Dist[1 * size + 0] > 0) {    // Route exists
                *params.foundRoute = true;
                params.route[0] = 0;    // Route = [0 1 0...0]
                params.route[1] = 1;
                for (int ii = 0; ii < params.maxVehicles - 1; ++ii)
                    params.route[size + ii] = 0; // Only one vehicle needed
            }
        }
        block.sync();
        return;
    }

    

    // Left: Connected graph with at least 3 nodes
    // Calculating average distance
    if (tr == 0) {
        DATATYPE sum = 0.0;   // Sum of edge values
        int numPos = 0;     // Number of edges
        for (int i = 0; i < size; i++)
            for (int j = 0; j < size; j++) {
                DATATYPE edge = params.Dist[i * size + j];
                if (edge > 0) 
                {
                    sum += edge;
                    numPos++;
                }
            }
        
        averageDist = sum / numPos * size;
        //printf("Average dist: %f\n", averageDist);
    }
    block.sync();


    // Initializing ant Routes 
    initAntRoute(params, antIndex);
    block.sync();

    // Ants travelling to all directions
    for (int repNumber = 0; repNumber < configParams.Repetitions; repNumber++) 
    {
        // Numerous random guess
        for (int j = 0; j < configParams.Random_Generations; j++)
        {
            generateRandomSolution(params, antIndex);
            DATATYPE multiplConstant = averageDist / configParams.Alpha * (configParams.Repetitions + 1 - repNumber);
            block.sync();

            // Evaluating the given solution: modifies Pheromone matrix more if shorter path found
            evaluateSolution(params, antIndex, multiplConstant,&minRes,configParams.Reward_Multiplier,repNumber);
            block.sync();
        }

        // Lots of ants following pheromone of previous ants
        for (int gen = 0; gen < configParams.Follower_Generations; gen++)
        {
            // Reducing previous pheromon values by value ALPHA (modifiable in the Control Panel)
            if (tr == 0) {
                for (int ii = 0; ii < size; ii++)
                    for (int jj = 0; jj < size; jj++) {
                        params.Pheromone[ii * size + jj] *= configParams.Alpha;
                    }
            }
            block.sync();

            // new ants following pheromone of previous ants
            followPheromones(params, antIndex, configParams.maxTryNumber);
            block.sync();
            DATATYPE multiplConstant = averageDist / configParams.Alpha * 10;
            // Evaluating the given solution: modifies Pheromone matrix more if shorter path found
            evaluateSolution(params, antIndex, multiplConstant, &minRes, configParams.Reward_Multiplier, repNumber);
            block.sync();
        }

    }
    // After that we only need one ant (thread)
    if (tr != 0)
        return;

    // Choosing path with greedy algorithm
    greedySequence(params);
    //sequencePrint(antRoute, Dist, size);

    block.sync();
    DATATYPE __length;
    __length = antRouteLength(params,0);
    *(params.foundRoute) = (__length > 0);


    //sequencePrint(params.Route, params.Dist, size, size + params.maxVehicles - 1);
}

// Multiblock sized kernel
__global__ void VRP_AntKernel_multiBlock(
    VRP_AntKernel_ParamTypedef params,
    VRP_AntKernel_Config_ParamTypedef configParams,
    VRP_AntKernel_Global_ParamTypedef globalParams) {

}


// Evaluates the given solution: modifies Pheromone matrix more if shorter path found
__device__ void evaluateSolution(
    VRP_AntKernel_ParamTypedef kernelParams,
    int antIndex,
    DATATYPE multiplConstant, 
    DATATYPE* minRes, 
    DATATYPE rewardMultiplier,
    int repNumber
) 
{
    DATATYPE length = antRouteLength(kernelParams,antIndex);
    DATATYPE additive; 
    assert(length != 0);
    additive = multiplConstant / length; // The longer the route is, the smaller amount we are adding

    if (length < *minRes && length > 0) {    // Rewarding the ant with the best yet route
        *minRes = length;
        additive *= rewardMultiplier * (repNumber + 1) * (repNumber + 1);
    }

    // Route valid if length > 0
    bool Dist_0_0_already_rewarded = false; // Bool variable to make sure Dist(0,0) only gets rewarded once
    DATATYPE* ptr;
    int routeSize = RouteSize(kernelParams.size, kernelParams.maxVehicles);
    int* antRouteOffset = kernelParams.antRoute + antIndex * routeSize;   // Optiminzing array addressing
    if (length > 0) {
        for (int i = 0; i < routeSize; i++) {
            int src = antRouteOffset[i];
            int dst = antRouteOffset[(i + 1) % routeSize];
            if (0 == src && 0 == dst && !Dist_0_0_already_rewarded) {
                Dist_0_0_already_rewarded = true;
                ptr = &(kernelParams.Pheromone[src * kernelParams.size + dst]);
                atomicAdd(ptr, additive);
            }
            else {
                ptr = &(kernelParams.Pheromone[src * kernelParams.size + dst]);
                atomicAdd(ptr, additive);
            }
        }
    }
}

// Inicializes a random seed for different threads
__global__ void setup_kernel(curandState* state, unsigned long seed) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    curand_init(seed, id, id, &state[id]);
}

// Gets initial value of Route arrays
__device__ void initAntRoute(
    VRP_AntKernel_ParamTypedef kernelParams,
    unsigned int antIndex
)
{
    // Route init [0, 1, 2 ... size-1, 0, 0 ... 0]

    int routeSize = RouteSize(kernelParams.size, kernelParams.maxVehicles);
    int* antRouteOffset = kernelParams.antRoute + antIndex * routeSize;   // Optiminzing array addressing

    for (int idx1 = 0; idx1 < kernelParams.size; ++idx1) {
        antRouteOffset[idx1] = idx1;
    }
    for (int idx2 = 0; idx2 < kernelParams.maxVehicles - 1; ++idx2) {
        antRouteOffset[kernelParams.size + idx2] = 0;
    }
};


// Generates a random sequence of numbers between 0 and (size - 1) starting with 0
__device__ void generateRandomSolution(
    VRP_AntKernel_ParamTypedef kernelParams,
    unsigned int antIndex
)
{
    int routeSize = RouteSize(kernelParams.size, kernelParams.maxVehicles);
    int* antRouteOffset = kernelParams.antRoute + antIndex * routeSize;   // Optimizing array addressing
    // Expected to start in node 0 (in normal use this is already set)
    antRouteOffset[0] = 0;
    // Route init [0, 1, 2 ... size-1, 0, 0 ... 0]
    int min_rand_int = 1, max_rand_int = kernelParams.size + kernelParams.maxVehicles - 2;

    // n+k-2 times random swap in the sequence, to shuffle the edges
    for (int idx = min_rand_int; idx < routeSize; idx++) 
    {
        float myrandf;
        int myrand;

        myrandf = curand_uniform(&kernelParams.state[antIndex]);  // RND Number between 0 and 1
        myrandf *= (max_rand_int - min_rand_int + 0.999999);
        myrandf += min_rand_int;
        myrand = (int)truncf(myrandf);

        assert(myrand <= max_rand_int);
        assert(myrand >= min_rand_int);

        int temp = antRouteOffset[idx];
        antRouteOffset[idx] = antRouteOffset[myrand];
        antRouteOffset[myrand] = temp;
    }
    //if (antIndex == 0) {
    //    printf("Generated random sequence: ");
    //    sequencePrint(kernelParams.antRoute, kernelParams.Dist, kernelParams.size, routeSize);
    //}
}


// Returns bool value of whether newParam is already listed in the route
// Special care for node 0, which can be in the route [maxVehicles] times.
__device__ bool alreadyListed(
    int* antRoute,  // Too few argument to worth giving out the whole kernel struct
    int size,
    int maxVehicles,
    int antIndex,
    int idx, 
    int newParam
) 
{
    int routeSize = RouteSize(size, maxVehicles);
        
    if (idx >= routeSize)    
        return true;    // Rather make infinite cycle than overaddressing

    // Count, how many vehicles are being used (0-s in the Route)
    int vehicleCntr = 0;
    int temp;
    int* antRouteOffset = antRoute + antIndex * routeSize;   // Optimizing array addressing
    if (newParam == 0) {   // Need to handle separately, because there can be many 0-s in the route
        for (int i = 0; i < idx; ++i)
        {
            temp = antRouteOffset[i];
            if (temp == 0)
                vehicleCntr++;
        }
        return vehicleCntr >= maxVehicles;
        // Already listed only if max amount of vehicles are still used
    }
    // Regular node
    for (int i = 0; i < idx; ++i)   // Compare with all previous route nodes
        if (newParam == antRouteOffset[i])
            return true;    // Matching previous node
    // No match found
    return false;
}

// Returns the length of the given route
// Returns -1 if route not possible (for example has dead end)
__device__ DATATYPE antRouteLength(VRP_AntKernel_ParamTypedef kernelParams,int antIndex) 
{
    int routeSize = RouteSize(kernelParams.size, kernelParams.maxVehicles);
    int* antRouteOffset = kernelParams.antRoute + antIndex * routeSize;   // Optiminzing array addressing
    DATATYPE length = 0;  // Return value
    int src, dst;

    for (int i = 0; i < routeSize; ++i) {
        src = antRouteOffset[i];
        dst = antRouteOffset[(i + 1) % routeSize];   // Next node

        DATATYPE edgeLength = kernelParams.Dist[src * kernelParams.size + dst];
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
    VRP_AntKernel_ParamTypedef kernelParams,
    unsigned int antIndex,
    int maxTryNumber
) 
{
    int routeSize = RouteSize(kernelParams.size, kernelParams.maxVehicles);
    int* antRouteOffset = kernelParams.antRoute + antIndex * routeSize;   // Optimizing array addressing
    curandState* statePtr = &(kernelParams.state[antIndex]);
    // Expected to start in node 0
    antRouteOffset[0] = 0;

    DATATYPE sumPheromone = 0.0;  // Weighted Roulette wheel: firstly calculating the sum of weights
    for (int i = 0; i < kernelParams.size; i++)
        sumPheromone += kernelParams.Pheromone[i];

    // Starting from 2nd element of the Route
    for (int i = 1; i < routeSize; ++i) {
        int source = antRouteOffset[i - 1];   // Prev element in route
        int newParam;   // Variable for new route element
        bool foundVertexByRoulette = false;
        
        for (int j = 0; j < maxTryNumber && !foundVertexByRoulette; j++) {
            // RND Number between 0 and sumPheromone
            
            DATATYPE myranddbl = curand_uniform_double(statePtr) * sumPheromone;
            DATATYPE* PheromoneOffset = &kernelParams.Pheromone[source * kernelParams.size];
            DATATYPE temp = PheromoneOffset[0]; // Used for storing the matrix values
        
            for (newParam = 0; newParam < kernelParams.size - 1; newParam++) 
            {
                if (myranddbl < temp)   // If newparam = size-1 then no other node to choose
                    break;
                temp += kernelParams.Pheromone[source * kernelParams.size + newParam + 1];
            }   // If not already listed then adding to the sequence
            foundVertexByRoulette = !alreadyListed( kernelParams.antRoute, 
                                                    kernelParams.size,
                                                    kernelParams.maxVehicles, 
                                                    antIndex, i, newParam);
        }
        if (!foundVertexByRoulette) {
            // Next vertex choosen by equal chances
            do {
                float newfloat = curand_uniform(&kernelParams.state[antIndex]);  // RND Number between 0 and 1
                newfloat *= (kernelParams.size - 1) + 0.999999;  // Transforming into the needed range
                newParam = (int)truncf(newfloat);
            } while (alreadyListed( kernelParams.antRoute,
                                    kernelParams.size,
                                    kernelParams.maxVehicles,
                                    antIndex, i, newParam));
        }
        // At last the new vertex
        antRouteOffset[i] = newParam;
    }

}

// Auxilary function for greedy sequence
// Return the highest vertex index not yet chosen
/// row : row of previous route element (decides, which row to watch in the function)
__device__ int maxInIdxRow(VRP_AntKernel_ParamTypedef kernelParams, int row, int idx) {
    int maxidx = -1;
    DATATYPE max = 0;
    for (int i = 0; i < kernelParams.size; i++) {
        // Go through the row elements to find the highest
        DATATYPE observed = kernelParams.Pheromone[row * kernelParams.size + i];
        if (observed > max && !alreadyListed(kernelParams.route,
                                             kernelParams.size,
                                             kernelParams.maxVehicles,
                                             0, idx, i)
            ) {
            max = observed;
            maxidx = i;
        }
    }
    //printf("%d. vertex with value of %.2f : %d\n", idx, max, maxidx);

    return maxidx;
}

// Generates a sequnce using greedy algorithm
// Always chooses the highest possible value for the next vertex
__device__ void greedySequence(VRP_AntKernel_ParamTypedef kernelParams) 
{
    // Size of Route is not equal to vertex size
    int routeSize = RouteSize(kernelParams.size, kernelParams.maxVehicles);
    kernelParams.route[0] = 0;
    for (int i = 1; i < routeSize; i++) {
        kernelParams.route[i] = maxInIdxRow(kernelParams, kernelParams.route[i - 1], i);
    }
}