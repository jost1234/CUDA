
// Special CUDA API headers
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cooperative_groups.h>
#include "curand.h"
#include "curand_kernel.h"

// Custom header containing Control Panel
#include "Header.h"

// General purpose headers
#include <stdio.h>
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
    bool foundDistMatrix = false;
    size_t size;    // Number of graph vertices
    
    int i;  // Iterator
    srand(time(0));
    for (i = 1; i < argc; ++i)  // Processing command line arguments
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
        fprintf(stderr, "Syntax: --dist [data_file].txt\n");
        return -1;
    }

    // File syntax : 1. row must contain graph size in decimal
    // Following rows: graph edge values separated with comma (,)
    if (fscanf_s(Dist_file, "%zu \n", &size) == 0) {
        fprintf(stderr, "Unable to read size! Make sure you have the right file syntax!\n");
        fclose(Dist_file);
        return -1;
    }
    // Distance matrix
    double* Dist = (double*)calloc(size * size, sizeof(double));
    
    // Variables used to calculating average edge length (for comparison with the end result)
    double sum = 0.0; /////
    int numpos = 0;
    for (int ii = 0; ii < size; ++ii) {
        double temp;

        for (int jj = 0; jj < size; ++jj) {
            if (fscanf_s(Dist_file, "%lf", &temp) == 0) {
                fprintf(stderr, "Error reading file \"%s\"\n", argv[Dist_filename_Idx]);
                fclose(Dist_file);
                return -1;
            }
            Dist[ii * size + jj] = temp;
            if (temp > 0) {
                sum += temp;
                numpos++;
            }    //////
        }
        fscanf_s(Dist_file, "\n");
    }
    printf("Closing file \"%s\"!\n", argv[Dist_filename_Idx]);
    if (fclose(Dist_file) != 0) {
        fprintf(stderr, "Unable to close file \"%s\"!\n", argv[Dist_filename_Idx]);
    }
    // Printing given matrix
    printf("Given Dist matrix:\n");
    print(Dist, size);

    printf("Average distance : %.2f\n", sum / numpos * size); //////

    // Host variables
    double* Pheromone = (double*)malloc(size * size * sizeof(double));
    if (Pheromone == NULL)
        goto End;
    // Sequence vector: Route[0] = 0, which means we start in vertex 0 
    int* Route = (int*)malloc(size * sizeof(int));
    if (Route == NULL)
        goto End;
    bool found;
    printf("Travelling Salesman problem with Ant Colony Algorithm\n");
    cudaError_t CUDAstate = TSP_Ant_CUDA(Dist, Route, Pheromone, &found, ants, size);


End:
    getchar();
    free(Pheromone);
    free(Route);

    return 0;
}


// Main CUDA function
cudaError_t TSP_Ant_CUDA(double* h_Dist, int* h_Route, double* h_Pheromone, bool* h_FoundRoute, unsigned int antNum, size_t size) {

    // size = 0,1 : invalid inputs
    if (size == 0 || size == 1 || antNum == 0 || antNum == 1) {
        fprintf(stderr, "Incorrect size or antNum! Must be at least 2.\n");
        return cudaError_t::cudaErrorInvalidConfiguration;
    }

    // Calculates the number of Grid blocks to execute
    // Number of threads = number of ants
    int BlockNum = 1;
    if (antNum > BLOCK_SIZE) {
        BlockNum = my_ceil(antNum, BLOCK_SIZE);
        antNum = BlockNum * BLOCK_SIZE; // For better usage of parallel threads
    }

    cudaError_t cudaStatus;
    // Choosing GPU, may be nessesary in a multi-GPU system
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed! Do you have a CUDA-capable GPU installed?\n");
        return cudaStatus;
    }

    // Device pointers
    double* d_Dist = NULL;
    bool* d_FoundRoute = NULL;

    double* d_Pheromone = NULL;
    int* antRoute = NULL, * d_Route = NULL;
    curandState* devStates = NULL;
    // We need global variables when executing numerous grid blocks
    // Alternate solution: giving status flag as function parameters
    
    bool* d_invalidInput = NULL;   // Variable used to detecting invalid input
    bool* d_isolatedVertex = NULL;  // Variable used to detecting isolated vertex (for optimization purposes)
    double* d_averageDist = NULL;
    // Size of device malloc
    size_t Dist_bytes = size * size * sizeof(double);
    size_t Route_bytes = size * sizeof(int);
    size_t FoundRoute_bytes = sizeof(bool); // May be optimized, only for better transparency

    size_t antRoute_bytes = antNum * size * sizeof(int);

    // CUDA Malloc

    // Dist
    cudaStatus = cudaMalloc((void**)&d_Dist, Dist_bytes);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "d_Dist cudaMalloc failed!\n");
        Free_device_memory(d_Dist, d_Pheromone, d_Route, d_FoundRoute, antRoute, d_invalidInput, d_isolatedVertex, d_averageDist, devStates);
        return cudaStatus;
    }
    // Pheromone
    cudaStatus = cudaMalloc((void**)&d_Pheromone, Dist_bytes);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "d_Pheromone cudaMalloc failed!\n");
        Free_device_memory(d_Dist, d_Pheromone, d_Route, d_FoundRoute, antRoute, d_invalidInput, d_isolatedVertex, d_averageDist, devStates);
        return cudaStatus;
        
    }
    // Route
    cudaStatus = cudaMalloc((void**)&d_Route, Route_bytes);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "d_Route cudaMalloc failed!\n");
        Free_device_memory(d_Dist, d_Pheromone, d_Route, d_FoundRoute, antRoute, d_invalidInput, d_isolatedVertex, d_averageDist, devStates);
        return cudaStatus;
        
    }
    // FoundRoute : flag
    cudaStatus = cudaMalloc((void**)&d_FoundRoute, FoundRoute_bytes);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "d_Route cudaMalloc failed!\n");
        // Temporary device data structures
        Free_device_memory(d_Dist, d_Pheromone, d_Route, d_FoundRoute, antRoute, d_invalidInput, d_isolatedVertex, d_averageDist, devStates);
        return cudaStatus;
        
    }
    // antRoute : auxiliary array
    cudaStatus = cudaMalloc((void**)&antRoute, antRoute_bytes);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "antRoute cudaMalloc failed!\n");
        // Temporary device data structures
        Free_device_memory(d_Dist, d_Pheromone, d_Route, d_FoundRoute, antRoute, d_invalidInput, d_isolatedVertex, d_averageDist, devStates);
        return cudaStatus;
    }

    cudaStatus = cudaMalloc(&devStates, antNum * sizeof(curandState));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!\n");
        Free_device_memory(d_Dist, d_Pheromone, d_Route, d_FoundRoute, antRoute, d_invalidInput, d_isolatedVertex, d_averageDist, devStates);
        return cudaStatus;
    }

    cudaStatus = cudaMalloc((void**)&d_invalidInput, sizeof(bool));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!\n");
        // Temporary device data structures
        Free_device_memory(d_Dist, d_Pheromone, d_Route, d_FoundRoute, antRoute, d_invalidInput, d_isolatedVertex, d_averageDist, devStates);
        return cudaStatus;
        
    }
    cudaStatus = cudaMalloc((void**)&d_isolatedVertex, sizeof(bool));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!\n");
        Free_device_memory(d_Dist, d_Pheromone, d_Route, d_FoundRoute, antRoute, d_invalidInput, d_isolatedVertex, d_averageDist, devStates);
        return cudaStatus;
        
    }
    cudaStatus = cudaMalloc((void**)&d_averageDist, sizeof(double));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!\n");
        Free_device_memory(d_Dist, d_Pheromone, d_Route, d_FoundRoute, antRoute, d_invalidInput, d_isolatedVertex, d_averageDist, devStates);
        return cudaStatus;
    }

    // Copying data : Host -> Device
    cudaStatus = cudaMemcpy(d_Dist, h_Dist, Dist_bytes, cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "Dist cudaMemcpy failed!\n");
        Free_device_memory(d_Dist, d_Pheromone, d_Route, d_FoundRoute, antRoute, d_invalidInput, d_isolatedVertex, d_averageDist, devStates);
        return cudaStatus;
    }

    h_Route[0] = 0; // Route[0] = 0, means we are starting in vertex 0 
    cudaStatus = cudaMemcpy(d_Route, h_Route, Route_bytes, cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "Route cudaMemcpy failed!\n");
        Free_device_memory(d_Dist, d_Pheromone, d_Route, d_FoundRoute, antRoute, d_invalidInput, d_isolatedVertex, d_averageDist, devStates);
        return cudaStatus;
    }


    printf("Called function with %d Block(s):  \n", BlockNum);
    int threadPerBlock = (antNum > BLOCK_SIZE) ? BLOCK_SIZE : antNum;
    
    
    // setup seeds

    setup_kernel <<< BlockNum, threadPerBlock >>> (devStates, time(NULL) * rand());

    // Kernel call

    double min = DBL_MAX;
    double sum = 0.0;

    for (int iter = 0; iter < SERIALMAXTRIES; iter++) {
        printf("Attempt #%d ||\n ", iter);

        if (BlockNum == 1) {
            AntKernel_1Block <<< 1, threadPerBlock >>> (d_Dist, d_Pheromone, d_Route, d_FoundRoute, size, antRoute, antNum, devStates);
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
                Free_device_memory(d_Dist, d_Pheromone, d_Route, d_FoundRoute, antRoute, d_invalidInput, d_isolatedVertex, d_averageDist, devStates);
                return cudaStatus;
            }

            // Call arguments
            void* kernelArgs[] = { &d_Dist,&d_Pheromone, &d_Route, &d_FoundRoute, &size, &antRoute,&antNum, &devStates,
                &d_invalidInput, &d_isolatedVertex, &d_averageDist };

            cudaLaunchCooperativeKernel((void*)AntKernel_multiBlock, BlockNum, threadPerBlock, kernelArgs);
        }


        // Error handling during Kernel execution
        cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "AntKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
            // Frees GPU device memory
            Free_device_memory(d_Dist, d_Pheromone, d_Route, d_FoundRoute, antRoute, d_invalidInput, d_isolatedVertex, d_averageDist, devStates);
            return cudaStatus;
        }

        // cudaDeviceSynchronize waits for the kernel to finish
        cudaStatus = cudaDeviceSynchronize();
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching antKernel!\n", cudaStatus);
            // Frees GPU device memory
            Free_device_memory(d_Dist, d_Pheromone, d_Route, d_FoundRoute, antRoute, d_invalidInput, d_isolatedVertex, d_averageDist, devStates);
            return cudaStatus;
        }

        // Copying processed data from GPU device
        cudaStatus = cudaMemcpy(h_Route, d_Route, Route_bytes, cudaMemcpyDeviceToHost);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "Route dev->host cudaMemcpy failed!");
            // Frees GPU device memory
            Free_device_memory(d_Dist, d_Pheromone, d_Route, d_FoundRoute, antRoute, d_invalidInput, d_isolatedVertex, d_averageDist, devStates);
            return cudaStatus;
        }
        cudaStatus = cudaMemcpy(h_FoundRoute, d_FoundRoute, sizeof(bool), cudaMemcpyDeviceToHost);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "FoundRoute flag dev->host cudaMemcpy failed!");
            // Frees GPU device memory
            Free_device_memory(d_Dist, d_Pheromone, d_Route, d_FoundRoute, antRoute, d_invalidInput, d_isolatedVertex, d_averageDist, devStates);
            return cudaStatus;
        }
        cudaStatus = cudaMemcpy(h_Pheromone, d_Pheromone, Dist_bytes, cudaMemcpyDeviceToHost);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "Pheromone dev->host cudaMemcpy failed!");
            // Frees GPU device memory
            Free_device_memory(d_Dist, d_Pheromone, d_Route, d_FoundRoute, antRoute, d_invalidInput, d_isolatedVertex, d_averageDist, devStates);
            return cudaStatus;
        }


        if (*h_FoundRoute && cudaStatus == cudaSuccess) {

            double _length = sequencePrint(h_Route, h_Dist, size);
            sum += _length;
            if (_length < min)
                min = _length;
        }
        else
            printf("Route not found!\n\n");
    }

    printf("Average length: %.3f\n", sum / SERIALMAXTRIES);
    printf("Minimal length: %.3f", min);
    // Frees GPU device memory
    Free_device_memory(d_Dist, d_Pheromone, d_Route, d_FoundRoute, antRoute, d_invalidInput, d_isolatedVertex, d_averageDist, devStates);
    return cudaStatus;
};


void Free_device_memory(double* d_Dist,double* d_Pheromone,  int* d_Route, bool* d_FoundRoute, int* antRoute, bool* d_invalidInput, bool* d_isolatedVertex, double* d_averageDist, curandState* devstate) {
    // Tempory device data structures
    if (NULL != d_Dist) cudaFree(d_Dist);
    if (NULL != d_Pheromone) cudaFree(d_Pheromone);
    if (NULL != d_Route) cudaFree(d_Route);
    if (NULL != d_FoundRoute) cudaFree(d_FoundRoute);
    if (NULL != antRoute) cudaFree(antRoute);
    if (NULL != devstate) cudaFree(devstate);

    // Incidental global variables
    if (NULL != d_invalidInput) cudaFree(d_invalidInput);
    if (NULL != d_isolatedVertex) cudaFree(d_isolatedVertex);
    if (NULL != d_averageDist) cudaFree(d_averageDist);
};

__device__ __host__ double sequencePrint(int* Route, double* Dist, size_t size) {
    printf("Sequence : ");
    double l = 0;
    for (int i = 0; i < size; ++i) {
        int src = Route[i];
        int dst = Route[(i + 1) % size];
        printf("%d ", src);
        l += Dist[src * size + dst];
    }
    printf("%d\n", Route[0]);
    printf(" Total length : %.2f\n ", l);
    return l;
}

__device__ double minRes;

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
)
{
    // Dist (i,j) means the distance from vertex i to vertex j
    // If no edge drawn between them: Dist(i,j) = -1 (expected syntax)
    thread_block block = this_thread_block();

    int antIndex = blockIdx.x * blockDim.x + threadIdx.x;  // Ant index 0 - (antNum-1)
    int tr = block.thread_rank();   // elvileg ugyanaz mint az előző sor
                                    // Ott használom, ahol ez az átláthatóságot segíti
    
    if (antIndex >= antNum)     // Itt megtehető ez az egyszerűsítés
        return;                 // Segít elkerülni a túlcimzést

    // Shared variables between threads in the same block
    __shared__ bool invalidInput;   // Variable used to detecting invalid input
    __shared__ bool isolatedVertex; // Variable used to detecting isolated vertex (for optimization purposes)
    __shared__ double averageDist;  // Average edge distance
    
    // Initialization with thread 0
    if (tr == 0) {
        invalidInput = false;
        isolatedVertex = false;
        averageDist = 0.0;
        *FoundRoute = false;
        minRes = DBL_MAX;
    }

    // Managing when size=0 or size=1
    if (size == 0 || size == 1)   // No graph, meaningless input
        return;

    block.sync();

    if (tr == 0) {
        bool foundNeighboor = false;    // Checking if any of the vertexes are isolated
        int ii, jj;
        for (ii = 0; ii < size; ii++) {
            for (jj = 0; jj < size; jj++) {
                // Initializing Pheromone graph (anti - unitmatrix, all main diagonal elements are 0)
                // 0 Pheromone value if no edge drawn
                // Initial Pheromone value is of consideration in the Control panel
                if ( (ii == jj) || (Dist[ii * size + jj] < 0) )
                    Pheromone[ii * size + jj] = 0;
                else
                    Pheromone[ii * size + jj] = INITIAL_PHEROMONE_VALUE;
  
                // Error handling 
                // Check if there are invalid given elements 
                // Valid input if: positive OR -1 OR 0 (only if i=j)
                if (ii != jj && Dist[ii * size + jj] <= 0 && Dist[ii * size + jj] != -1) {
                    printf("Dist(%d,%d) incorrect!\n", ii, jj);
                    invalidInput = true;
                    break;
                }
                if (!foundNeighboor && Dist[ii * size + jj] > 0) {
                    // Has neighboor therefore not isolated
                    foundNeighboor = true;
                }
            }
            if (!foundNeighboor) { // Did not have any neighboors => wrong model of TSP
                printf("Vertex %d isolated!\n", ii);
                isolatedVertex = true;
            }
        }
    }

    block.sync();

    if (invalidInput || isolatedVertex) {   // Ha érvénytelen bemenet volt, akkor nem folytatjuk
        return;                             // Ha van izolált csúcs, akkor nem folytatjuk
    }


    // 2 csúcs esete : lekezeljük gyorsan 1 szálon
    if (size == 2) {
        if (tr == 0) {
            if (Dist[0 * size + 1] > 0 && Dist[1 * size + 0] > 0) {    // Route exists
                *FoundRoute = true;
                Route[0] = 0;
                Route[1] = 1;
            }
        }
        block.sync();
        return;
    }

    // Left: Connected graph with at least 3 vertices

    // Calculating average distance
    if (tr == 0) {
        double sum = 0.0;   // Sum of edge values
        int numPos = 0;     // Number of edges
        for(int i = 0; i < size; i++)
            for (int j = 0; j < size; j++) {
                double edge = Dist[i * size + j];
                if (edge > 0) {
                    sum += edge;
                    numPos++;
                }
            }
        averageDist = sum / numPos * size;
        //printf("Average dist: %f\n", averageDist);
    }
    block.sync();

    // Ants travelling to all directions
    for (int repNumber = 0; repNumber < REPETITIONS; repNumber++) {
        

        // Trying for every possible second vertices
        for (int secondVertex = 1; secondVertex < size; secondVertex++) {
            generateRandomSolution(antRoute, antIndex, secondVertex, Dist, size, state);
            double multiplicationConstant = averageDist / RHO * 5;
            // Evaluating the given solution: modifies Pheromone matrix more if shorter path found
            evaluateSolution(Dist, Pheromone, antRoute, antIndex, size, multiplicationConstant,repNumber);
            block.sync();
        }

        // Numerous random guesses
        for (int j = 0; j < RANDOM_GENERATIONS; j++) {
            // Random second vertices
            generateRandomSolution(antRoute, antIndex, -1, Dist, size, state);
            double multiplicationConstant = averageDist / RHO * (REPETITIONS + 1 - repNumber);
            // Evaluating the given solution: modifies Pheromone matrix more if shorter path found
            evaluateSolution(Dist, Pheromone, antRoute, antIndex, size, multiplicationConstant);
            block.sync();
        }
        //if (tr == 0)
        //    print(Pheromone, size);
        //block.sync();
        

        // Lots of ants following pheromone of previous ants
        for (int gen = 0; gen < FOLLOWER_GENERATIONS; gen++) {

            // Reducing previous pheromon values by value RHO (modifiable in the Control Panel)
            if (tr == 0) {
                for (int ii = 0; ii < size; ii++)
                    for (int jj = 0; jj < size; jj++)
                        Pheromone[ii * size + jj] *= RHO;
            }
            block.sync();

            // new ants following pheromone of previous ants
            followPheromones(Pheromone, antRoute, antIndex, size, state);
            block.sync();
            
            double multiplicationConstant = averageDist / RHO * 10;
            // Evaluating the given solution: modifies Pheromone matrix more if shorter path found
            evaluateSolution(Dist, Pheromone, antRoute, antIndex, size, multiplicationConstant);
            block.sync();
        }
    }

    // After that we only need one ant (thread)
    if (tr != 0)
        return;

    //print(Pheromone, size);
    // Choosing path with greedy algorithm
    greedySequence(Pheromone, Route, size);
    //sequencePrint(antRoute, Dist, size);

    block.sync();
    double __length;
    __length = antRouteLength(Dist, Route, 0, size);
    *FoundRoute = (__length > 0);
}


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
) {
    // Dist (i,j) means the distance from vertex i to vertex j
    // If no edge drawn between them: Dist(i,j) = -1 (expected syntax)
    
    // Synchronization variable across the whole grid
    grid_group grid = this_grid();
    if (!grid.is_valid())
        return;
    grid.sync();
    int antIndex = blockIdx.x * blockDim.x + threadIdx.x;  // ant index
    int tr = grid.thread_rank();
    // Initialization with thread 0
    if (tr == 0) {
        *invalidInput = false;
        *isolatedVertex = false;
        *averageDist = 0.0;
        *FoundRoute = false;
        minRes = DBL_MAX;
    }
    // Managing when size=0 or size=1
    if (size == 0 || size == 1)   // No graph, meaningless input
        return;
    grid.sync();

    if (tr == 0) {
        bool foundNeighboor = false;
        int ii, jj;
        for (ii = 0; ii < size; ii++) {
            for (jj = 0; jj < size; jj++) {
                
                // Initializing Pheromone graph (anti - unitmatrix, all main diagonal elements are 0)
                // 0 Pheromone value if no edge drawn
                // Initial Pheromone value is of consideration in the Control panel
                if ((ii == jj) || (Dist[ii * size + jj] < 0))
                    Pheromone[ii * size + jj] = 0;
                else
                    Pheromone[ii * size + jj] = INITIAL_PHEROMONE_VALUE;

                // Error handling 
                // Check if there are invalid given elements 
                // Valid input if: positive OR -1 OR 0 (only if i=j)
                if (ii != jj && Dist[ii * size + jj] <= 0 && Dist[ii * size + jj] != -1) {
                    printf("Dist(%d,%d) incorrect!\n", ii, jj);
                    *invalidInput = true;
                    break;
                }

                if (!foundNeighboor && Dist[ii * size + jj] > 0) {
                    // Has neighboor therefore not isolated
                    foundNeighboor = true;
                }
            }

            if (!foundNeighboor) { // Did not have any neighboors
                printf("Vertex %d isolated!\n", ii);
                *isolatedVertex = true;
                break;
            }
        }
    }
    grid.sync();

    if (*invalidInput || *isolatedVertex) {   // Stopping if there was an invalid input
        return;                             // Stopping if isolated vertex found
    }

    // When there are only 2 vertices: only one possible route
    if (size == 2) {
        if (tr == 0) {
            if (Dist[0 * size + 1] > 0 && Dist[1 * size + 0] > 0) {    // Route exists
                *FoundRoute = true;
                Route[0] = 0;
                Route[1] = 1;
            }
        }
        grid.sync();
        return;
    }

    // Left: Connected graph with at least 3 vertices

    // Calculating average distance
    if (tr == 0) {
        double sum = 0.0;   // Sum of edge values
        int numPos = 0;     // Number of edges
        for (int i = 0; i < size; i++)
            for (int j = 0; j < size; j++) {
                double edge = Dist[i * size + j];
                if (edge > 0) {
                    sum += edge;
                    numPos++;
                }
            }
        (*averageDist) = sum / numPos * size;
        //printf("Average dist: %f\n", averageDist);
    }
    grid.sync();

    // Ants travelling to all directions
    for (int i = 0; i < REPETITIONS; i++) {
    
        // Trying for every possible second vertices
        for (int j = 1; j < size; j++) {
            generateRandomSolution(antRoute, antIndex, j, Dist, size, state);
            // Evaluating the given solution: modifies Pheromone matrix more if shorter path found
            double multiplicationConstant = (*averageDist) / RHO * 5;
            evaluateSolution(Dist, Pheromone, antRoute, antIndex, size, multiplicationConstant);
            grid.sync();
        }

        // Numerous random guess
        for (int j = 0; j < RANDOM_GENERATIONS; j++) {
            // Random second vertices
            generateRandomSolution(antRoute, antIndex, -1, Dist, size, state);
            // Evaluating the given solution: modifies Pheromone matrix more if shorter path found
            double multiplicationConstant = (*averageDist) / RHO * (REPETITIONS + 1 - i);
            evaluateSolution(Dist, Pheromone, antRoute, antIndex, size, multiplicationConstant);
            grid.sync();
        }

        // Lots of ants following pheromone of previous ants
        for (int gen = 0; gen < FOLLOWER_GENERATIONS; gen++) {

            // Reducing previous pheromon values by value RHO (modifiable in the Control Panel)
            if (tr == 0) {
                for (int ii = 0; ii < size; ii++)
                    for (int jj = 0; jj < size; jj++)
                        Pheromone[ii * size + jj] *= RHO;
            }
            grid.sync();

            // Ants following previous ants
            followPheromones(Pheromone, antRoute, antIndex, size, state);
            grid.sync();
            // Evaluating the given solution: modifies Pheromone matrix more if shorter path found
            double multiplicationConstant = (*averageDist) / RHO * 10;
            evaluateSolution(Dist, Pheromone, antRoute, antIndex, size, multiplicationConstant);
            grid.sync();
        }
    }

    // Only one ant needed in the end
    if (tr != 0)
        return;

    // Choosing path with greedy algorithm
    greedySequence(Pheromone, Route, size);
    //sequencePrint(antRoute, Dist, size);

    double __length;
    __length = antRouteLength(Dist, Route, 0, size);
    *FoundRoute = (__length > 0);
};

// Evaluates the given solution: modifies Pheromone matrix more if shorter path found
__device__ void evaluateSolution(double* Dist, double* Pheromone, int* antRoute, int antIndex, 
    size_t size, double multiplConstant, int repNumber) 
{
    double length = antRouteLength(Dist, antRoute, antIndex, size);
    assert(length != 0);
    double additive = multiplConstant / length; // The longer the route is, the smaller amount we are adding
    if (length < minRes && length > 0) {    // Rewarding the ant with the best yet route
        //printf("New min found: %f\n", length);
        minRes = length;
        additive *= REWARD_MULTIPLIER * (repNumber + 1) * (repNumber + 1);
    }
    
    // Route valid if length > 0
    if (length > 0) {

        for (int jj = 0; jj < size; jj++) {
            int source = antRoute[antIndex * size + jj];
            int dest = antRoute[antIndex * size + (jj + 1) % size];
            double* ptr = &(Pheromone[source * size + dest]);

            atomicAdd(ptr, additive);
        }
    }
};

// Inicializes a random seed for different threads
__global__ void setup_kernel(curandState* state, unsigned long seed)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    curand_init(seed, id, id, &state[id]);
}



// Generates a random sequence of numbers between 0 and (size - 1) starting with 0
// secondVertex: Variable used for giving an arbitrary second vertex
//      0 < secondvertex < size : valid input (condition = 1)
//      else: invalid input, no mandatory second vertex (condition = 0) 
__device__ void generateRandomSolution(int* antRoute, unsigned int antIndex, int secondVertex, double* Dist, size_t size, curandState* state) {
    // Expected to start in vertex 0
    for (int idx = 0; idx < size; idx++) {    // Array: [0, 1, 2 ... size-1]
        antRoute[antIndex * size + idx] = idx;
    }
    int min_rand_int, max_rand_int = size - 1;
    bool condition = (secondVertex > 0 && secondVertex < size);
    min_rand_int = condition ? 2 : 1;
    // If there's valid input for second vertex
    if (condition) {    // Swap with vertex 1
        antRoute[antIndex * size + 1] = secondVertex;
        antRoute[antIndex * size + secondVertex] = 1;
    }

    // n db random swap in the sequence, to shuffle the edges
    // executing [size] times random swaps
    // min_rand_int means the lower limit for the swap range
    // -> if there is an exact 2.vertex, then only the (3. - size.) vertex sequence needs to be changed
    for (int idx = min_rand_int; idx < size; idx++) {
        float myrandf;
        int myrand;

        myrandf = curand_uniform(&state[antIndex]);  // RND Number between 0 and 1
        myrandf *= (max_rand_int - min_rand_int + 0.999999);
        myrandf += min_rand_int;
        myrand = (int)truncf(myrandf);

        assert(myrand <= max_rand_int);
        assert(myrand >= min_rand_int);

        int temp = antRoute[antIndex * size + idx];
        antRoute[antIndex * size + idx] = antRoute[antIndex * size + myrand];
        antRoute[antIndex * size + myrand] = temp;
    }
    //if (antIndex == 0) {
    //    printf("Generated random sequence: ");
    //    sequencePrint(antRoute, Dist, size);
    //}
}

// Auxiliary function for generating random sequence
__device__ bool alreadyListed(int* antRoute, int antIndex, size_t size, int idx, int newParam) {
    if (idx >= size)
        return true;    // Rather make infinite cicle than overaddressing
    for (int i = 0; i < idx; ++i)
        if (newParam == antRoute[antIndex * size + i])
            return true;
    return false;
}

// Returns the length of the given route
// Returns -1 if route has dead end
__device__ double antRouteLength(double* Dist, int* antRoute, int antIndex, size_t size) {
    double length = 0;  // Return value
    int src, dst;

    for (int i = 0; i < size; ++i) {
        src = antRoute[antIndex * size + i];
        dst = antRoute[antIndex * size + (i + 1) % size];   // Next vertex

        double x = Dist[src * size + dst];  // Processing Dist(i,j) 
        if (x < 0)  // No edge between them
            return -1;
        else        // Edge between them, adding its value
            length += x;
    }
    return length;
}


// Represents az ant who follows other ants' pheromones
// Generates a route with Roulette wheel method given the values of the Pheromone matrix
__device__ void followPheromones(const double* Pheromone, int* antRoute, int antIndex, size_t size, curandState* state) {
    curandState* statePtr = &(state[antIndex]);
    // Expected to start in vertex 0
    antRoute[antIndex * size + 0] = 0;

    double sumPheromone = 0.0;  // Weighted Roulette wheel: firstly calculating the sum of weights
    for (int i = 0; i < size; i++)
        sumPheromone += Pheromone[i];

    for (int i = 1; i < size; i++) {
        int source = antRoute[antIndex * size + i - 1];
        int newParam, maxTryNumber = 4 * size;
        bool foundVertexByRoulette = false;
        for (int j = 0; j < maxTryNumber && !foundVertexByRoulette; j++) {
            // RND Number between 0 and sumPheromone
            
            double myranddbl = curand_uniform_double(statePtr) * sumPheromone;
            double temp = Pheromone[source * size + 0]; // Used to store the matrix values

            for (newParam = 0; newParam < size - 1; newParam++) {   // If newparam == size-1 then no other vertex to choose
                if (myranddbl < temp)
                    break;
                temp += Pheromone[source * size + newParam + 1];
            }   // If not already listed then adding to the sequence
            foundVertexByRoulette = !alreadyListed(antRoute, antIndex, size, i, newParam);
      
        }
        if (!foundVertexByRoulette) {
            // Next vertex choosen by equal chances
            do {
                float newfloat = curand_uniform(statePtr);  // RND Number between 0 and 1
                newfloat *= (size - 1) + 0.999999;  // Transforming into the needed range
                newParam = (int)truncf(newfloat);
            } while (alreadyListed(antRoute, antIndex, size, i, newParam));
        }
        // At last the new vertex
        antRoute[antIndex * size + i] = newParam;
    }
}


// Auxilary function for greedy sequence
// Return the highest vertex index not yet chosen
__device__ int maxInIdxRow(const double* Pheromone, int row, size_t size, int idx, int* antRoute) {
    int maxidx = -1;
    double max = 0;

    for (int i = 0; i < size; i++) {
        double observed = Pheromone[row * size + i];
        if (observed > max && !alreadyListed(antRoute, 0, size, idx, i)) {
            max = observed;
            maxidx = i;
        }
    }
    //printf("%d. vertex with value of %.2f : %d\n", idx, max, maxidx);

    return maxidx;
}

// Generates a sequnce using greedy algorithm
// Always chooses the highest possible value for the next vertex
__device__ void greedySequence(const double* Pheromone, int* antRoute, size_t size) {
    antRoute[0] = 0;
    for (int i = 1; i < size; i++) {
        antRoute[i] = maxInIdxRow(Pheromone, antRoute[i - 1], size, i, antRoute);
    }
}