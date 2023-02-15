
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

    printf("Average distance between vertexes: %.2f\n", sum / numpos); //////

    // Host variables
    double* Pheromone = (double*)malloc(size * size * sizeof(double));
    if (Pheromone == NULL)
        goto End;
    // Sorrend vektor: Route[0] = 0, azaz a 0. városban kezdünk 
    // (önkényesen kijelölt kezdő város)
    int* Route = (int*)malloc(size * sizeof(int));
    if (Route == NULL)
        goto End;
    bool found;
    printf("Travelling Salesman problem with Ant Colony Algorithm\n");
    cudaError_t CUDAstate = AntCUDA(Dist, Route, Pheromone, &found, ants, size);

End:
    free(Pheromone);
    free(Route);

    return 0;
}


// Main CUDA function
cudaError_t AntCUDA(double* h_Dist, int* h_Route, double* h_Pheromone, bool* h_FoundRoute, unsigned int antNum, size_t size) {

    // size = 0,1 : invalid inputs
    if (size == 0 || size == 1 || antNum == 0 || antNum == 1) {
        fprintf(stderr, "Incorrect size or antNum! Must be at least 2.");
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
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        return cudaStatus;
    }

    // Device pointers
    double* d_Dist;
    bool* d_FoundRoute;

    double* d_Pheromone;
    int* antRoute, * d_Route;

    // We need global variables when executing numerous grid blocks
    // Alternate solution: giving status flag as function parameters
    
    bool* d_invalidInput;   // Variable used to detecting invalid input
    bool* d_isolatedVertex;  // Variable used to detecting isolated vertex (for optimization purposes)
    double* d_averageDist;
    // Adathalmaz mérete, amit lefoglalunk
    size_t Dist_bytes = size * size * sizeof(double);
    size_t Route_bytes = size * sizeof(int);
    size_t FoundRoute_bytes = sizeof(bool); // Csak az átláthatóság érdekében...

    size_t antRoute_bytes = antNum * size * sizeof(int);

    // Adatfoglalás

    // Dist
    cudaStatus = cudaMalloc((void**)&d_Dist, Dist_bytes);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "d_Dist cudaMalloc failed!");
        goto Error;
    }
    // Pheromone
    cudaStatus = cudaMalloc((void**)&d_Pheromone, Dist_bytes);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "d_Pheromone cudaMalloc failed!");
        goto Error;
    }
    // Route
    cudaStatus = cudaMalloc((void**)&d_Route, Route_bytes);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "d_Route cudaMalloc failed!");
        goto Error;
    }
    // FoundRoute : flag
    cudaStatus = cudaMalloc((void**)&d_FoundRoute, FoundRoute_bytes);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "d_Route cudaMalloc failed!");
        goto Error;
    }
    // antRoute : segédtömb
    cudaStatus = cudaMalloc((void**)&antRoute, antRoute_bytes);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "antRoute cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&d_invalidInput, sizeof(bool));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }
    cudaStatus = cudaMalloc((void**)&d_isolatedVertex, sizeof(bool));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }
    cudaStatus = cudaMalloc((void**)&d_averageDist, sizeof(double));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Adatok másolása : Host -> Device
    cudaStatus = cudaMemcpy(d_Dist, h_Dist, Dist_bytes, cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "Dist cudaMemcpy failed!");
        goto Error;
    }

    h_Route[0] = 0; // Route[0] = 0, azaz a 0. városban kezdünk (önkényesen kijelölhető kezdő város)
    cudaStatus = cudaMemcpy(d_Route, h_Route, Route_bytes, cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "Route cudaMemcpy failed!");
        goto Error;
    }


    printf("Called function %d Block:  \n", BlockNum);
    int threadPerBlock = (antNum > BLOCK_SIZE) ? BLOCK_SIZE : antNum;
    curandState* devStates;
    cudaMalloc(&devStates, antNum * sizeof(curandState));

    // setup seeds

    setup_kernel << < BlockNum, threadPerBlock >> > (devStates, time(NULL) * rand());

    // Kernel hívása

    double min = 10000000.0;
    double sum = 0.0;

    for (int iter = 0; iter < SERIALMAXTRIES; iter++) {
        printf("Attempt #%d ||\n ", iter);

        if (BlockNum == 1) {
            AntKernel_1Block << < 1, threadPerBlock >> > (d_Dist, d_Pheromone, d_Route, d_FoundRoute, size, antRoute, antNum, devStates);
        }
        else {
            // Kernel hívás esetén fontos, hogy <<<...>>> syntax helyett
            // a cudaLaunchCooperativeKernel CUDA runtime launch API-t kell használni
            // vagy annak CUDA driver megfelelőjét

            // 1-be állítja a supportsCoopLaunch-t ha a művelet támogatott a device 0-n.
            // Csak compute capability 6.0 felett
            int dev = 0;
            int supportsCoopLaunch = 0;
            cudaDeviceGetAttribute(&supportsCoopLaunch, cudaDevAttrCooperativeLaunch, dev);
            if (supportsCoopLaunch != 1) {
                fprintf(stderr, "Cooperative Launch is not supported on this machine configuration.");
            }




            // Hívás
            void* kernelArgs[] = { &d_Dist,&d_Pheromone, &d_Route, &d_FoundRoute, &size, &antRoute,&antNum, &devStates,
                &d_invalidInput, &d_isolatedVertex, &d_averageDist };

            cudaLaunchCooperativeKernel((void*)AntKernel_multiBlock, BlockNum, threadPerBlock, kernelArgs);


        }


        // Hibakeresés kernel lauch közben
        cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "AntKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
            // Felszabdítjuk a lefoglalt GPU memoriát
            Free_device_memory(d_Dist, d_Pheromone, d_Route, d_FoundRoute, antRoute, d_invalidInput, d_isolatedVertex, d_averageDist);
        }

        // cudaDeviceSynchronize vár arra, hogy befejeződjön a kernel, utána visszatér
        cudaStatus = cudaDeviceSynchronize();
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching antKernel!\n", cudaStatus);
            // Felszabdítjuk a lefoglalt GPU memoriát
            Free_device_memory(d_Dist, d_Pheromone, d_Route, d_FoundRoute, antRoute, d_invalidInput, d_isolatedVertex, d_averageDist);
        }

        // Feldolgozott adat átvitele a GPU-ról
        cudaStatus = cudaMemcpy(h_Route, d_Route, Route_bytes, cudaMemcpyDeviceToHost);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "Route dev->host cudaMemcpy failed!");
            // Felszabdítjuk a lefoglalt GPU memoriát
            Free_device_memory(d_Dist, d_Pheromone, d_Route, d_FoundRoute, antRoute, d_invalidInput, d_isolatedVertex, d_averageDist);
        }
        cudaStatus = cudaMemcpy(h_FoundRoute, d_FoundRoute, sizeof(bool), cudaMemcpyDeviceToHost);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "FoundRoute flag dev->host cudaMemcpy failed!");
            // Felszabdítjuk a lefoglalt GPU memoriát
            Free_device_memory(d_Dist, d_Pheromone, d_Route, d_FoundRoute, antRoute, d_invalidInput, d_isolatedVertex, d_averageDist);
        }
        cudaStatus = cudaMemcpy(h_Pheromone, d_Pheromone, Dist_bytes, cudaMemcpyDeviceToHost);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "Pheromone dev->host cudaMemcpy failed!");
            // Felszabdítjuk a lefoglalt GPU memoriát
            Free_device_memory(d_Dist, d_Pheromone, d_Route, d_FoundRoute, antRoute, d_invalidInput, d_isolatedVertex, d_averageDist);
        }


        if (*h_FoundRoute && cudaStatus == cudaSuccess) {

            double _length = sequencePrint(h_Route, h_Dist, size);
            sum += _length;
            if (_length < min)
                min = _length;
            //printf("Length : %.3f\n\n", _length);
        }
        else
            printf("Route not found!\n\n");
    }

    printf("Average length: %.3f\n", sum / SERIALMAXTRIES);
    printf("Minimal length: %.3f", min);
    // Felszabdítjuk a lefoglalt GPU memoriát
Error:
    Free_device_memory(d_Dist, d_Pheromone, d_Route, d_FoundRoute, antRoute, d_invalidInput, d_isolatedVertex, d_averageDist);
    return cudaStatus;
};

void Free_device_memory(double* d_Dist,double* d_Pheromone,  int* d_Route, bool* d_FoundRoute, int* antRoute, bool* d_invalidInput, bool* d_isolatedVertex, double* d_averageDist) {
    // Ideiglenes adattárolók felszabadítása
    cudaFree(d_Dist);
    cudaFree(d_Pheromone);
    cudaFree(d_Route);
    cudaFree(d_FoundRoute);
    cudaFree(antRoute);

    // Esetleges globális változók
    cudaFree(d_invalidInput);
    cudaFree(d_isolatedVertex);
    cudaFree(d_averageDist);
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

// 1 blokkos kernel függvény
__global__ void AntKernel_1Block(
    double* Dist,       // Költségfüggvény input
    double* Pheromone,
    int* Route,         // Sorrend output
    bool* FoundRoute,   // Létezés output
    size_t size,        // Gráf csúcsainak száma
    int* antRoute,      // Segédtömb
    int antNum,         // Hangyák száma
    curandState* state
)
{
    // Dist (a,b) eleme a. csúcsból a b. csúcs távolsága
    // Ha nincs köztük él akkor Dist(i,j) = -1 (elvárt szintaktika)
    thread_block block = this_thread_block();

    int antIndex = blockIdx.x * blockDim.x + threadIdx.x;  // Hangya index 0 - (antNum-1)
    int tr = block.thread_rank();   // elvileg ugyanaz mint az előző sor
                                    // Ott használom, ahol ez az átláthatóságot segíti
    
    if (antIndex >= antNum)     // Itt megtehető ez az egyszerűsítés
        return;                 // Segít elkerülni a túlcimzést

    // Megosztott változók
    __shared__ bool invalidInput;   // Hibás bemenetet időben kell észlelni
    __shared__ bool isolatedVertex; // Ha van izolált csúcs, akkor nincs bejárás (egyszerű teszt)
    __shared__ double averageDist;  // Átlagos élhossz
    
    // Kezdeti érték adása a 0. thread által
    if (tr == 0) {
        invalidInput = false;
        isolatedVertex = false;
        averageDist = 0.0;
        *FoundRoute = false;
        minRes = DBL_MAX;
    }

    // Kezeljük a size=0 vagy 1 esetet
    if (size == 0 || size == 1)   // Nincs gráf, értelmetlen bemenet
        return;

    block.sync();

    // Hibatesztelések 1 szálon
    if (tr == 0) {
        bool foundNeighboor = false;    // Minden csúcs izolált voltát ellenörizzük (ha van izolált, nincs bejárás)
        int ii, jj;
        for (ii = 0; ii < size; ii++) {
            for (jj = 0; jj < size; jj++) {
                // Pheromone gráf kezdeti értéke (anti-egységmátrix, csak főátlóban 0)
                // Ahol nem vezet él, szintén 0 feromon kerül
                // Vizsgálat tárgya a megfelelő kezdeti feromon érték
                if ( (ii == jj) || (Dist[ii * size + jj] < 0) )
                    Pheromone[ii * size + jj] = 0;
                else
                    Pheromone[ii * size + jj] = INITIAL_PHEROMONE_VALUE;
 
                // Megvizsgáljuk, hogy van-e érvénytelenül megadott gráfél 
                // (nem pozitív, vagy -1, vagy 0 ha i=j)
                if (ii != jj && Dist[ii * size + jj] <= 0 && Dist[ii * size + jj] != -1) {
                    printf("Dist(%d,%d) incorrect!\n", ii, jj);
                    invalidInput = true;
                    break;
                }
                if (Dist[ii * size + jj] > 0) {
                    // Van szomszédja, tehát nem izolált csúcs
                    foundNeighboor = true;
                }
            }
            if (!foundNeighboor) { // Nem volt szomszédja
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
            if (Dist[0 * size + 1] > 0 && Dist[1 * size + 0] > 0) {    // Van kör
                *FoundRoute = true;
                Route[0] = 0;
                Route[1] = 1;
            }
        }
        block.sync();
        return;
    }

    // Maradt: Helyes bemenet, legalább 3 csúcs, mindegyik mutat valahova

    // Számoljuk ki az átlagos úthosszat
    if (tr == 0) {
        double sum = 0.0; /////
        int numPos = 0;
        for(int i=0;i<size;i++)
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

    // Hangyák indulnak mindenfelé
    for (int repNumber = 0; repNumber < REPETITIONS; repNumber++) {
        

        // Megpróbálunk legalább egyszer minden csúcsot 2. csúcsnak választani (ez még nem rontja el a polinomidejüséget mert O(n) )
        for (int j = 1; j < size; j++) {
            generateRandomSolution(antRoute, antIndex, j, Dist, size, state);
            double multiplicationConstant = averageDist / ALPHA * 5;
            evaluateSolution(Dist, Pheromone, antRoute, antIndex, size, multiplicationConstant,repNumber);
            block.sync();
        }

        // Néhány teljesen random útra induló hangya
        for (int j = 0; j < RANDOM_GENERATIONS; j++) {
            // Random 2. csúcsok
            generateRandomSolution(antRoute, antIndex, -1, Dist, size, state);
            double multiplicationConstant = averageDist / ALPHA * (REPETITIONS + 1 - repNumber);
            evaluateSolution(Dist, Pheromone, antRoute, antIndex, size, multiplicationConstant);
            block.sync();
        }
        //if (tr == 0)
        //    print(Pheromone, size);
        //block.sync();
        

        // Sok korábbit követö hangya
        for (int gen = 0; gen < FOLLOWER_GENERATIONS; gen++) {

            // Előző ciklus feromonjainak mérséklése
            if (tr == 0) {
                for (int ii = 0; ii < size; ii++)
                    for (int jj = 0; jj < size; jj++)
                        Pheromone[ii * size + jj] *= ALPHA;
            }
            block.sync();

            // Új hangyák indulnak a feromonok után
            followPheromones(Pheromone, antRoute, antIndex, size, state);
            block.sync();
            // Kiértékeljük a kapott utat
            double multiplicationConstant = averageDist / ALPHA * 10;
            evaluateSolution(Dist, Pheromone, antRoute, antIndex, size, multiplicationConstant);
            block.sync();
        }
    }

    // Küldjük el a feromon úton a végső hangyát
    if (tr != 0)
        return;



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
    }
    // Managing when size=0 or size=1
    if (size == 0 || size == 1)   // No graph, meaningless input
        return;
    grid.sync();

    if (tr == 0) {
        bool foundNeighboor;
        int ii, jj;
        for (ii = 0; ii < size; ii++) {
            foundNeighboor = false;
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

            if (!foundNeighboor) { // Nem volt szomszédja
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

    // Maradt: Helyes bemenet, legalább 3 csúcs, mindegyik mutat valahova
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
            double multiplicationConstant = (*averageDist) / ALPHA * 5;
            evaluateSolution(Dist, Pheromone, antRoute, antIndex, size, multiplicationConstant);
            grid.sync();
        }

        // Numerous random guess
        for (int j = 0; j < RANDOM_GENERATIONS; j++) {
            // Random second vertices
            generateRandomSolution(antRoute, antIndex, -1, Dist, size, state);
            // Evaluating the given solution: modifies Pheromone matrix more if shorter path found
            double multiplicationConstant = (*averageDist) / ALPHA * (REPETITIONS + 1 - i);
            evaluateSolution(Dist, Pheromone, antRoute, antIndex, size, multiplicationConstant);
            grid.sync();
        }

        // Lots of ants following pheromoe of previous ants
        for (int gen = 0; gen < FOLLOWER_GENERATIONS; gen++) {

            // Reducting previous pheromon values by value ALPHA (modifiable in the Control Panel)
            if (tr == 0) {
                for (int ii = 0; ii < size; ii++)
                    for (int jj = 0; jj < size; jj++)
                        Pheromone[ii * size + jj] *= ALPHA;
            }
            grid.sync();

            // Ants following previous ants
            followPheromones(Pheromone, antRoute, antIndex, size, state);
            grid.sync();
            // Evaluating the given solution: modifies Pheromone matrix more if shorter path found
            double multiplicationConstant = (*averageDist) / ALPHA * 10;
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
__device__ void evaluateSolution(double* Dist, double* Pheromone, int* antRoute, int antIndex, size_t size, double multiplConstant, int repNumber) {
    double length = antRouteLength(Dist, antRoute, antIndex, size);
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
    int id = blockIdx.x * blockDim.x + threadIdx.x;;
    curand_init(seed, id, id, &state[id]);
}



// Generál egy random, de a 0. csúccsal kezdödö sorrendet
// secondVertex: ha negatív, vagy nagyobb mint size, akkor nincs önkényes 2. csúcs (condition=0), különben van (condition=1) 
__device__ void generateRandomSolution(int* antRoute, unsigned int antIndex, int secondVertex, double* Dist, size_t size, curandState* state) {
    // Azt elvárom, hogy a 0. csúcsban kezdjen
    for (int idx = 0; idx < size; idx++) {    // Tömb [0, 1, 2 ... size-1]
        antRoute[antIndex * size + idx] = idx;
    }
    int min_rand_int, max_rand_int = size - 1;
    bool condition = (secondVertex > 0 && secondVertex < size);
    min_rand_int = condition ? 2 : 1;
    // Ha van érvényes megadás a második csúcsra
    if (condition) {    // Helyet cserél az 1-es indexü (valójában a 2.) elem a secondVertex indexüvel
        antRoute[antIndex * size + 1] = secondVertex;
        antRoute[antIndex * size + secondVertex] = 1;
    }

    // n db random cserét hajtunk végre a sorrendben, ezzel megkeverve a csúcsokat
    // min_rand_int jelöli ki a keverendö tartomány alsó határát
    // -> ha van elöirt 2. csúcs, akkor csak a (3. - size.) csúcsok sorrendjét kell megváltoztatni
    for (int idx = min_rand_int; idx < size; idx++) {
        float myrandf;
        int myrand;

        myrandf = curand_uniform(&state[antIndex]);  // 0 és 1 közötti számot ad
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
    //    printf("Generalt random sorrend: ");
    //    sequencePrint(antRoute, Dist, size);
    //
    //}
}

// Sorrend generáláshoz használt függvény
__device__ bool alreadyListed(int* antRoute, int antIndex, size_t size, int idx, int newParam) {
    if (idx >= size)
        return true;    // Inkább okozzunk végtelen ciklust, mint túlcímzést
    for (int i = 0; i < idx; ++i)
        if (newParam == antRoute[antIndex * size + i])
            return true;
    return false;
}

// Megadja egy adott bejárás hosszát
// (-1) -gyel tér vissza ha az adott körút nem bejárható
__device__ double antRouteLength(double* Dist, int* antRoute, int antIndex, size_t size) {
    double length = 0;  // Visszatérési érték
    int source, dest;

    for (int i = 0; i < size; ++i) {
        source = antRoute[antIndex * size + i];
        dest = antRoute[antIndex * size + (i + 1) % size];   // Következő csúcs

        double x = Dist[source * size + dest];  // Dist(i,j) feldolgozása
        if (x < 0)  // Nem fut közöttük él
            return -1;
        else        // Fut közöttük él, hozzáadjuk a hosszát
            length += x;
    }
    return length;
}


// A feromon mátrix értékei alapján rulettkerék módszerrel választ következö csúcsot
__device__ void followPheromones(const double* Pheromone, int* antRoute, int antIndex, size_t size, curandState* state) {
    // Azt elvárom, hogy a 0. csúcsban kezdjen
    antRoute[antIndex * size + 0] = 0;

    double sumPheromone = 0.0;  // Súlyozott Rulettkerék: elöször kiszámoljuk a sorösszeget
    for (int i = 0; i < size; i++)
        sumPheromone += Pheromone[i];

    for (int i = 1; i < size; i++) {
        int source = antRoute[antIndex * size + i - 1];
        int newparam, maxTryNumber = 4 * size;
        bool foundVertexByRoulette;
        for (int j = 0; j < maxTryNumber; j++) {
            // 0 és sumPheromone közötti random szám generálása
            curandState* statePtr = &(state[antIndex]);
            double myranddbl = curand_uniform_double(statePtr) * sumPheromone;
            double temp = Pheromone[source * size + 0]; // Ebben fogjuk összeadni a mátrix értékeket

            for (newparam = 0; newparam < size - 1; newparam++) {   // Ha newparam = size-1 akkor már nincs hova nagyobbat keresni
                if (myranddbl < temp)
                    break;
                temp += Pheromone[source * size + newparam + 1];
            }   // Ha még nincs a listában akkor beírjuk
            foundVertexByRoulette = !alreadyListed(antRoute, antIndex, size, i, newparam);

            //if (antIndex == 0 && source == 0)
               //printf("Ant%d Source: %d \t rnd=%.2f\t max=%.2f, chosen %d\n", antIndex, source, myranddbl, sumPheromone, newparam);

            if (foundVertexByRoulette)
                break;
        }
        if (!foundVertexByRoulette) {
            // A következo csucs egyenlo eselyek alapjan
            do {
                float newfloat = curand_uniform(&state[antIndex]);  // 0 és 1 közötti számot ad
                newfloat *= (size - 1) + 0.999999;  // A kívánt tartományba transzformáljuk
                newparam = (int)truncf(newfloat);
            } while (alreadyListed(antRoute, antIndex, size, i, newparam));
        }
        // Végül a kapott új csúcs
        antRoute[antIndex * size + i] = newparam;
    }
}


/// 
/// Megadja, hogy az adott sorban melyik a legnagyobb még nem választott feromon
/// Mohó algoritmus használja sorrend kialakításához
/// param name="Pheromone" : Ez a mátrix tartalmazza a feromon értékeket
/// <param name="row"></param>
/// <param name="size"></param>
/// <param name="idx"></param>
/// <param name="antRoute"></param>
/// <returns></returns>
__device__ int maxInIdxRow(const double* Pheromone, int row, size_t size, int idx, int* antRoute) {
    int maxidx = idx;
    double max = Pheromone[row * size + idx];

    for (int i = 0; i < size; i++) {
        double vizsgalt = Pheromone[row * size + i];
        if (vizsgalt > max && !alreadyListed(antRoute, 0, size, idx, i)) {
            max = vizsgalt;
            maxidx = i;
        }
    }
    //printf("%d. elem %.2f ertekkel: %d\n", idx, max, maxidx);

    return maxidx;
}

// Mohó algoritmus alapján választ utat egy megadott Pheromone gráf szerint:
// Mindig a még be nem járt csúcsok közül arra megy, ahol a legmagasabb a feromon értéke
__device__ void greedySequence(const double* Pheromone, int* antRoute, size_t size) {
    antRoute[0] = 0;
    for (int i = 1; i < size; i++) {
        antRoute[i] = maxInIdxRow(Pheromone, antRoute[i - 1], size, i, antRoute);
    }
}