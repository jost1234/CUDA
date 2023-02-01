
// Speciális CUDA headerek
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cooperative_groups.h>
#include "curand.h"
#include "curand_kernel.h"

// Saját headerek
#include "Header.h"

// Általános célú headerek
#include <stdio.h>
#include <stdbool.h>
#include <stdlib.h>
#include <assert.h>

using namespace cooperative_groups;


// Main függvény
int main(int argc, char* argv[])
{
    FILE* Dist_file;
    int Dist_filename_Idx;
    bool foundDistMatrix;
    size_t size;
    // Távolság mátrix
    int i;
    srand(time(0));
    for (i = 1; i < argc; ++i)
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

    // File szintaktika : 1. sorban a méret
    // Utána soronkánt az értékek vesszovel elvalasztva
    if (fscanf_s(Dist_file, "%zu \n", &size) == 0) {
        fprintf(stderr, "Unable to read size! Make sure you have the right file syntax!\n");
        fclose(Dist_file);
        return -1;
    }
    double* Dist = (double*)calloc(size * size, sizeof(double));
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
    printf("Given Dist matrix:\n");
    print(Dist, size);

    printf("Average distance between vertexes: %.2f\n", sum / numpos); //////

    double* Pheromone = (double*)malloc(size * size * sizeof(double));
    if (Pheromone == NULL)
        goto End;
    // Sorrend vektor: Route[0] = 0, azaz a 0. városban kezdünk 
    // (önkényesen kijelölt kezdő város)
    int* Route = (int*)malloc(size * sizeof(int));
    if (Route == NULL)
        goto End;
    bool found;
    printf("Alternating Salesman problem with Ant Colony Algorithm\n");
    cudaError_t CUDAstate = AntCUDA(Dist, Route, Pheromone, &found,ants, size);

End:
    free(Pheromone);
    free(Route);

    return 0;
}


// Meghívandó CUDA függvény
cudaError_t AntCUDA(double* h_Dist, int* h_Route, double* h_Pheromone, bool* h_FoundRoute,unsigned int antNum, size_t size) {

    // size = 0,1 : érvénytelen bemenetek
    if (size == 0 || size == 1 || antNum == 0 || antNum == 1) {
        fprintf(stderr, "Incorrect size or antNum! Must be at least 2.");
        return cudaError_t::cudaErrorInvalidConfiguration;
    }

    // Kiszámoljuk, hogy hány blokkot kell futtatni
    // Minden hangya külön blokkot kap
    int BlockNum = 1;
    if (antNum > BLOCK_SIZE) {
        BlockNum = my_ceil(antNum, BLOCK_SIZE);
        antNum = BlockNum * BLOCK_SIZE; // Hogy kihasználjuk jobban a párhuzamos threadeket
    }

    cudaError_t cudaStatus;
    // Kiválasztjuk a GPU-t, multi-GPU rendszerben lényeges lehet.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        return cudaStatus;
    }

    // Device pointerek
    double* d_Dist;
    bool* d_FoundRoute;

    double* d_Pheromone;
    int* antRoute, * d_Route;

    // Globális belső változók szükségesek ha több block van
    bool* d_invalidInput;   // Hibás bemenetet időben kell észlelni
    bool* d_isolatedVertex;  // Ha van izolált csúcs, akkor nincs bejárás (egyszerű teszt)
    double* d_bestFit;
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
        fprintf(stderr, "antRoute cudaMalloc failed!");
        goto Error;
    }
    cudaStatus = cudaMalloc((void**)&d_bestFit, sizeof(double));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "antRoute cudaMalloc failed!");
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

    setup_kernel <<< BlockNum, threadPerBlock >>> (devStates, time(NULL) * rand());

    // Kernel hívása

    double min = 10000000.0;
    double sum = 0.0;

    for (int iter = 0; iter < SERIALMAXTRIES; iter++) {
        printf("Attempt #%d ||\n ", iter);

        if (BlockNum == 1) {
            AntKernel_1Block <<< 1, threadPerBlock >>> (d_Dist, d_Pheromone, d_Route, d_FoundRoute, size, antRoute,antNum, devStates);
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
                &d_invalidInput, &d_isolatedVertex, &d_bestFit};
            
            cudaLaunchCooperativeKernel((void*)AntKernel_multiBlock, BlockNum, threadPerBlock, kernelArgs);


        }


        // Hibakeresés kernel lauch közben
        cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "AntKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
            goto Error;
        }

        // cudaDeviceSynchronize vár arra, hogy befejeződjön a kernel, utána visszatér
        cudaStatus = cudaDeviceSynchronize();
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching antKernel!\n", cudaStatus);
            goto Error;
        }

        // Feldolgozott adat átvitele a GPU-ról
        cudaStatus = cudaMemcpy(h_Route, d_Route, Route_bytes, cudaMemcpyDeviceToHost);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "Route dev->host cudaMemcpy failed!");
            goto Error;
        }
        cudaStatus = cudaMemcpy(h_FoundRoute, d_FoundRoute, sizeof(bool), cudaMemcpyDeviceToHost);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "FoundRoute flag dev->host cudaMemcpy failed!");
            goto Error;
        }
        cudaStatus = cudaMemcpy(h_Pheromone, d_Pheromone, Dist_bytes, cudaMemcpyDeviceToHost);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "Pheromone dev->host cudaMemcpy failed!");
            goto Error;
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
    // Hiba esetén ide lépünk egyből
    // Normál működés során is ide kell érjünk
Error:
    // Ideiglenes adattárolók felszabadítása
    cudaFree(d_Dist);
    cudaFree(d_Pheromone);
    cudaFree(d_Route);
    cudaFree(d_FoundRoute);
    cudaFree(antRoute);

    // Esetleges globális változók
    cudaFree(d_invalidInput);
    cudaFree(d_isolatedVertex);
    cudaFree(d_bestFit);
    return cudaStatus;
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
    int tr = block.thread_rank();
    if (antIndex >= antNum)  // Itt megtehető ez az egyszerűsítés
        return;

    // Megosztott változók
    __shared__ bool invalidInput;   // Hibás bemenetet időben kell észlelni
    __shared__ bool isolatedVertex; // Ha van izolált csúcs, akkor nincs bejárás (egyszerű teszt)
    __shared__ double bestFit;

    // Kezdeti érték adása a 0. thread által
    if (tr == 0) {
        invalidInput = false;
        isolatedVertex = false;
        bestFit = 0.0;
        *FoundRoute = false;
    }

    // Kezeljük a size=0 vagy 1 esetet
    if (size == 0 || size == 1)   // Nincs gráf, értelmetlen bemenet
        return;

    block.sync();

    if (tr == 0) {
        bool foundNeighboor = false;
        int ii, jj;
        for (ii = 0; ii < size; ii++) {
            for (jj = 0; jj < size; jj++) {
                // Pheromone gráf kezdeti értéke (anti-egységmátrix, csak főátlóban 0)
                if (ii == jj)
                    Pheromone[ii * size + jj] = 0;
                else
                    Pheromone[ii * size + jj] = INITIAL_PHEROMONE_VALUE;

                // Hibakeresés a gráfban N szálon 
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

    // Számoljuk ki az ideális (nem mindig elérhető) alsó becslést a bejárás hosszára
    // Minden csúcshoz a hozzá legközelebb állót vesszük hozzá
    if (tr == 0) {
        for (int ii = 0; ii < size; ii++) {
            double minAdditive = minDist(Dist, ii, size);
            if (minAdditive > 0) // Csak ha van legalább 1 szomszédja: kell legyen
                bestFit += minAdditive;
        }
    }
    
    block.sync();

    
    // Választunk egy random utat a hangyának (az sem biztos hogy lehetséges az út)
    //generateRandomSolution(antRoute, antIndex,-1, Dist, size, state);
    //block.sync();

    double length, additive;
    for (int iii = 0; iii < 10; iii++) {
        // Választunk egy random utat a hangyának (az sem biztos hogy lehetséges az út)
        for (int ii = 0; ii < (RANDOM_GENERATIONS-50*iii) || ii < size; ii++) {
            if (ii < size) // Mindenféle első csúcs legyen legalább egyszer
                generateRandomSolution(antRoute, antIndex, ii, Dist, size, state);
            else
                generateRandomSolution(antRoute, antIndex, -1, Dist, size, state);
            length = antRouteLength(Dist, antRoute, antIndex, size);
            additive = (bestFit) / length / ALPHA * (11-iii);
            if (length > 0) {
                for (int jj = 0; jj < size; jj++) {
                    int source = antRoute[antIndex * size + jj];
                    int dest = antRoute[antIndex * size + (jj + 1) % size];
                    double* ptr = &(Pheromone[source * size + dest]);

                    atomicAdd(ptr, additive);
                }
            }
            block.sync();
        }
        

        for (int gen = 0; gen < maxGenerations; gen++) {

            // Előző ciklus feromonjainak mérséklése
            if (tr == 0) {
                for (int ii = 0; ii < size; ii++)
                    for (int jj = 0; jj < size; jj++)
                        Pheromone[ii * size + jj] *= ALPHA;
            }
            block.sync();


            double length, additive;

            // értékeljük a kapott megoldást
            length = antRouteLength(Dist, antRoute, antIndex, size);
            additive = bestFit / length / ALPHA * 10;
            //sequencePrint(&(antRoute[i * size]), Dist, size);


            if (length > 0) {   // bejárható kör
                for (int jj = 0; jj < size; jj++) {
                    int source = antRoute[antIndex * size + jj];
                    int dest = antRoute[antIndex * size + (jj + 1) % size];
                    double* ptr = &(Pheromone[source * size + dest]);

                    atomicAdd(ptr, additive);
                }

            }
            block.sync();

            // Új hangyák indulnak a feromonok után
            followPheromones(Pheromone, antRoute, antIndex, size, state);

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


__global__ void AntKernel_multiBlock(
    double* Dist,       // Költségfüggvény input
    double* Pheromone,
    int* Route,         // Sorrend output
    bool* FoundRoute,   // Létezés output
    size_t size,        // Gráf csúcsainak száma
    int* antRoute,      // Segédtömb
    int antNum,         // Hangyák száma
    curandState* state,

    // Globális változók
    bool* invalidInput,   // Hibás bemenetet időben kell észlelni
    bool* isolatedVertex, // Ha van izolált csúcs, akkor nincs bejárás (egyszerű teszt)
    double* bestFit
) {
    // Dist (i,j) eleme i. csúcsból a j. csúcs távolsága
    // Ha nincs köztük él akkor Dist(i,j) = -1 (elvárt szintaktika)
    // Szinkronizációs változó a teljes griden belül
    grid_group grid = this_grid();
    if (!grid.is_valid())
        return;
    grid.sync();
    int antIndex = blockIdx.x * blockDim.x + threadIdx.x;  // Hangya index
    int tr = grid.thread_rank();
    // Kezdeti érték adása a 0. thread által
    if (tr == 0) {
        *invalidInput = false;
        *isolatedVertex = false;
        *bestFit = 0.0;
        *FoundRoute = false;
    }
    // Kezeljük, a size=0 vagy 1 esetet
    if (size == 0 || size == 1)   // Nincs gráf, értelmetlen bemenet
        return;
    grid.sync();

    if (tr == 0) {
        bool foundNeighboor = false;
        int ii, jj;
        for (ii = 0; ii < size; ii++) {
            for (jj = 0; jj < size; jj++) {
                // Pheromone gráf kezdeti értéke (anti-egységmátrix, csak főátlóban 0)
                if (ii == jj)
                    Pheromone[ii * size + jj] = 0;
                else
                    Pheromone[ii * size + jj] = INITIAL_PHEROMONE_VALUE;

                // Hibakeresés a gráfban N szálon 
                // Megvizsgáljuk, hogy van-e érvénytelenül megadott gráfél 
                // (nem pozitív, vagy -1, vagy 0 ha i=j)
                if (ii != jj && Dist[ii * size + jj] <= 0 && Dist[ii * size + jj] != -1) {
                    printf("Dist(%d,%d) incorrect!\n", ii, jj);
                    *invalidInput = true;
                    break;
                }
                if (Dist[ii * size + jj] > 0) {
                    // Van szomszédja, tehát nem izolált csúcs
                    foundNeighboor = true;
                }
            }
            if (!foundNeighboor) { // Nem volt szomszédja
                printf("Vertex %d isolated!\n", ii);
                *isolatedVertex = true;
            }
        }
    }
    grid.sync();

    if (*invalidInput || *isolatedVertex) {   // Ha érvénytelen bemenet volt, akkor nem folytatjuk
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
        grid.sync();
        return;
    }

    // Maradt: Helyes bemenet, legalább 3 csúcs, mindegyik mutat valahova

    // Számoljuk ki az ideális (nem mindig elérhető) alsó becslést a bejárás hosszára
    // Minden csúcshoz a hozzá legközelebb állót vesszük hozzá
    // Ez az eredmény a Pheromone mátrix átértékeléséhez kell
    if (tr == 0) {
        for (int ii = 0; ii < size; ii++) {
            double minAdditive = minDist(Dist, ii, size);
            if (minAdditive > 0) // Csak ha van legalább 1 szomszédja: kell legyen
                *bestFit += minAdditive;
        }
    }
    grid.sync();


    double length, additive;
    // Választunk egy random utat a hangyának (az sem biztos hogy lehetséges az út)
    for (int ii = 0; ii < RANDOM_GENERATIONS || ii < size; ii++) {
        if(ii<size) // Mindenféle első csúcs legyen legalább egyszer
            generateRandomSolution(antRoute, antIndex, ii, Dist, size, state);
        else
            generateRandomSolution(antRoute, antIndex,-1, Dist, size, state);
        length = antRouteLength(Dist, antRoute, antIndex, size);
        additive = (*bestFit) / length / ALPHA * 10;
        if (length > 0) {
            for (int jj = 0; jj < size; jj++) {
                int source = antRoute[antIndex * size + jj];
                int dest = antRoute[antIndex * size + (jj + 1) % size];
                double* ptr = &(Pheromone[source * size + dest]);

                atomicAdd(ptr, additive);
            }
        }
        grid.sync();
    }
    /*if (tr == 0) {
        print(Pheromone, size, size);
        printf("Moho algoritmus szekvencia: ");
        greedySequence(Pheromone, antRoute, size);
        sequencePrint(antRoute, Dist, size);
    }
    grid.sync();*/



    for (int gen = 0; gen < maxGenerations;gen++){    // Előző ciklus feromonjainak mérséklése
        if (tr == 0) {
            for (int ii = 0; ii < size; ii++)
                for (int jj = 0; jj < size; jj++)
                    Pheromone[ii * size + jj] *= ALPHA;
        }
        grid.sync();
 
        // értékeljük a kapott megoldást
        length = antRouteLength(Dist, antRoute, antIndex, size);
        additive = (*bestFit) / length / ALPHA * 10;
        if (length > 0) {
            for (int jj = 0; jj < size; jj++) {
                int source = antRoute[antIndex * size + jj];
                int dest = antRoute[antIndex * size + (jj + 1) % size];
                double* ptr = &(Pheromone[source * size + dest]);

                atomicAdd(ptr, additive);
            }
        }
        grid.sync();

        // Új hangyák indulnak a feromonok után
        followPheromones(Pheromone, antRoute, antIndex, size, state);
        
        grid.sync();
    }

    // Küldjük el a feromon úton a végső hangyát
    if (tr == 0) {


        followPheromones(Pheromone, Route, 0, size, state);
        double length;
        length = antRouteLength(Dist, Route, 0, size);
        *FoundRoute = (length > 0);
    }
    return;
};

// Inicializálja minden szál számára a random seedet
__global__ void setup_kernel(curandState* state, unsigned long seed)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;;
    curand_init(seed, id, id, &state[id]);
}

// Megadja, hogy az idx. csúcstól milyen távol van a legközelebbi csúcs
// Dist tömbben megkeresi a legkisebb pozitív számot, ha nincs: -1
__device__ double minDist(double* Dist, int idx, size_t size) {

    double min = -1;
    bool foundPositive = false;
    for (int i = 0; i < size; ++i) {
        // Vagy még nem találtunk: ekkor ha pozitív, ő lesz a minimum
        // Ha már találtunk, akkor ha kisebb de pozitív, ő lesz a minimum
        bool condition = (!foundPositive && i != idx && Dist[idx * size + i] > 0) ||
            (foundPositive && i != idx && Dist[idx * size + i] < min && Dist[idx * size + i] > 0);
        if (condition) {
            foundPositive = true;
            min = Dist[idx * size + i];
        }
    }
    return min;
}

// Megadja, hogy az idx. csúcshoz melyik a legközelebbi csúcs
// Dist tömbben megkeresi a legkisebb pozitív szám indexét, ha nincs: -1
__device__ int minDistIdx(double* Dist, int idx, size_t size) {
    int minIdx = -1, min = -1;
    bool foundPositive = false;
    for (int i = 0; i < size; ++i) {
        // Vagy még nem találtunk: ekkor ha pozitív, ő lesz a minimum
        // Ha már találtunk, akkor ha kisebb de pozitív, ő lesz a minimum
        bool condition = (!foundPositive && i != idx && Dist[idx * size + i] > 0) ||
            (foundPositive && i != idx && Dist[idx * size + i] < min && Dist[idx * size + i] > 0);
        if (condition) {
            foundPositive = true;
            min = Dist[idx * size + i];
            minIdx = i;
        }
    }
    return minIdx;
}

// Generál egy random csúcs sorrendet
__device__ void generateRandomSolution(int* antRoute,unsigned int antIndex,int secondVertex, double* Dist, size_t size, curandState* state) {
    // Azt elvárom, hogy a 0. csúcsban kezdjen
    for (int idx = 0; idx < size; idx++) {    // Tömb [0, 1, 2 ... size-1]
        antRoute[antIndex * size + idx] = idx;
    }
    int min_rand_int, max_rand_int = size - 1;
    bool condition = (secondVertex > 0 && secondVertex < size);
    min_rand_int = condition ? 2 : 1;
    if (condition && secondVertex != 1) {
        antRoute[antIndex * size + 1] = secondVertex;
        antRoute[antIndex * size + secondVertex] = 1;
    }
    // Van érvényes megadás a második csúcsra
    
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

/*
// Generál egy random csúcs sorrendet
// Különlegesség: a 0. hangyát megpróbálja mindig a lehető legközelebbre küldeni (hátha)
__device__ void generateRandomSolution(int* antRoute, int antIndex, double* Dist, size_t size, curandState* state) {
    // Azt elvárom, hogy a 0. csúcsban kezdjen
    antRoute[antIndex * size + 0] = 0;
    int newParam;
    for (int idx = 1; idx < size; idx++) {
        int mindistidx = minDistIdx(Dist, idx - 1, size);
        if ((antIndex % INITIALHINTPERCENTAGE) == 0 && !alreadyListed(antRoute, 0, size, idx, mindistidx)) {
            newParam = mindistidx;
            //double dist = Dist[(idx-1) * size + mindistidx];
            //printf("Source: %d Managed min neightboor: %d, distance: %.0f\n", idx-1, mindistidx, dist);
        }
        else
            do {
                float newfloat = curand_uniform(&state[antIndex]);  // 0 és 1 közötti számot ad
                newfloat *= (size - 1) + 0.999999;  // A kívánt tartományba transzformáljuk
                newParam = (int)truncf(newfloat);
            } while (alreadyListed(antRoute, antIndex, size, idx, newParam));
            antRoute[antIndex * size + idx] = newParam;
    }
}
*/

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
//
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
        for(int j=0;j<maxTryNumber;j++){
            // 0 és sumPheromone közötti random szám generálása
            curandState* statePtr = &(state[antIndex]);
            double myranddbl = curand_uniform_double(statePtr) * sumPheromone;
            double temp = Pheromone[source * size + 0]; // Ebben fogjuk összeadni a mátrix értékeket
            
            for (newparam = 0; newparam < size - 1; newparam++) {   // Ha newparam = size-1 akkor már nincs hova nagyobbat keresni
                if (myranddbl < temp)
                    break;
                temp += Pheromone[source * size + newparam+1];
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


/*//__device__ void followPheromones(const double* Pheromone, int* antRoute, int antIndex, size_t size, curandState* state) {
    // Azt elvárom, hogy a 0. csúcsban kezdjen
    antRoute[antIndex * size + 0] = 0;
    for (int i = 0; i < size - 1; i++) {
        int newParam;
        int source = antRoute[antIndex * size + i];
        double sumPheromone = 0; // Ez alapján súlyozzuk a csúcsokat
        for (int j = 0; j < size; j++)
            if (j != source)
                sumPheromone += Pheromone[source * size + j];

        size_t tryNumber = 0,
            maxTrynumber = 40 * size; // Védelem a végtelen váróciklusok ellen : 
        // Ha olyan helyre kerülne ahonnan 0 vsz-gel tudna kikerülni

        do {
            // 0 és sumPheromone közötti random számot ad
            double newfloat = curand_uniform_double(&state[antIndex]) * sumPheromone;

            double temp = Pheromone[source * size + 0];
            newParam = 0;

            while (newParam < size - 1) {  // Sorsolási mechanizmus
                if (newfloat < temp)
                    break;
                newParam++;
                temp += Pheromone[source * size + newParam];

            }
            //if(antIndex == 0 && source == 0)
            //printf("Ant%d Source: %d, %f %f, chosen %d\n",antIndex,source, newfloat, sumPheromone, newParam);
            tryNumber++;
        } while (alreadyListed(antRoute, antIndex, size, i + 1, newParam) && tryNumber < maxTrynumber);

        // Ha nem talált volna a súlyozott vsz szerint kiutat: minden csúcs azonos vsz-gel
        // Előfordulhat pl nagy, páros gráf esetén
        if (tryNumber == maxTrynumber) {
            //printf("%d. Hangya %d. iteracio veletlenszeru sorsolas\n", antIndex, i);
            do {
                float newfloat = curand_uniform(&state[antIndex]);  // 0 és 1 közötti számot ad
                newfloat *= (size - 1) + 0.999999;  // A kívánt tartományba transzformáljuk
                newParam = (int)truncf(newfloat);
            } while (alreadyListed(antRoute, antIndex, size, i + 1, newParam));
        }

        // Végül a kapott új csúcs
        antRoute[antIndex * size + i + 1] = newParam;
    }

}*/

/// 
/// Megadja, hogy az adott sorban melyik a legnagyobb még nem választott feromon
/// 
/// param name="Pheromone" : Ez a mátrix tartalmazza a feromon értékeket
/// <param name="row"></param>
/// <param name="size"></param>
/// <param name="idx"></param>
/// <param name="antRoute"></param>
/// <returns></returns>
__device__ int maxInIdxRow(const double* Pheromone,int row, size_t size,int idx,int* antRoute){
    int maxidx = idx;
    double max = Pheromone[row * size + idx];
    
    for (int i = 0; i < size; i++) {
        double vizsgalt = Pheromone[row * size + i];
        if (vizsgalt > max && !alreadyListed(antRoute,0,size,idx,i)) {
            max = vizsgalt;
            maxidx = i;
        }
    }
    //printf("%d. elem %.2f ertekkel: %d\n", idx, max, maxidx);

    return maxidx;
}

__device__ void greedySequence(const double* Pheromone, int* antRoute, size_t size) {
    antRoute[0] = 0;
    for (int i = 1; i < size; i++) {
        antRoute[i] = maxInIdxRow(Pheromone, antRoute[i-1], size, i, antRoute);
    }
}