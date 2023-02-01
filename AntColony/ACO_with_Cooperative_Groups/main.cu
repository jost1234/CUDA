
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
#include <string.h>
#include <stdlib.h>

using namespace cooperative_groups;

// Random sorrend generáláshoz használt függvény
__device__ bool alreadyListed(int* antRoute, int antIndex, size_t size, int idx, int newParam) {
    if (idx >= size)
        return true;    // Inkább okozzunk végtelen ciklust, mint túlcímzést
    for (int i = 0; i < idx; ++i)
        if (newParam == antRoute[antIndex * size + i])
            return true;
    return false;
}

// Generál egy random csúcs sorrendet
// Különlegesség: a 0. hangyát megpróbálja mindig a lehető legközelebbre küldeni (hátha)
__device__ void generateRandomSolution(int* antRoute, int antIndex,double* Dist, size_t size, curandState* state) {
    // Azt elvárom, hogy a 0. csúcsban kezdjen
    antRoute[antIndex * size + 0] = 0;
    int newParam;
    for (int idx = 1; idx < size; idx++) {
        int mindistidx = minDistIdx(Dist, idx-1, size);
        if ((antIndex% INITIALHINTPERCENTAGE) == 0 && !alreadyListed(antRoute, 0, size, idx, mindistidx)) {
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
};

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
    int minIdx = -1, int min = -1;
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

__device__ void followPheromones(const double* Pheromone, int* antRoute, int antIndex, size_t size, curandState* state) {
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
            //if(antIndex == 0)
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

};

__global__ void AntKernel_1Block(
    double* Dist,       // Költségfüggvény input
    double* Pheromone,
    int* Route,         // Sorrend output
    bool* FoundRoute,   // Létezés output
    size_t size,        // Gráf csúcsainak száma
    int* antRoute,      // Segédtömb
    curandState* state
)
{
    // Dist (i,j) eleme i. csúcsból a j. csúcs távolsága
    // Ha nincs köztük él akkor Dist(i,j) = -1 (elvárt szintaktika)
    thread_block block = this_thread_block();
    
    int i = blockIdx.x * blockDim.x + threadIdx.x;  // Hangya index 0 - (antNum-1)
    int j = blockIdx.y * blockDim.y + threadIdx.y;  // size db : 0 - (size-1)
    int k = blockIdx.z * blockDim.z + threadIdx.z;  // 1 db
    int tr = block.thread_rank();
    if (i >= antNum || j >= size || k > 0)  // Itt megtehető ez az egyszerűsítés
        return;

    // Megosztott változók
    __shared__ bool invalidInput;   // Hibás bemenetet időben kell észlelni
    __shared__ bool isolatedVertex; // Ha van izolált csúcs, akkor nincs bejárás (egyszerű teszt)
    __shared__ double bestFit;
    __shared__ size_t generation;

    // Kezdeti érték adása a 0. thread által
    if (i == 0 && j == 0) {
        invalidInput = false;
        isolatedVertex = false;
        bestFit = 0.0;
        generation = 0;
        *FoundRoute = false;
    }

    // Kezeljük, a size=0 vagy 1 esetet
    if (size == 0 || size == 1)   // Nincs gráf, értelmetlen bemenet
        return;

    block.sync();

    if (i == 0) {
        bool foundNeighboor = false;
        for (int jj = 0; jj < size; jj++) {
            // Pheromone gráf kezdeti értéke (anti-egységmátrix, csak főátlóban 0)
            if (j == jj)
                Pheromone[j * size + jj] = 0;
            else
                Pheromone[j * size + jj] = 100;
            
            // Hibakeresés a gráfban N szálon 
            // Megvizsgáljuk, hogy van-e érvénytelenül megadott gráfél 
            // (nem pozitív, vagy -1, vagy 0 ha i=j)
            if (j != jj && Dist[j * size + jj] <= 0 && Dist[j * size + jj] != -1) {
                printf("Dist(%d,%d) incorrect!\n", j, jj);
                invalidInput = true;
                break;
            }
            if (Dist[j * size + jj] > 0) {
                // Van szomszédja, tehát nem izolált csúcs
                foundNeighboor = true;
            }
        }
        if (!foundNeighboor) { // Nem volt szomszédja
            printf("Vertex %d isolated!\n", j);
            isolatedVertex = true;
        }
    }

    block.sync();
    
    if (invalidInput || isolatedVertex) {   // Ha érvénytelen bemenet volt, akkor nem folytatjuk
        return;                             // Ha van izolált csúcs, akkor nem folytatjuk
    }
    
    
    // 2 csúcs esete : lekezeljük gyorsan 1 szálon
    if (size == 2) {
        if (i == 0 && j == 0) {
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
    if (i == 0) {
        double minAdditive = minDist(Dist, j, size);
        if (minAdditive > 0)
            atomicAdd(&bestFit, minAdditive);
        else
            printf("Isolated vertex!\n minDist=%.2f\n", minAdditive);
    }
    block.sync();

    //printf("Theoretical minimum: %f", bestFit);

    // i darab hangyát használunk

    if (j == 0) {
        // Választunk egy random utat a hangyának (az sem biztos hogy lehetséges az út)
        generateRandomSolution(antRoute, i,Dist, size, state);
    }
    block.sync();

    do {

        // Előző ciklus feromonjainak mérséklése
        if (i == 0) {

            for (int jj = 0; jj < size; jj++)
                Pheromone[j * size + jj] *= ALPHA;
        }
        block.sync();

        double length, additive;
        if (j == 0) {
            // értékeljük a kapott megoldást
            length = antRouteLength(Dist, antRoute, i, size);
            additive = bestFit / length / ALPHA * 10;
            //sequencePrint(&(antRoute[i * size]), Dist, size);
            //int antindex = i;
            //printf("Ant%d Length: %f, Additive : %f Sequence: %d %d %d %d %d\n\n", antindex, length, additive,
             //   antRoute[i * size], antRoute[i * size + 1], antRoute[i * size + 2], antRoute[i * size + 3], antRoute[i * size + 4]);
        }

        if (length > 0) {   // bejárható kör

            if (j == 0)
                for (int jj = 0; jj < size; jj++) {
                    int source = antRoute[i * size + jj];
                    int dest = antRoute[i * size + (jj + 1) % size];
                    double* ptr = &(Pheromone[source * size + dest]);

                    atomicAdd(ptr, additive);
                }

        }
        block.sync();

        // Új hangyák indulnak a feromonok után
        if (j == 0) {
            followPheromones(Pheromone, antRoute, i, size, state);
        }
        block.sync();

        if (i == 0 && j == 0) {
            generation++;
            //printf("Gen%d over\n\n", generation);
            //printf("Pheromone matrix:\n");
            //print(Pheromone, size, size);
        }
        block.sync();

    } while (generation < maxGenerations);

    block.sync();

    // Küldjük el a feromon úton a végső hangyát
    if (i != 0 || j != 0)
        return;

    followPheromones(Pheromone, Route, 0, size, state);
    double length;
    length = antRouteLength(Dist, Route, 0, size);
    *FoundRoute = (length > 0);
}

__global__ void AntKernel_multiBlock(
    double* Dist,       // Költségfüggvény input
    double* Pheromone,
    int* Route,         // Sorrend output
    bool* FoundRoute,   // Létezés output
    size_t size,        // Gráf csúcsainak száma
    int* antRoute,      // Segédtömb
    curandState* state,
                        // Globális változók
    bool* invalidInput,   // Hibás bemenetet időben kell észlelni
    bool* isolatedVertex, // Ha van izolált csúcs, akkor nincs bejárás (egyszerű teszt)
    double* bestFit,
    size_t* generation   
    )
{
    // Dist (i,j) eleme i. csúcsból a j. csúcs távolsága
    // Ha nincs köztük él akkor Dist(i,j) = -1 (elvárt szintaktika)
    // Szinkronizációs változó a teljes griden belül
    grid_group grid = this_grid();
    int i = blockIdx.x * blockDim.x + threadIdx.x;  // oszlopváltozó
    int j = blockIdx.y * blockDim.y + threadIdx.y;  // sorváltozó
    int tr = grid.thread_rank();

    if (!grid.is_valid())
        return;
    grid.sync();
    
    // Kezdeti érték adása a 0. thread által
    if (tr == 0) {
        *invalidInput = false;
        *isolatedVertex = false;
        *bestFit = 0.0;
        *generation = 0;
        *FoundRoute = false;
    }

    // Kezeljük, a size=0 vagy 1 esetet
    if (size == 0 || size == 1)   // Nincs gráf, értelmetlen bemenet
        return;
    grid.sync();

    if (i == 0 && j < size) {
        bool foundNeighboor = false;
        for (int jj = 0; jj < size; jj++) {
            // Pheromone gráf kezdeti értéke (anti-egységmátrix, csak főátlóban 0)
            if (j == jj)
                Pheromone[j * size + jj] = 0;
            else
                Pheromone[j * size + jj] = 100;

            // Hibakeresés a gráfban N szálon 
            // Megvizsgáljuk, hogy van-e érvénytelenül megadott gráfél 
            // (nem pozitív, vagy -1, vagy 0 ha i=j)
            if (j != jj && Dist[j * size + jj] <= 0 && Dist[j * size + jj] != -1) {
                printf("Dist(%d,%d) incorrect!\n", j, jj);
                *invalidInput = true;
                break;
            }
            if (Dist[j * size + jj] > 0) {
                // Van szomszédja, tehát nem izolált csúcs
                foundNeighboor = true;
            }
        }
        if (!foundNeighboor) { // Nem volt szomszédja
            printf("Vertex %d isolated!\n", j);
            *isolatedVertex = true;
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
    if (i == 0 && j < size) {
        double minAdditive = minDist(Dist, j, size);
        if (minAdditive > 0)
            atomicAdd(bestFit, minAdditive);
        else    // Elvileg soha nem kéne lefusson, ugyanis le van kezelve
            printf("Isolated vertex!\n minDist=%.2f\n", minAdditive);
    }
    grid.sync();

    if (j == 0 && i < antNum) {
        // Választunk egy random utat a hangyának (az sem biztos hogy lehetséges az út)
        generateRandomSolution(antRoute, i,Dist, size, state);
    }
    grid.sync();

    do {
        // Előző ciklus feromonjainak mérséklése
        if (i == 0 && j < size) {
            for (int jj = 0; jj < size; jj++)
                Pheromone[j * size + jj] *= ALPHA;
        }
        grid.sync();

        double length, additive;
        if (j == 0 && i < antNum) {
            // értékeljük a kapott megoldást
            length = antRouteLength(Dist, antRoute, i, size);
            additive = (*bestFit) / length / ALPHA * 10;
        }
        if (length > 0) {   // bejárható kör

            if (j == 0 && i < antNum)
                for (int jj = 0; jj < size; jj++) {
                    int source = antRoute[i * size + jj];
                    int dest = antRoute[i * size + (jj + 1) % size];
                    double* ptr = &(Pheromone[source * size + dest]);

                    atomicAdd(ptr, additive);
                }
        }
        grid.sync();

        // Új hangyák indulnak a feromonok után
        if (j == 0 && i < antNum) {
            followPheromones(Pheromone, antRoute, i, size, state);
        }
        if (tr == 0) {
            (*generation)++;
            //printf("Gen%d over\n\n", *generation);
            //printf("Pheromone matrix:\n");
            //print(Pheromone, size, size);
        }
        grid.sync();
    } while (*generation < maxGenerations);


    // Küldjük el a feromon úton a végső hangyát
    if (tr == 0) {
        

        followPheromones(Pheromone, Route, 0, size, state);
        double length;
        length = antRouteLength(Dist, Route, 0, size);
        *FoundRoute = (length > 0);
    }
    return;
}

// Inicializálja minden szál számára a random seedet
__global__ void setup_kernel(curandState* state, unsigned long seed)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;;
    curand_init(seed, id, id, &state[id]);
}

cudaError_t AntCUDA(double* Dist, int* Route, double* hPheromone, bool* FoundRoute, size_t size) {

    // size = 0,1 : érvénytelen bemenetek
    if (size == 0 && size == 1) {
        fprintf(stderr, "Incorrect size! Must be at least 2.");
        return cudaError_t::cudaErrorInvalidConfiguration;
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
    size_t* d_generation;
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
    cudaStatus = cudaMalloc((void**)&d_generation, sizeof(size_t));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "antRoute cudaMalloc failed!");
        goto Error;
    }

    // Adatok másolása : Host -> Device
    cudaStatus = cudaMemcpy(d_Dist, Dist, Dist_bytes, cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "Dist cudaMemcpy failed!");
        goto Error;
    }

    Route[0] = 0; // Route[0] = 0, azaz a 0. városban kezdünk (önkényesen kijelölhető kezdő város)
    cudaStatus = cudaMemcpy(d_Route, Route, Route_bytes, cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "Route cudaMemcpy failed!");
        goto Error;
    }

    int gridAntNum = my_ceil(antNum * size,BLOCK_SIZE),
        gridSize = my_ceil(size,BLOCK_SIZE);
    
    dim3 tpb(my_ceil(antNum, gridAntNum), 1, 1);
    curandState* devStates;
    cudaMalloc(&devStates, antNum * sizeof(curandState));
    
    // setup seeds
    
    setup_kernel << < gridAntNum, tpb >> > (devStates, time(NULL)+rand());

    // Kernel hívás
    //dim3 dimBlock(antNum, size);
    //AntKernel_1Block << < 1, dimBlock >> > (d_Dist, d_Pheromone, d_Route, d_FoundRoute, size, antRoute, devStates);
    
    
    dim3 dimBlock((int)antNum/gridAntNum, my_ceil(size, gridSize),1);
    dim3 dimGrid(gridAntNum, gridSize,1);

    double min = 10000000.0;
    double sum = 0.0;
    for (int iter = 0; iter < SERIALMAXTRIES; iter++) {
        printf("Attempt #%d | ", iter);

        if (gridAntNum == 1 && gridSize == 1) {
            printf("Called function with 1 Block size Grid: dimBlock(%d,%d)\n", dimBlock.x, dimBlock.y);
            AntKernel_1Block << < 1, dimBlock >> > (d_Dist, d_Pheromone, d_Route, d_FoundRoute, size, antRoute, devStates);
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
            void* kernelArgs[] = { &d_Dist,&d_Pheromone, &d_Route, &d_FoundRoute, &size, &antRoute, &devStates,
                &d_invalidInput, &d_isolatedVertex, &d_bestFit, &d_generation };
            printf("Called function multiBlock: dimGrid(%d,%d) dimBlock(%d,%d)\n", dimGrid.x, dimGrid.y, dimBlock.x, dimBlock.y);
            cudaLaunchCooperativeKernel((void*)AntKernel_multiBlock, dimGrid, dimBlock, kernelArgs);


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
        cudaStatus = cudaMemcpy(Route, d_Route, Route_bytes, cudaMemcpyDeviceToHost);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "Route dev->host cudaMemcpy failed!");
            goto Error;
        }
        cudaStatus = cudaMemcpy(FoundRoute, d_FoundRoute, sizeof(bool), cudaMemcpyDeviceToHost);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "FoundRoute flag dev->host cudaMemcpy failed!");
            goto Error;
        }
        cudaStatus = cudaMemcpy(hPheromone, d_Pheromone, Dist_bytes, cudaMemcpyDeviceToHost);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "Pheromone dev->host cudaMemcpy failed!");
            goto Error;
        }

        
        if (*FoundRoute && cudaStatus == cudaSuccess) {

            double _length = sequencePrint(Route, Dist, size);
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
    cudaFree(d_generation);
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
                fprintf(stderr, "Unable to open file \"%s\"",argv[i]);
                return -1;
            }
            Dist_filename_Idx = i;
            printf("Opening file \"%s\"!\n", argv[Dist_filename_Idx]);
            foundDistMatrix = true;
        }
    if (!foundDistMatrix) {
        fprintf(stderr, "Please give a file in command line arguments to set the Distance Matrix!\n");
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
    for (int ii = 0; ii < size; ++ii) {
        double temp;
        for (int jj = 0; jj < size; ++jj) {
            if (fscanf_s(Dist_file, "%lf", &temp) == 0) {
                fprintf(stderr, "Error reading file \"%s\"\n", argv[Dist_filename_Idx]);
                fclose(Dist_file);
                return -1;
            }
            Dist[ii * size + jj] = temp;
        }
        fscanf_s(Dist_file, "\n");
    }
    printf("Closing file \"%s\"!\n", argv[Dist_filename_Idx]);
    if (fclose(Dist_file) != 0) {
        fprintf(stderr, "Unable to close file \"%s\"!\n", argv[Dist_filename_Idx]);
    }
    printf("Given Dist matrix:\n");
    print(Dist, size, size);



    double *Pheromone = (double*)malloc(size*size*sizeof(double));
    if (Pheromone == NULL)
        goto End;
    // Sorrend vektor: Route[0] = 0, azaz a 0. városban kezdünk 
    // (önkényesen kijelölt kezdő város)
    int* Route = (int*)malloc(size*sizeof(int));
    if (Route == NULL)
        goto End;
    bool found;
    printf("Alternating Salesman problem with Ant Colony Algorithm\n");
    cudaError_t CUDAstate = AntCUDA(Dist, Route, Pheromone, &found, size);
    
End:
    free(Pheromone);
    free(Route);

    return 0;
}

