#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cooperative_groups.h>
#include <iostream>



// Thread block size
#define BLOCK_SIZE 1024


const int antNum = 300;
#define ALPHA 0.8
const int maxGenerations = 500;

#define INITIALHINTPERCENTAGE 10
#define SERIALMAXTRIES 200

// Hogy felismerje a precompiler
#ifdef __INTELLISENSE__
void __syncthreads();
#endif


//print the matrix.
template<class T>
__host__ __device__ void print(T* A, int Row, int Col) {
    for (int ii = 0; ii < Row; ii++)
    {
        for (int jj = 0; jj < Col; jj++)
            printf("%.2f ", A[ii * Col + jj]);
        printf("\n");
    }
    printf("\n");
}

__device__ __host__ int my_ceil(int osztando, int oszto) {
    if (!(osztando % oszto)) return osztando / oszto;
    else return	osztando / oszto + 1;
}

// Random sorrend generáláshoz használt függvény
__device__ bool alreadyListed(int* antRoute, int antIndex, size_t size, int idx, int newParam);

// Generál egy random csúcs sorrendet
// Különlegesség: a 0. hangyát megpróbálja mindig a lehetõ legközelebbre küldeni (hátha)
__device__ void generateRandomSolution(int* antRoute, int antIndex, double* Dist, size_t size, curandState* state);

// Megadja egy adott bejárás hosszát
// (-1) -gyel tér vissza ha az adott körút nem bejárható
__device__ double antRouteLength(double* Dist, int* antRoute, int antIndex, size_t size);

// Megadja, hogy az idx. csúcstól milyen távol van a legközelebbi csúcs
__device__ double minDist(double* Dist, int idx, size_t size);

// Megadja, hogy az idx. csúcshoz melyik a legközelebbi csúcs
// Dist tömbben megkeresi a legkisebb pozitív szám indexét, ha nincs: -1
__device__ int minDistIdx(double* Dist, int idx, size_t size);

__device__ void followPheromones(const double* Pheromone, int* antRoute,
    int antIndex, size_t size, curandState* state);

__global__ void AntKernel_1Block(
    double* Dist,       // Költségfüggvény input
    double* Pheromone,
    int* Route,         // Sorrend output
    bool* FoundRoute,   // Létezés output
    size_t size,        // Gráf csúcsainak száma
    int* antRoute,      // Segédtömb
    curandState* state
);

__global__ void setup_kernel(curandState* state, unsigned long seed);

cudaError_t AntCUDA(double* Dist, int* Route, double* hPheromone, bool* FoundRoute, size_t size);

__device__ __host__ double sequencePrint(int* Route, double* Dist, size_t size);