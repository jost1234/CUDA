#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cooperative_groups.h>
#include <iostream>

// Thread block size
#define BLOCK_SIZE 1024


const int ants = 1024;
#define ALPHA 0.75
const int maxGenerations = 2000;
#define INITIAL_PHEROMONE_VALUE 1000
#define RANDOM_GENERATIONS 400
#define SERIALMAXTRIES 100

__host__ __device__ void print(double* A, size_t size) {
    for (int ii = 0; ii < size; ii++) {
        for (int jj = 0; jj < size; jj++)
            printf("%.2f ", A[ii * size + jj]);
        printf("\n");
    }
    printf("\n");
}

__device__ __host__ int my_ceil(int osztando, int oszto) {
    if (!(osztando % oszto)) return osztando / oszto;
    else return	osztando / oszto + 1;
}

// Meghívandó CUDA függvény
cudaError_t AntCUDA(double* h_Dist, int* h_Route, double* h_Pheromone, bool* h_FoundRoute, unsigned int antNum, size_t size);

// Inicializálja minden szál számára a random seedet
__global__ void setup_kernel(curandState* state, unsigned long seed);

// 1 blokkos függvény
__global__ void AntKernel_1Block(
    double* Dist,       // Költségfüggvény input
    double* Pheromone,
    int* Route,         // Sorrend output
    bool* FoundRoute,   // Létezés output
    size_t size,        // Gráf csúcsainak száma
    int* antRoute,      // Segédtömb
    int antNum,         // Hangyák száma
    curandState* state
);

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
    bool* invalidInput,   // Hibás bemenetet idõben kell észlelni
    bool* isolatedVertex, // Ha van izolált csúcs, akkor nincs bejárás (egyszerû teszt)
    double* bestFit
);

// Megadja, hogy az idx. csúcstól milyen távol van a legközelebbi csúcs
__device__ double minDist(double* Dist, int idx, size_t size);

// Megadja, hogy az idx. csúcshoz melyik a legközelebbi csúcs
// Dist tömbben megkeresi a legkisebb pozitív szám indexét, ha nincs: -1
__device__ int minDistIdx(double* Dist, int idx, size_t size);

// Generál egy random csúcs sorrendet
// Különlegesség: a 0. hangyát megpróbálja mindig a lehetõ legközelebbre küldeni (hátha)
__device__ void generateRandomSolution(int* antRoute, unsigned int antIndex,int secondVertex, double* Dist, size_t size, curandState* state);

// Sorrend generáláshoz használt függvény
__device__ bool alreadyListed(int* antRoute, int antIndex, size_t size, int idx, int newParam);

// Megadja egy adott bejárás hosszát
// (-1) -gyel tér vissza ha az adott körút nem bejárható
__device__ double antRouteLength(double* Dist, int* antRoute, int antIndex, size_t size);

// Új hangyák indulnak a feromonok után
__device__ void followPheromones(const double* Pheromone, int* antRoute, int antIndex, size_t size, curandState* state);

__device__ __host__ double sequencePrint(int* Route, double* Dist, size_t size);


// Kitörlendõ ha nem jó
__device__ void greedySequence(const double* Pheromone, int* antRoute, size_t size);