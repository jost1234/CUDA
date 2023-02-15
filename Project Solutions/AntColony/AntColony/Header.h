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

// Megh�vand� CUDA f�ggv�ny
cudaError_t AntCUDA(double* h_Dist, int* h_Route, double* h_Pheromone, bool* h_FoundRoute, unsigned int antNum, size_t size);

// Inicializ�lja minden sz�l sz�m�ra a random seedet
__global__ void setup_kernel(curandState* state, unsigned long seed);

// 1 blokkos f�ggv�ny
__global__ void AntKernel_1Block(
    double* Dist,       // K�lts�gf�ggv�ny input
    double* Pheromone,
    int* Route,         // Sorrend output
    bool* FoundRoute,   // L�tez�s output
    size_t size,        // Gr�f cs�csainak sz�ma
    int* antRoute,      // Seg�dt�mb
    int antNum,         // Hangy�k sz�ma
    curandState* state
);

__global__ void AntKernel_multiBlock(
    double* Dist,       // K�lts�gf�ggv�ny input
    double* Pheromone,
    int* Route,         // Sorrend output
    bool* FoundRoute,   // L�tez�s output
    size_t size,        // Gr�f cs�csainak sz�ma
    int* antRoute,      // Seg�dt�mb
    int antNum,         // Hangy�k sz�ma
    curandState* state,

    // Glob�lis v�ltoz�k
    bool* invalidInput,   // Hib�s bemenetet id�ben kell �szlelni
    bool* isolatedVertex, // Ha van izol�lt cs�cs, akkor nincs bej�r�s (egyszer� teszt)
    double* bestFit
);

// Megadja, hogy az idx. cs�cst�l milyen t�vol van a legk�zelebbi cs�cs
__device__ double minDist(double* Dist, int idx, size_t size);

// Megadja, hogy az idx. cs�cshoz melyik a legk�zelebbi cs�cs
// Dist t�mbben megkeresi a legkisebb pozit�v sz�m index�t, ha nincs: -1
__device__ int minDistIdx(double* Dist, int idx, size_t size);

// Gener�l egy random cs�cs sorrendet
// K�l�nlegess�g: a 0. hangy�t megpr�b�lja mindig a lehet� legk�zelebbre k�ldeni (h�tha)
__device__ void generateRandomSolution(int* antRoute, unsigned int antIndex,int secondVertex, double* Dist, size_t size, curandState* state);

// Sorrend gener�l�shoz haszn�lt f�ggv�ny
__device__ bool alreadyListed(int* antRoute, int antIndex, size_t size, int idx, int newParam);

// Megadja egy adott bej�r�s hossz�t
// (-1) -gyel t�r vissza ha az adott k�r�t nem bej�rhat�
__device__ double antRouteLength(double* Dist, int* antRoute, int antIndex, size_t size);

// �j hangy�k indulnak a feromonok ut�n
__device__ void followPheromones(const double* Pheromone, int* antRoute, int antIndex, size_t size, curandState* state);

__device__ __host__ double sequencePrint(int* Route, double* Dist, size_t size);


// Kit�rlend� ha nem j�
__device__ void greedySequence(const double* Pheromone, int* antRoute, size_t size);