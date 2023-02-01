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

// Random sorrend gener�l�shoz haszn�lt f�ggv�ny
__device__ bool alreadyListed(int* antRoute, int antIndex, size_t size, int idx, int newParam);

// Gener�l egy random cs�cs sorrendet
// K�l�nlegess�g: a 0. hangy�t megpr�b�lja mindig a lehet� legk�zelebbre k�ldeni (h�tha)
__device__ void generateRandomSolution(int* antRoute, int antIndex, double* Dist, size_t size, curandState* state);

// Megadja egy adott bej�r�s hossz�t
// (-1) -gyel t�r vissza ha az adott k�r�t nem bej�rhat�
__device__ double antRouteLength(double* Dist, int* antRoute, int antIndex, size_t size);

// Megadja, hogy az idx. cs�cst�l milyen t�vol van a legk�zelebbi cs�cs
__device__ double minDist(double* Dist, int idx, size_t size);

// Megadja, hogy az idx. cs�cshoz melyik a legk�zelebbi cs�cs
// Dist t�mbben megkeresi a legkisebb pozit�v sz�m index�t, ha nincs: -1
__device__ int minDistIdx(double* Dist, int idx, size_t size);

__device__ void followPheromones(const double* Pheromone, int* antRoute,
    int antIndex, size_t size, curandState* state);

__global__ void AntKernel_1Block(
    double* Dist,       // K�lts�gf�ggv�ny input
    double* Pheromone,
    int* Route,         // Sorrend output
    bool* FoundRoute,   // L�tez�s output
    size_t size,        // Gr�f cs�csainak sz�ma
    int* antRoute,      // Seg�dt�mb
    curandState* state
);

__global__ void setup_kernel(curandState* state, unsigned long seed);

cudaError_t AntCUDA(double* Dist, int* Route, double* hPheromone, bool* FoundRoute, size_t size);

__device__ __host__ double sequencePrint(int* Route, double* Dist, size_t size);