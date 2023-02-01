/*
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <cstdlib>
#include <ctime>


// Diagnosztika, paraméterek
#include "Matrix_plusfunctions.cuh"

__global__ void LUdecompKernel(float(*A)[N], float(*L)[N], float(*U)[N]) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;  // oszlopváltozó
    int j = blockIdx.y * blockDim.y + threadIdx.y;  // sorváltozó
    //int k = blockIdx.z * blockDim.z + threadIdx.z;
    if (i >= N || j >= N)
        return;

    // Elõformázás
    U[i][j] = A[i][j];
    if (i < j)
        L[i][j] = 0;
    else if (i == j)
        L[i][i] = 1;
    
    __syncthreads();
    
    //Gauss elimináció u mátrixban
    for (int m = 0; m < N; m++) {
        if (U[m][m] == 0) {
            cudaError_t error = cudaErrorUnknown;
            //exit(error);
            return;
        }
        float x = U[i][m] / U[m][m];    // Ez soronként ugyanaz
        if (j == m && i > m) {
            // Ha a nullázódó oszlopban vagyunk
            U[i][j] = 0;
            L[i][j] = x;

        }
        else if (i > m && j > m) {
            U[i][j] -= x * U[m][j];
        }
        __syncthreads();
    }
    
}

__host__ void LUdecompCUDA(float(*A)[N], float(*L)[N], float(*U)[N]) {
    // Dinamikus helyfoglalás a GPU-ra: használt pointerek
    float(*d_A)[N], (*d_L)[N], (*d_U)[N];
    // Adathalmaz mérete, amit lefoglalunk
    size_t bytes = N * N * sizeof(float);
    // Adatfoglalás
    cudaMalloc((void**)&d_A, bytes);
    cudaMalloc((void**)&d_L, bytes);
    cudaMalloc((void**)&d_U, bytes);
    // Adatok másolása
    cudaMemcpy(d_A, A, bytes, cudaMemcpyHostToDevice);
    
    // Kernel grid, blokk méret
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid(my_ceil(N, dimBlock.x), my_ceil(N, dimBlock.y));
    // Kernel hívás (ideiglenes adattárolókon)
    LUdecompKernel <<<dimGrid, dimBlock >>> (d_A, d_L, d_U);
    
    
    // Feldolgozott adat átvitele a GPU-ról
    cudaMemcpy(L, d_L, bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(U, d_U, bytes, cudaMemcpyDeviceToHost);
    // Ideiglenes adattárolók felszabadítása

    cudaFree(d_A);
    cudaFree(d_L);
    cudaFree(d_U);
}



// Soros változat szabványosan #2, azaz L tartalmaz csupa 1 fõátlót
int LUdecomp(float a[N][N], float l[N][N], float u[N][N]) {
    int i = 0, j = 0, k = 0;
    // Alsó hszög mátrix triviális elemei, a átkerül u-ba
    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            u[i][j] = a[i][j];  // a->u
            if (i < j)          // Felsõ elemek nullák
                l[i][j] = 0;
            else if (i == j)    // Fõátló csupa 1
                l[i][i] = 1;
        }
    }
    // Gauss elimináció u-ban
    for (i = 0; i < N; ++i) {
        float x;
        if (u[i][i] == 0)
            return 1;
        for (j = i + 1; j < N;j++) {
            x = u[j][i] / u[i][i];
            u[j][i] = 0;
            l[j][i] = x;
            for (k = i+1; k < N; k++)
                u[j][k] -= x * u[i][k];
        }
        
    }
    
}

// 3x3 -as példa: 
// 1 1 0
// 2 1 3
// 3 1 1
//



int main() {
    float  (*L)[N], (*U)[N];
    //A = new float[N][N];
    L = new float[N][N];
    U = new float[N][N];
    int i = 0, j = 0;


    std::cout << "Enter matrix values: " << std::endl;
    
    for (i = 0; i < N; i++)
        for (j = 0; j < N; j++)
            std::cin >> A[i][j];
    
    float A[N][N] = { {2,4,3,5},{-4,-7,-5,-8},{6,8,2,9},{4,9,-2,14} };
    LUdecompCUDA(A, L, U);
    std::cout << "L Decomposition is as follows..." << std::endl;
    print(L);
    std::cout << "U Decomposition is as follows..." << std::endl;
    print(U);

    //delete[] A;
    delete[] L;
    delete[] U;

    return 0;
}
*/