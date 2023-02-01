
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>

#include "Header_dense.cuh"

const int Arow = 5;
const int Acol = 10;

const int Brow = Acol;
const int Bcol = 10;

// Képezi egy adott C indexhez a szorzatösszeget
__host__ __device__ float productSum(float* A, float* B, int i, int j) {
    float sum = 0;
    for (int k = 0; k < Acol; k++)
        sum += A[i * Acol + k] * B[k * Bcol + j];
    return sum;
}

// Kernel
__global__ void MultiplKernel(float* A, float* B, float* C) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if(i < Arow && j < Bcol)
        C[i*Bcol + j] = productSum(A, B, i, j);
}

// CUDA függvény
void Multiplication(float* A, float* B, float* C) {
    // Dinamikus helyfoglalás
    float* pA, * pB, * pC;

    // Adathalmaz mérete
    int Abytes = (Arow * Acol) * sizeof(float);
    int Bbytes = (Brow * Bcol) * sizeof(float);
    int Cbytes = (Arow * Bcol) * sizeof(float);


    cudaMalloc((void**)&pA, Abytes);
    cudaMalloc((void**)&pB, Bbytes);
    cudaMalloc((void**)&pC, Cbytes);

    // Adatok másolása
    cudaMemcpy(pA, A, Abytes, cudaMemcpyHostToDevice);
    cudaMemcpy(pB, B, Bbytes, cudaMemcpyHostToDevice);
    cudaMemcpy(pC, C, Cbytes, cudaMemcpyHostToDevice);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid(my_ceil(Arow, dimBlock.x), my_ceil(Bcol, dimBlock.y));

    // Kernel hívás
    MultiplKernel <<< dimGrid, dimBlock >>> (pA, pB, pC);

    // Feldolgozott adat átvitele a GPU-ról
    cudaMemcpy(C, pC, Cbytes, cudaMemcpyDeviceToHost);

    // Ideiglenes adattárolók felszabadítása
    cudaFree(pA);
    cudaFree(pB);
    cudaFree(pC);

}


int main()
{
    float A[Arow * Acol];
    float B[Brow * Bcol];
    for (int i = 0; i < Arow; ++i)
        for (int j = 0; j < Acol; ++j)
            A[i * Acol + j] = 3;
    
    for (int i = 0; i < Brow; ++i)
        for (int j = 0; j < Bcol; ++j)
            B[i * Bcol + j] = 3;

    float C[Arow * Bcol];

    // Ezt így utólag belegondolva nemis kell megtenni
    for (int i = 0; i < Arow; ++i)
        for (int j = 0; j < Bcol; ++j)
            C[i * Bcol + j] = 0;

    Multiplication(A, B, C);
    print(C,Arow,Bcol);


    return 0;
}