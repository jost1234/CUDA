#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "stdio.h"


// Thread block size
#define BLOCK_SIZE 32

#define DATATYPE float

// Hogy felismerje a precompiler
#ifdef __INTELLISENSE__
void __syncthreads();
#endif

////print the matrix.
//void print(DATATYPE* A, int Row, int Col) {
//    for (int i = 0; i < Row; i++)
//    {
//        for (int j = 0; j < Col; j++)
//            printf("%.2f ", A[i * Col + j]);
//        printf("\n");
//    }
//    printf("\n");
//}

void print(DATATYPE* A, int Row, int Col) {
    for (int i = 0; i < Row; i++)
    {
        for (int j = 0; j < Col; j++)
            std::cout << A[i * Col + j] << " ";
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

__device__ __host__ int my_ceil(int osztando, int oszto) {
    if (!(osztando % oszto)) return osztando / oszto;
    else return	osztando / oszto + 1;
}

__host__ cudaError_t inversionCUDA(DATATYPE* Matrix, int size, DATATYPE* InvMatrix);

void inversionCudaFree(DATATYPE* d_Matrix, DATATYPE* d_InvMatrix, DATATYPE* d_L, DATATYPE* d_U, DATATYPE* d_Z);

// Megoldás LU dekompozicióval
__global__ void inversionKernel_1Block(DATATYPE* Matrix, DATATYPE* InvMatrix, DATATYPE* L, DATATYPE* U, DATATYPE* Z, int size);

__global__ void inversionKernel_multiBlock(DATATYPE* Matrix, DATATYPE* InvMatrix, DATATYPE* L, DATATYPE* U, DATATYPE* Z, int size);

void determinantCudaFree(DATATYPE* d_Matrix, DATATYPE* d_det);

cudaError_t DeterminantWithCUDA(DATATYPE* Matrix, int size, DATATYPE* det);


// Megkeresi a vezérelem alatti elsõ nem nulla elemet az oszlopban
__device__ bool firstNotZero(DATATYPE* Matrix, int size, int k, int* idx);

// Grid: 1x1
__global__ void detKernel_1Block(DATATYPE* Matrix, int size, DATATYPE* det);

// Grid: >1x1
__global__ void detKernel_multiBlock(DATATYPE* Matrix, int size, DATATYPE* det);