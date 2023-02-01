#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>

// Thread block size
#define BLOCK_SIZE 4
const int N = 13;

// Hogy felismerje a precompiler
#ifdef __INTELLISENSE__
void __syncthreads();
#endif

//print the matrix.
template<class T>
void print(float* A, int Row, int Col) {
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