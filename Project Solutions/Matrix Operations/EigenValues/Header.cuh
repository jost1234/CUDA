#pragma once

// Thread block size
#define BLOCK_SIZE 10
const int N = 10;

// Hogy felismerje a precompiler
#ifdef __INTELLISENSE__
void __syncthreads();
#endif

//print the matrix.
template<class T>
void print(T A[N][N]) {
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
            std::cout << A[i][j] << " ";
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

__device__ __host__ int my_ceil(int osztando, int oszto) {
    if (!(osztando % oszto)) return osztando / oszto;
    else return	osztando / oszto + 1;
}

template <class T>
__global__ void transposeKernel(const T Matrix[N][N], T TranposedMatrix[N][N]) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < N && j < N && i != j)
        TransposedMatrix[i][j] = Matrix[j][i];
}