
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>

#define BLOCK_SIZE 16;
const int N = 5;

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


__global__ void child(int(*c)[N], const int(*a)[N], const int(*b)[N], int i) {
    int j = threadIdx.x;
    if (j < N)
        c[i][j] = a[i][j] + b[i][j];
}

__global__ void parent(int (* c)[N], const int (* a)[N], const int (* b)[N])
{
    int i = threadIdx.x;
    if(i<N)
    child <<<1,BLOCK_SIZE>>> (c, a, b, i);
}

int main()
{
    const int a[N][N] = { {1,2,3,4,5},{6,7,8,9,10},{11,12,13,14,15},{16,17,18,19,20},{21,22,23,24,25} };
    const int b[N][N] = { {1,2,3,4,5},{6,7,8,9,10},{11,12,13,14,15},{16,17,18,19,20},{21,22,23,24,25} };
    int c[N][N] = { {0},{0},{0},{0},{0} };

    // Cuda pointerek
    int(*d_a)[N], (*d_b)[N], (*d_c)[N];
    size_t bytes = N * N * sizeof(int);
    cudaMalloc((void**)d_a, bytes);
    cudaMalloc((void**)d_b, bytes);
    cudaMalloc((void**)d_c, bytes);

    // Adatok másolása
    cudaMemcpy(d_a, a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, bytes, cudaMemcpyHostToDevice);


    // Kernel hívás
    parent <<<1, BLOCK_SIZE>>>(d_a,d_b,d_c);

    // Feldolgozott adat átvitele a GPU-ról
    cudaMemcpy(c, d_c, bytes, cudaMemcpyDeviceToHost);

    print(c);

    // Ideiglenes adattárolók felszabadítása
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_b);


    return 0;
}

