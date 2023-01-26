#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>
#include <stdlib.h>
#include <time.h>

// Thread block size
#define BLOCK_SIZE 16
const int N = 32;



__host__ __device__ float productSum(float A[][N], float B[][N], int i, int j) {
    float sum = 0;
       for (int k = 0; k < N; k++)
            sum += A[i][k] * B[k][j];
    return sum;
}

// Egyszálú megoldás
void MxMultipl_serial(float A[][N], float B[][N], float C[][N]) {
    int i, j;
    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            C[i][j] = productSum(A, B, i, j);
        }
    }
}

// Kiírja egy mátrix elemeit
void MxPrint(float M[][N]) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++)
            std::cout << M[i][j] << " ";
        std::cout << std::endl;
    }
    std::cout << std::endl;
}


// Kernel
__global__ void MultiplKernel(float A[][N], float B[][N], float C[][N]) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < N && j < N)

    // Lehetséges újabb párhuzamosítás?
    C[i][j] = productSum(A, B, i, j);

    // Ez azért nem jó, mert mindegyik szál 0-ról ad hozzá, elég mókás (nem atomi műveletvégrehajtás)
    // syncthreaddel megoldható, de az újra tönkre teszi az O(1)-et
    //C[i][j] += A[i][k] * B[k][j];


}

// CUDA főfüggvény
void MxMultipl_parallel(float (*A)[N], float (*B)[N], float (*C)[N]) {

    // Dinamikus helyfoglalás a GPU-ra
    float (*pA)[N], (*pB)[N], (*pC)[N];

    // Adathalmaz mérete
    int bytes = (N * N) * sizeof(float);

    cudaMalloc((void**)&pA, bytes);
    cudaMalloc((void**)&pB, bytes);
    cudaMalloc((void**)&pC, bytes);


    // Adatok másolása
    cudaMemcpy(pA, A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(pB, B, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(pC, C, bytes, cudaMemcpyHostToDevice);

    // Kernel invocation with one block of N * N * 1 threads
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid(ceil(N / dimBlock.x), ceil(N / dimBlock.y));
    MultiplKernel<<<dimGrid,dimBlock>>>(pA, pB, pC);


    // Feldolgozott adat átvitele a GPU-ról
    cudaMemcpy(C, pC, bytes, cudaMemcpyDeviceToHost);

    // Ideiglenes adattárolók felszabadítása
    cudaFree(pA);
    cudaFree(pB);
    cudaFree(pC);
}

int main()
{
    //srand(time(0));
    float A[N][N];
    float B[N][N];
    for (int i = 0; i < N; ++i)
    {
        for (int j = 0; j < N; ++j)
        {
            A[i][j] = 3;
            B[i][j] = 3;
        }
    }
    //float C_serial[N][N];
    float C_parallel[N][N];
    for (int i = 0; i < N; ++i)
    {
        for (int j = 0; j < N; ++j)
        {
            //C_serial[i][j] = 0;
            C_parallel[i][j] = 0;
        }
    }

    //MxMultipl_serial(A, B, C_serial);
    //MxPrint(C_serial);

    MxMultipl_parallel(A, B, C_parallel);
    MxPrint(C_parallel);


    return 0;
}
