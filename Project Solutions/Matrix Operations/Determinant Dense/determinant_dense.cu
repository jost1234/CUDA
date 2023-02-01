

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cooperative_groups.h>
#include <math.h>
#include <iostream>
#include <cassert>

#include "Header_dense.cuh" 

using namespace cooperative_groups;

// Számoljuk a sorcserék paritását: 
__device__ int global_sign;
// Ha bármikor csupa 0 oszlopot találunk, tudjuk, hogy 0 a determináns
__device__ bool global_fullZeroColoumn;

// Megkeresi az "első" nem nulla elemet az oszlopban
__device__ bool firstNotZero(float *Matrix, int k, int* idx) {
    int i;
    for (i = k + 1; i < N; ++i) {
        if (Matrix[i * N + k]) {
            *idx = i;
            return true;
        }
    }
    return false;
}

// Grid: >1x1
__global__ void detKernel_multiBlock(float *Matrix, float* det) {
    // Szinkronizációs változó a teljes griden belül
    grid_group grid = this_grid();
    int i = blockIdx.x * blockDim.x + threadIdx.x;  // oszlopváltozó
    int j = blockIdx.y * blockDim.y + threadIdx.y;  // sorváltozó

    int k;
    int idx;    // Sorcserénél használt változó
    float temp;
    if (!grid.is_valid())
        return;
    grid.sync();
    
    // Kezdeti érték adás a 0. thread által
    if (i == 0 && j == 0) {
        global_sign = 1;
        global_fullZeroColoumn = false;
    }

    for (k = 0; k < N - 1; ++k) {
        // Szinkronizálni mindig jó
        grid.sync();

        // Mi van akkor, amikor a vezérelem 0?
        if (Matrix[k * N + k] == 0) {
            // Keresünk másik sort, ahol nem 0 vezérelem van, különben det=0
            // Mindig csak az egyik szálon teszteljük a vezérelem 0 voltát, mert megosztott változót állítunk
            if (i == k && j == k) {

                global_fullZeroColoumn = !firstNotZero(Matrix, k, &idx);

                if (global_fullZeroColoumn) {
                    // Csupa 0 oszlopot találtunk
                    *det = 0;
                }
                global_sign = -global_sign;
            }
            // Bevárjuk a [k][k] threadet, hogy megfelelőre állítsa a változót
            grid.sync();
            // A többi threadet is értesítjük arról, ha kész vagyunk; értéket már nem kell állítaniuk
            if (global_fullZeroColoumn)
                return;
        }

    
    
    
    }
};

// Grid: 1x1
__global__ void detKernel_1Block(float *Matrix, float* det) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;  // oszlopváltozó
    int j = blockIdx.y * blockDim.y + threadIdx.y;  // sorváltozó
    if (i >= N || j >= N)
        return;
    int k;
    int idx;    // Sorcserénél használt változó
    float temp;

    __shared__ int sign;
    __shared__ bool fullZeroColoumn;

    // Kezdeti érték adás a 0. thread által
    if (i == 0 && j == 0) {
        sign = 1;
        fullZeroColoumn = false;
    }

    // Gauss elimináció
    for (k = 0; k < N - 1; ++k) {
        //a már nem kellő szálak kiléphetnek
        if (i < k || j < k)
            return;
        __syncthreads();

        // Mi van akkor, amikor a vezérelem 0?
        if (Matrix[k * N + k] == 0){
            // Keresünk másik sort, ahol nem 0 vezérelem van, különben det=0
            // Mindig csak az egyik szálon teszteljük a vezérelem 0 voltát, mert megosztott változót állítunk
            if (i == k && j == k) {

                fullZeroColoumn = !firstNotZero(Matrix, k, &idx);

                if (fullZeroColoumn) {
                    // Csupa 0 oszlopot találtunk
                    *det = 0;
                }
                sign = -sign;
            }

            // Bevárjuk a [k][k] threadet, hogy megfelelőre állítsa a változót
            __syncthreads();
            // A többi threadet is értesítjük arról, ha kész vagyunk; értéket már nem kell állítaniuk
            if (fullZeroColoumn)
                return;
        
            // Kicseréljük a  k. és idx. sort: ilyenkor a determináns a -1 -szeresére változik
        
            // 1 dimenziós párhuzam, mert vektorművelet
            __syncthreads();
            if (i == k && j == k) {

                // ha egyszer megértem hogy miért rossz akkor megjavítom és kicserélem
                // elvileg az if fejlécéből a j==k elhagyható
                //temp = Matrix[k * N + j];
                //Matrix[k * N + j] = Matrix[idx * N + j];
                //Matrix[idx * N + j] = temp;


                for (int l = 0; l < N; l++) {
                    temp = Matrix[k * N + l];
                    Matrix[k * N + l] = Matrix[idx * N + l];
                    Matrix[idx * N + l] = temp;
                }
            }
            // Sorcsere után újabb szinkronizáció
            __syncthreads();
        }

        // Nem nulla a vezérelem, kezdődhet a Gauss elimináció, a k-adik oszlopot felesleges kinullázni, többet nem kellenek
        if (i > k && j > k) // diagnosztika végett nem j>=k lehetséges
            Matrix[i * N + j] -= Matrix[i * N + k] / Matrix[k * N + k] * Matrix[k * N + j];
        }
    if (i == N - 1 && j == N - 1) {
        *det = sign;
        for (k = 0; k < N; ++k)
            *det *= Matrix[k * N + k];
    }
};

cudaError_t DeterminantWithCUDA(const float* Matrix, float* det) {
    cudaError_t cudaStatus;
    
    // Kiválasztjuk a GPU-t, multi-GPU rendszerben lényeges lehet.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Device pointerek
    float * d_Matrix, *d_det;

    // Adathalmaz mérete, amit lefoglalunk
    size_t bytes = N * N * sizeof(float);

    // Adatfoglalás
    cudaStatus = cudaMalloc((void**)&d_Matrix, bytes);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }
    cudaStatus = cudaMalloc((void**)&d_det, sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Adatok másolása
    cudaStatus = cudaMemcpy(d_Matrix, Matrix, bytes, cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // Kernel hívás
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);

    int dimGridx = my_ceil(N, dimBlock.x),
        dimGridy = my_ceil(N, dimBlock.y);


    dim3 dimGrid(dimGridx, dimGridy);
    if (dimGridx == 1 && dimGridy == 1) {
        printf("\nLefutott fuggveny: detKernel_1Block\n");
        detKernel_1Block <<< dimGrid, dimBlock >>> (d_Matrix, d_det);
    }
    else {
        // Kernel hívás esetén fontos, hogy <<<...>>> syntax helyett  
        // a cudaLaunchCooperativeKernel CUDA runtime launch API-t kell használni
        // vagy annak CUDA driver megfelelőjét

        // 1-be állítja a supportsCoopLaunch-t ha a művelet támogatott a device 0-n. 
        // Csak compute capability 6.0 felett 
        int dev = 0;
        int supportsCoopLaunch = 0;
        cudaDeviceGetAttribute(&supportsCoopLaunch, cudaDevAttrCooperativeLaunch, dev);
        if (supportsCoopLaunch != 1)
            throw std::runtime_error("Cooperative Launch is not supported on this machine configuration.");

        // launch
        void* kernelArgs[2] = { &d_Matrix, &d_det }; // add kernel args 
        printf("\nLefutott fuggveny: detKernel_multiBlock\n");
        cudaLaunchCooperativeKernel((void*)detKernel_multiBlock, dimGrid, dimBlock, kernelArgs);

    }

    // Hibakeresés kernel lauch közben
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "detKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }

    // cudaDeviceSynchronize vár arra, hogy befejeződjön a kernel, utána visszatér
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching detKernel!\n", cudaStatus);
        goto Error;
    }

    // Feldolgozott adat átvitele a GPU-ról
    cudaStatus = cudaMemcpy(det, d_det, sizeof(float), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    // Ideiglenes adattárolók felszabadítása
    cudaFree(d_Matrix);
    cudaFree(d_det);
    return cudaStatus;
};

int main(void) {
    // Különböző tesztek
    //float A[N * N] = { 1,2,3,4,-2, -5,6,7,8,4, 9,10,-11,12,1, 13,-14,-15,0,9, 20,-26,16,-17,25 };
    float det;
    //float A[N * N] = { 0,4,3,5,2, -4,7,-15,-8,0.5, 6,8,2,9,1, 4,9,-2,14,7, 1,2,3,-4,5 };
    // /*
    float A[N * N] = {
        1,4,3,5,7.3,2.9,4,-1.93,2,7,
        -4,-7,-15,-8,4.5,2.3,1.9,-13.5,0,3,
        6,8,2,9,12,-2.8,5.6,1.9,-4.2,3,
        4,9,-2,14,0.4,7.13,2.98,-5.73,9.81,6.3,
        -2,6,8,4.1,7.19,-9.3,-4.4,3.6,-14.5,3,
        3,2,6,7,-3,-1.15,3.32,-1.29,0.32,3.4,
        4.53,-3.58,4,-7,6.9,8.085,3.8,-5,-0.58,1.2,
        0.24,4.91,-3.57,3.14,1.2,-5,6.43,7.27,0,2.11,
        1.23,3.21,-4.24,-0.31,2.67,-2.51,4.4,-1,9,-14,
        5,6.2,-4.73,3.72,-2,0.4,-0.6,4.71,-2.67,3.1
    };
    // */
    if(DeterminantWithCUDA(A, &det) == cudaSuccess)
    std::cout << "Determinant: " << det << std::endl;
    // Csak azért roncsolja az eredeti mátrixot, hogy láthatóak legyenek az esetleges hibák
    //print(A, N, N);

}



