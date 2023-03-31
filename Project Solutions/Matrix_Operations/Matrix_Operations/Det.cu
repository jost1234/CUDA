#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cooperative_groups.h>
#include <math.h>
#include <iostream>
#include <cassert>

#include "Matrix_plusfunctions.cuh" 

using namespace cooperative_groups;


// Megkeresi az "első" nem nulla elemet az oszlopban
__device__ bool firstNotZero(float(*Matrix)[N], int k, int* idx) {
    int i;
    for (i = k + 1; i < N; ++i) {
        if (Matrix[i][k]) {
            *idx = i;
            return true;
        }
    }
    return false;
}

// Számoljuk a sorcserék paritását: 
__device__ int sign;
// Ha bármikor csupa 0 oszlopot találunk, tudjuk, hogy 0 a determináns
__device__ bool fullZeroColoumn;

__global__ void detKernel_multiBlock(float(*Matrix)[N], float* det) {
    // Szinkronizációs változó a teljes griden belül
    grid_group grid = this_grid();
    int i = blockIdx.x * blockDim.x + threadIdx.x;  // oszlopváltozó
    int j = blockIdx.y * blockDim.y + threadIdx.y;  // sorváltozó
    int tr = grid.thread_rank();
    //if (i >= N || j >= N)
    //    return;
    int k;
    int idx;    // Sorcserénél használt változó
    float temp;
    if (!grid.is_valid())
        return;
    grid.sync();

    if (i == 0 && j == 0) {
        sign = 1;
        fullZeroColoumn = false;

    }

    for (k = 0; k < N - 1; ++k) {

        //a már nem kellő szálak kiléphetnek
        //if (i < k || j < k)
        //    return;

        //grid = this_grid();
        grid.sync();

        // Mi van akkor, amikor a vezérelem 0?
        // Keresünk másik sort, ahol nem 0 vezérelem van, különben det=0
        // Mindig csak az egyik szálon teszteljük a vezérelem 0 voltát
        if (Matrix[k][k] == 0) {
            // Függvényhívás csak 1 szálon
            if (i == k && j == k) {

                fullZeroColoumn = !firstNotZero(Matrix, k, &idx);

                if (fullZeroColoumn) {
                    // Csupa 0 oszlopot találtunk
                    *det = 0;
                }
                sign = -sign;
            }
            grid.sync();
            if (fullZeroColoumn)
                return;
            if (k == 5)
                printf("thread=%d: for loop2 %d\n", tr, k);
            grid.sync();
            // A többi threadet is értesítjük arról, ha kész vagyunk; értéket már nem kell állítaniuk


            // Kicseréltük a két sort: ilyenkor a determináns a -1 -szeresére változik

            // 1 Dimenziós párhuzam, mert vektorművelet
            // !!! Helyett egy szálas mert valamiért nem akar működni a párhuzamos cserélés
            grid.sync();
            if (i == k && j == k) {


                //temp = Matrix[k][j];
                //Matrix[k][j] = Matrix[idx][j];
                //Matrix[idx][j] = temp;


                for (int l = 0; l < N; l++) {
                    temp = Matrix[k][l];
                    Matrix[k][l] = Matrix[idx][l];
                    Matrix[idx][l] = temp;
                }
            }
        }
        grid.sync();

        // Nem nulla a vezérelem, kezdődhet a Gauss elimináció, a k-adik oszlopot felesleges kinullázni, többet nem kellenek
        if (i > k && j > k && i < N && j < N) // diagnosztika végett nem j>=k lehetséges
            Matrix[i][j] -= Matrix[i][k] / Matrix[k][k] * Matrix[k][j];
    }


    if (i == N - 1 && j == N - 1) {
        *det = sign;
        for (k = 0; k < N; ++k)
            *det *= Matrix[k][k];
    }

}

__global__ void detKernel_1Block(float(*Matrix)[N], float* det) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;  // oszlopváltozó
    int j = blockIdx.y * blockDim.y + threadIdx.y;  // sorváltozó
    if (i >= N || j >= N)
        return;
    int k;
    int idx;    // Sorcserénél használt változó
    float temp;



    if (i == 0 && j == 0) {
        sign = 1;
        fullZeroColoumn = false;

    }


    for (k = 0; k < N - 1; ++k) {
        //a már nem kellő szálak kiléphetnek
        if (i < k || j < k)
            return;
        __syncthreads();
        // Mi van akkor, amikor a vezérelem 0?
        // Keresünk másik sort, ahol nem 0 vezérelem van, különben det=0
        // Mindig csak az egyik szálon teszteljük a vezérelem 0 voltát
        if (Matrix[k][k] == 0) {
            // Függvényhívás csak 1 szálon
            if (i == k && j == k) {

                fullZeroColoumn = !firstNotZero(Matrix, k, &idx);

                if (fullZeroColoumn) {
                    // Csupa 0 oszlopot találtunk
                    *det = 0;
                    return;
                }
                sign = -sign;
            }

            __syncthreads();
            // A többi threadet is értesítjük arról, ha kész vagyunk; értéket már nem kell állítaniuk
            if (fullZeroColoumn)
                return;

            // Kicseréltük a két sort: ilyenkor a determináns a -1 -szeresére változik

            // 1 Dimenziós párhuzam, mert vektorművelet
            // !!! Helyett egy szálas mert valamiért nem akar működni a párhuzamos cserélés
            __syncthreads();
            if (i == k && j == k) {


                //temp = Matrix[k][j];
                //Matrix[k][j] = Matrix[idx][j];
                //Matrix[idx][j] = temp;


                for (int l = 0; l < N; l++) {
                    temp = Matrix[k][l];
                    Matrix[k][l] = Matrix[idx][l];
                    Matrix[idx][l] = temp;
                }
            }
        }
        __syncthreads();

        // Nem nulla a vezérelem, kezdődhet a Gauss elimináció, a k-adik oszlopot felesleges kinullázni, többet nem kellenek
        if (i > k && j > k) // diagnosztika végett nem j>=k lehetséges
            Matrix[i][j] -= Matrix[i][k] / Matrix[k][k] * Matrix[k][j];
    }
    if (i == N - 1 && j == N - 1) {
        *det = sign;
        for (k = 0; k < N; ++k)
            *det *= Matrix[k][k];
    }
}



void detCUDA(float(*Matrix)[N], int n, float* det) {
    // Ideiglenes változó
    float(*d_Matrix)[N], * d_det;
    //float* d_temp;

    // Adathalmaz mérete, amit lefoglalunk
    size_t bytes = N * N * sizeof(float);

    // Adatfoglalás
    cudaMalloc((void**)&d_Matrix, bytes);

    cudaMalloc((void**)&d_det, sizeof(float));
    //cudaMalloc((void**)&d_temp, N * sizeof(float));
    // Adatok másolása
    cudaMemcpy(d_Matrix, Matrix, bytes, cudaMemcpyHostToDevice);

    // Kernel hívás
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);

    int dimGridx = my_ceil(N, dimBlock.x),
        dimGridy = my_ceil(N, dimBlock.y);


    dim3 dimGrid(dimGridx, dimGridy);
    if (dimGridx == 1 && dimGridy == 1)
        detKernel_1Block << < dimGrid, dimBlock >> > (d_Matrix, d_det);
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


    // Feldolgozott adat átvitele a GPU-ról
    cudaMemcpy(det, d_det, sizeof(float), cudaMemcpyDeviceToHost);
    //cudaMemcpy(Matrix, d_Matrix, bytes, cudaMemcpyDeviceToHost);

Error:
    // Ideiglenes adattárolók felszabadítása
    cudaFree(d_Matrix);
    cudaFree(d_det);

    return;
}

float determinant(float(*Matrix)[N], int n) {
    if (n == 1)
        return Matrix[0][0];
    else if (n == 2)
        return Matrix[0][0] * Matrix[1][1] - Matrix[0][1] * Matrix[1][0];

    float det;
    detCUDA(Matrix, n, &det);
    return det;
}



int main(void) {
    // Különböző tesztek
    //float A[N][N] = { {1,2,3,4,-2}, {-5,6,7,8,4}, {9,10,-11,12,1}, {13,-14,-15,0,9}, {20,-26,16,-17,25} };
    //float A[N][N] = { {0,4,3,5,2},{-4,7,-15,-8,0.5},{6,8,2,9,1},{4,9,-2,14,7},{1,2,3,-4,5} };
    // /*
    float A[N][N] = {
        {1,4,3,5,7.3,2.9,4,-1.93,2,7},
        {-4,-7,-15,-8,4.5,2.3,1.9,-13.5,0,3},
        {6,8,2,9,12,-2.8,5.6,1.9,-4.2,3},
        {4,9,-2,14,0.4,7.13,2.98,-5.73,9.81,6.3},
        {-2,6,8,4.1,7.19,-9.3,-4.4,3.6,-14.5,3},
        {3,2,6,7,-3,-1.15,3.32,-1.29,0.32,3.4},
        {4.53,-3.58,4,-7,6.9,8.085,3.8,-5,-0.58,1.2},
        {0.24,4.91,-3.57,3.14,1.2,-5,6.43,7.27,0,2.11},
        {1.23,3.21,-4.24,-0.31,2.67,-2.51,4.4,-1,9,-14},
        {5,6.2,-4.73,3.72,-2,0.4,-0.6,4.71,-2.67,3.1}
    };
    // */
    std::cout << "Determinant: " << determinant(A, N) << std::endl;
    // Csak azért roncsolja az eredeti mátrixot, hogy láthatóak legyenek az esetleges hibák
    //print(A);

}
