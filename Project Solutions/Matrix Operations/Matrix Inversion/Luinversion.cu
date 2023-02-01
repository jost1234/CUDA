#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <cstdlib>
#include <ctime>


// Diagnosztika, paraméterek
//#include "Matrix_plusfunctions.cuh"

// Thread block size
#define BLOCK_SIZE 5
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

// Megkeresi az "elsõ" nem nulla elemet az oszlopban
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

__global__ void detKernel(float(*Matrix)[N], float* det) {
    int i = blockIdx.x * N + threadIdx.x;  // oszlopváltozó
    int j = blockIdx.y * N + threadIdx.y;  // sorváltozó
    if (i >= N || j >= N)
        return;
    int k;
    int idx;    // Sorcserénél használt változó
    float temp;


    // Számoljuk a sorcserék paritását: 
    __shared__ int sign;
    // Ha bármikor csupa 0 oszlopot találunk, tudjuk, hogy 0 a determináns
    __shared__ bool fullZeroColoumn;
    if (i == 0 && j == 0) {
        sign = 1;
        fullZeroColoumn = false;

    }


    for (k = 0; k < N - 1; ++k) {
        //a már nem kellõ szálak kiléphetnek
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

            // 1 Dimenziós párhuzam, mert vektormûvelet
            // !!! Helyett egy szálas mert valamiért nem akar mûködni a párhuzamos cserélés
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

        // Nem nulla a vezérelem, kezdõdhet a Gauss elimináció, a k-adik oszlopot felesleges kinullázni, többet nem kellenek
        if (i > k && j > k) // diagnosztika végett nem j>=k lehetséges
            Matrix[i][j] -= Matrix[i][k] / Matrix[k][k] * Matrix[k][j];
    }
    if (i == N - 1 && j == N - 1) {
        *det = sign;
        for (k = 0; k < N; ++k)
            *det *= Matrix[k][k];
    }
}

float detCUDA(float(*Matrix)[N], int n) {
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
    dim3 dimGrid(ceil(N / (float)dimBlock.x), ceil(N / (float)dimBlock.y));
    detKernel <<< dimGrid, dimBlock >>> (d_Matrix, d_det);

    // Kinyert adat
    float det;
    // Feldolgozott adat átvitele a GPU-ról
    cudaMemcpy(&det, d_det, sizeof(float), cudaMemcpyDeviceToHost);
    //cudaMemcpy(Matrix, d_Matrix, bytes, cudaMemcpyDeviceToHost);


    // Ideiglenes adattárolók felszabadítása
    cudaFree(d_Matrix);
    cudaFree(d_det);

    return det;
}

float determinant(float(*Matrix)[N], int n) {
    if (n == 1)
        return Matrix[0][0];
    else if (n == 2)
        return Matrix[0][0] * Matrix[1][1] - Matrix[0][1] * Matrix[1][0];

    float det = detCUDA(Matrix, n);
    return det;
}

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

__global__ void inversionKernel(float(*A)[N], float(*A_inv)[N], float(*L)[N], float(*U)[N], float(*Z)[N]) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;  // oszlopváltozó
    int j = blockIdx.y * blockDim.y + threadIdx.y;  // sorváltozó
    if (i >=N || j >= N)
        return;
    int k = 0;
    int l = 0;

    Z[i][j] = 0;
    __syncthreads();
    

    // 1 dimenziós párhuzamosítás (szinkronizálások mindenkire vonatkoznak)
    // N db teljesen független mûveletvégzés Matrix[][i] oszlopvektorokon
    
    // 1. lépés: Z (oszlopvektor!! azért van i j fordítva) meghatározása
    if (j == 0) {
        Z[i][i] = 1;
    }
    __syncthreads();
    if (j == 0) {
        for(k=i+1;k<N;k++)
            for(l=0;l<k;l++)
            Z[k][i] -= Z[l][i] * L[k][l];
    }
    __syncthreads();
    // Elkészült Z

    // 2. lépés: Ainv[][i] oszlopvektor meghatározása, módszer backpropagation
    A_inv[j][i] = Z[j][i];
    __syncthreads();
    if (j == 0) {
        for (k = N - 1; k >= 0; k--) {
            
            for (l = k + 1; l < N; l++)
                A_inv[k][i] -= U[k][l] * A_inv[l][i];
            A_inv[k][i] /= U[k][k];
        }
    }
}

// Soros változat
__global__ void inversionKernel1(float(*A)[N], float(*A_inv)[N], float(*L)[N], float(*U)[N], float(*Z)[N]) {
    int ii = blockIdx.x * blockDim.x + threadIdx.x;  // oszlopváltozó
    int jj = blockIdx.y * blockDim.y + threadIdx.y;  // sorváltozó
    //int k = blockIdx.z * blockDim.z + threadIdx.z;
    if (ii > 0 || jj > 0)   // teszt egyszálon
        return;
    int i = 0; //oszlop
    int j = 0; //sor
    int k = 0;
    //Adott L és U
    // Kell Z: L*Z = (0 0 ... 0 1 0 0...), 1-es elõtt i db 0
    // Ezután U*Ainv[i] = Z 

    for (i = 0; i < N; i++)
        for (j = 0; j < N; j++) 
            Z[i][j] = 0;    // Miért ne, ez úgyis csak teszt
            
        
    

    for (i = 0; i < N; i++) {
        // N db teljesen független mûveletvégzés Matrix[][i] oszlopvektorokon
        
        // 1. lépés: Z (oszlopvektor!! azért van i j fordítva) meghatározása
        for (j = 0; j < i; j++) 
            Z[j][i] = 0;
        
        Z[i][i] = 1;
        for (j = i + 1; j < N; j++) {
            Z[j][i] = 0;
            for (k = 0; k < j; k++)
                Z[j][i] -= Z[k][i] * L[j][k];
        }

        // 2. lépés: Ainv[][i] oszlopvektor meghatározása, módszer backpropagation
        for (j = N-1; j >= 0; j--) {
            A_inv[j][i] = Z[j][i];
            for (k = j+1; k < N; k++)
                A_inv[j][i] -= U[j][k] * A_inv[k][i];
            A_inv[j][i] /= U[j][j];
        }
    }
}


__host__ void inversionCUDA(float(*A)[N], float(*A_inv)[N]) {

    int DET = determinant(A, N);
    if (DET == 0)
    {
        std::cout << "Non-invertable matrix!";
        return;
    }
    // Dinamikus helyfoglalás a GPU-ra: használt pointerek
    float (*d_A)[N], (*d_A_inv)[N], (*d_L)[N], (*d_U)[N];
    float(*d_Z)[N]; // InverseKernelbe szükséges segédmátrix, ott van kifejtve funkciója
    
    // Adathalmaz mérete, amit lefoglalunk
    size_t bytes = N * N * sizeof(float);
    
    // Adatfoglalás
    cudaMalloc((void**)&d_A, bytes);
    cudaMalloc((void**)&d_A_inv, bytes);
    cudaMalloc((void**)&d_L, bytes);
    cudaMalloc((void**)&d_U, bytes);

    cudaMalloc((void**)&d_Z, bytes);
    
    
    // Adatok másolása
    cudaMemcpy(d_A, A, bytes, cudaMemcpyHostToDevice);
    
    // Kernel grid, blokk méret
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid(my_ceil(N, dimBlock.x), my_ceil(N, dimBlock.y));
    
    // LU faktorizáció kernel hívás (ideiglenes adattárolókon)
    LUdecompKernel <<<dimGrid, dimBlock >>> (d_A, d_L, d_U);
    
    // Inverzszámítás kernel (ideiglenes adattárolókon
    inversionKernel <<<dimGrid, dimBlock >>> (d_A, d_A_inv, d_L, d_U, d_Z);
    

    // Feldolgozott adat átvitele a GPU-ról
    cudaMemcpy(A_inv, d_A_inv, bytes, cudaMemcpyDeviceToHost);
    // Ideiglenes adattárolók felszabadítása
    
    cudaFree(d_A);
    cudaFree(d_A_inv);
    cudaFree(d_L);
    cudaFree(d_U);
    cudaFree(d_Z);

}


int main() {
    float(*L)[N], (*U)[N];
    //A = new float[N][N];
    L = new float[N][N];
    U = new float[N][N];
    int i = 0, j = 0;


    std::cout << "Entry matrix values: " << std::endl;
    
    // N=4 próba
    //float A[N][N] = { {2,4,3,5},{-4,-7,-15,-8},{6,8,2,9},{4,9,-2,14} };
    // N=10 próba
    float A[N][N] = { 
        {2,4,3,5,7.3,2.9,4,
        -1.93,2,7},
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
    print(A);
    //std::cout << determinant(A, N);
    float Ainv[N][N];
    inversionCUDA(A, Ainv);
    std::cout << "Inverse matrix values: " << std::endl;
    print(Ainv);

    //delete[] A;
    delete[] L;
    delete[] U;

    return 0;
}