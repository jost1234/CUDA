#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <cstdlib>
#include <ctime>


// Diagnosztika, parameterek
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

// Megkeresi az "els�" nem nulla elemet az oszlopban
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
    int i = blockIdx.x * N + threadIdx.x;  // oszlopv�ltoz�
    int j = blockIdx.y * N + threadIdx.y;  // sorv�ltoz�
    if (i >= N || j >= N)
        return;
    int k;
    int idx;    // Sorcser�n�l haszn�lt v�ltoz�
    float temp;


    // Sz�moljuk a sorcser�k parit�s�t: 
    __shared__ int sign;
    // Ha b�rmikor csupa 0 oszlopot tal�lunk, tudjuk, hogy 0 a determin�ns
    __shared__ bool fullZeroColoumn;
    if (i == 0 && j == 0) {
        sign = 1;
        fullZeroColoumn = false;

    }


    for (k = 0; k < N - 1; ++k) {
        //a m�r nem kell� sz�lak kil�phetnek
        if (i < k || j < k)
            return;
        __syncthreads();
        // Mi van akkor, amikor a vez�relem 0?
        // Keres�nk m�sik sort, ahol nem 0 vez�relem van, k�l�nben det=0
        // Mindig csak az egyik sz�lon tesztelj�k a vez�relem 0 volt�t
        if (Matrix[k][k] == 0) {
            // F�ggv�nyh�v�s csak 1 sz�lon
            if (i == k && j == k) {

                fullZeroColoumn = !firstNotZero(Matrix, k, &idx);

                if (fullZeroColoumn) {
                    // Csupa 0 oszlopot tal�ltunk
                    *det = 0;
                    return;
                }
                sign = -sign;
            }

            __syncthreads();
            // A t�bbi threadet is �rtes�tj�k arr�l, ha k�sz vagyunk; �rt�ket m�r nem kell �ll�taniuk
            if (fullZeroColoumn)
                return;

            // Kicser�lt�k a k�t sort: ilyenkor a determin�ns a -1 -szeres�re v�ltozik

            // 1 Dimenzi�s p�rhuzam, mert vektorm�velet
            // !!! Helyett egy sz�las mert valami�rt nem akar m�k�dni a p�rhuzamos cser�l�s
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

        // Nem nulla a vez�relem, kezd�dhet a Gauss elimin�ci�, a k-adik oszlopot felesleges kinull�zni, t�bbet nem kellenek
        if (i > k && j > k) // diagnosztika v�gett nem j>=k lehets�ges
            Matrix[i][j] -= Matrix[i][k] / Matrix[k][k] * Matrix[k][j];
    }
    if (i == N - 1 && j == N - 1) {
        *det = sign;
        for (k = 0; k < N; ++k)
            *det *= Matrix[k][k];
    }
}

float detCUDA(float(*Matrix)[N], int n) {
    // Ideiglenes v�ltoz�
    float(*d_Matrix)[N], * d_det;
    //float* d_temp;

    // Adathalmaz m�rete, amit lefoglalunk
    size_t bytes = N * N * sizeof(float);

    // Adatfoglal�s
    cudaMalloc((void**)&d_Matrix, bytes);

    cudaMalloc((void**)&d_det, sizeof(float));
    //cudaMalloc((void**)&d_temp, N * sizeof(float));
    // Adatok m�sol�sa
    cudaMemcpy(d_Matrix, Matrix, bytes, cudaMemcpyHostToDevice);


    // Kernel h�v�s
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid(ceil(N / (float)dimBlock.x), ceil(N / (float)dimBlock.y));
    detKernel <<< dimGrid, dimBlock >>> (d_Matrix, d_det);

    // Kinyert adat
    float det;
    // Feldolgozott adat �tvitele a GPU-r�l
    cudaMemcpy(&det, d_det, sizeof(float), cudaMemcpyDeviceToHost);
    //cudaMemcpy(Matrix, d_Matrix, bytes, cudaMemcpyDeviceToHost);


    // Ideiglenes adatt�rol�k felszabad�t�sa
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

    int i = blockIdx.x * blockDim.x + threadIdx.x;  // oszlopv�ltoz�
    int j = blockIdx.y * blockDim.y + threadIdx.y;  // sorv�ltoz�
    //int k = blockIdx.z * blockDim.z + threadIdx.z;
    if (i >= N || j >= N)
        return;

    // El�form�z�s
    U[i][j] = A[i][j];
    if (i < j)
        L[i][j] = 0;
    else if (i == j)
        L[i][i] = 1;

    __syncthreads();
    
    //Gauss elimin�ci� u m�trixban
    for (int m = 0; m < N; m++) {
        if (U[m][m] == 0) {
            cudaError_t error = cudaErrorUnknown;
            //exit(error);
            return;
        }
        float x = U[i][m] / U[m][m];    // Ez soronk�nt ugyanaz
        if (j == m && i > m) {
            // Ha a null�z�d� oszlopban vagyunk
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
    int i = blockIdx.x * blockDim.x + threadIdx.x;  // oszlopv�ltoz�
    int j = blockIdx.y * blockDim.y + threadIdx.y;  // sorv�ltoz�
    if (i >=N || j >= N)
        return;
    int k = 0;
    int l = 0;

    Z[i][j] = 0;
    __syncthreads();
    

    // 1 dimenzi�s p�rhuzamos�t�s (szinkroniz�l�sok mindenkire vonatkoznak)
    // N db teljesen f�ggetlen m�veletv�gz�s Matrix[][i] oszlopvektorokon
    
    // 1. l�p�s: Z (oszlopvektor!! az�rt van i j ford�tva) meghat�roz�sa
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
    // Elk�sz�lt Z

    // 2. l�p�s: Ainv[][i] oszlopvektor meghat�roz�sa, m�dszer backpropagation
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

// Soros v�ltozat
__global__ void inversionKernel1(float(*A)[N], float(*A_inv)[N], float(*L)[N], float(*U)[N], float(*Z)[N]) {
    int ii = blockIdx.x * blockDim.x + threadIdx.x;  // oszlopv�ltoz�
    int jj = blockIdx.y * blockDim.y + threadIdx.y;  // sorv�ltoz�
    //int k = blockIdx.z * blockDim.z + threadIdx.z;
    if (ii > 0 || jj > 0)   // teszt egysz�lon
        return;
    int i = 0; //oszlop
    int j = 0; //sor
    int k = 0;
    //Adott L �s U
    // Kell Z: L*Z = (0 0 ... 0 1 0 0...), 1-es el�tt i db 0
    // Ezut�n U*Ainv[i] = Z 

    for (i = 0; i < N; i++)
        for (j = 0; j < N; j++) 
            Z[i][j] = 0;    // Mi�rt ne, ez �gyis csak teszt
            
        
    

    for (i = 0; i < N; i++) {
        // N db teljesen f�ggetlen m�veletv�gz�s Matrix[][i] oszlopvektorokon
        
        // 1. l�p�s: Z (oszlopvektor!! az�rt van i j ford�tva) meghat�roz�sa
        for (j = 0; j < i; j++) 
            Z[j][i] = 0;
        
        Z[i][i] = 1;
        for (j = i + 1; j < N; j++) {
            Z[j][i] = 0;
            for (k = 0; k < j; k++)
                Z[j][i] -= Z[k][i] * L[j][k];
        }

        // 2. l�p�s: Ainv[][i] oszlopvektor meghat�roz�sa, m�dszer backpropagation
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
    // Dinamikus helyfoglal�s a GPU-ra: haszn�lt pointerek
    float (*d_A)[N], (*d_A_inv)[N], (*d_L)[N], (*d_U)[N];
    float(*d_Z)[N]; // InverseKernelbe sz�ks�ges seg�dm�trix, ott van kifejtve funkci�ja
    
    // Adathalmaz m�rete, amit lefoglalunk
    size_t bytes = N * N * sizeof(float);
    
    // Adatfoglal�s
    cudaMalloc((void**)&d_A, bytes);
    cudaMalloc((void**)&d_A_inv, bytes);
    cudaMalloc((void**)&d_L, bytes);
    cudaMalloc((void**)&d_U, bytes);

    cudaMalloc((void**)&d_Z, bytes);
    
    
    // Adatok m�sol�sa
    cudaMemcpy(d_A, A, bytes, cudaMemcpyHostToDevice);
    
    // Kernel grid, blokk m�ret
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid(my_ceil(N, dimBlock.x), my_ceil(N, dimBlock.y));
    
    // LU faktoriz�ci� kernel h�v�s (ideiglenes adatt�rol�kon)
    LUdecompKernel <<<dimGrid, dimBlock >>> (d_A, d_L, d_U);
    
    // Inverzsz�m�t�s kernel (ideiglenes adatt�rol�kon
    inversionKernel <<<dimGrid, dimBlock >>> (d_A, d_A_inv, d_L, d_U, d_Z);
    

    // Feldolgozott adat �tvitele a GPU-r�l
    cudaMemcpy(A_inv, d_A_inv, bytes, cudaMemcpyDeviceToHost);
    // Ideiglenes adatt�rol�k felszabad�t�sa
    
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
    
    // N=4 pr�ba
    //float A[N][N] = { {2,4,3,5},{-4,-7,-15,-8},{6,8,2,9},{4,9,-2,14} };
    // N=10 pr�ba
    float A[N][N] = { 
        {2,4,3,5,7.3,2.9,4,-1.93,2,7},
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