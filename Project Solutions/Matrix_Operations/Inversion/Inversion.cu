
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cooperative_groups.h>
#include <math.h>
#include <iostream>
#include <cassert>
#
#include "Header.cuh"

using namespace cooperative_groups;

int main(int argc, char* argv[]) {
    // Variables used for reading from txt file
    FILE* pFile;    // File pointer
    int fileNameIdx;
    bool foundMatrix = false;   // Error handling

    FILE* pSolutionFile = NULL;
    bool givenSolution = false;
    double* Solution;
    

    int size;    // Number of matrix rows (=matrix coloumns)

    int i;  // Iterator
    for (i = 1; i < argc; ++i)  // Processing command line arguments
    {
        // Command Line Syntax: ... --m [file_name]
        if ((strcmp(argv[i], "--m") == 0) || (strcmp(argv[i], "--Matrix") == 0))
        {
            pFile = fopen(argv[++i], "r");
            if (pFile == NULL) {
                fprintf(stderr, "Unable to open file \"%s\"", argv[i]);
                return -1;
            }
            fileNameIdx = i;
            printf("Opening file \"%s\"!\n", argv[fileNameIdx]);
            foundMatrix = true;
        }
        // Command Line Syntax: ... --s [Solution]
        if ((strcmp(argv[i], "--s") == 0) || (strcmp(argv[i], "--Solution") == 0))
        {
            pSolutionFile = fopen(argv[++i], "r");
            if (pSolutionFile == NULL)
            {
                fprintf(stderr, "Unable to open file \"%s\"", argv[i]);
                return -1;
            }
            else 
            {
                givenSolution = true;
            }


        }
    }
    if (!foundMatrix) {
        fprintf(stderr, "Please give a file in command line arguments to set the input Matrix!\n");
        fprintf(stderr, "Command Line Syntax:\n\t--Matrix [data_file].txt\n");
        fprintf(stderr, "File Syntax:\n\t[Number of rows]\n\tM11  M12  ...\n\tM21 ... \n");
        return -1;
    }


    // Process data
    if (fscanf_s(pFile, "%d\n", &size) != 1) {
        fprintf(stderr, "Unable to read Size!\n Make sure you have the right file syntax!\n");
        fclose(pFile);
        if (givenSolution) fclose(pSolutionFile);
        return -1;
    }


    DATATYPE* Matrix = (DATATYPE*)malloc(size * size * sizeof(DATATYPE));
    DATATYPE* InvMatrix = (DATATYPE*)malloc(size * size * sizeof(DATATYPE));

    // Reading Matrix values
    for (int ii = 0; ii < size; ++ii) 
    {
        double temp;

        for (int jj = 0; jj < size; ++jj) 
        {
            if (fscanf_s(pFile, "%lf ", &temp) != 1) 
            {
                fprintf(stderr, "Error reading file \"%s\" Matrix(%d %d)\n", argv[fileNameIdx], ii, jj);
                fclose(pFile);
                return -1;
            }
            Matrix[ii * size + jj] = temp;
        }
        //fscanf_s(pFile  "\n");
    }

    // Reading solution values
    // Following syntax of matlab txt file generating
    if (givenSolution)
    {
        Solution = (double*)malloc(size * size * sizeof(double));
        for (int ii = 0; ii < size; ++ii)
        {
            double temp;

            for (int jj = 0; jj < size-1; ++jj)
            {
                if (fscanf_s(pSolutionFile, "%lf,", &temp) != 1)
                {
                    fprintf(stderr, "Error reading SolutionMatrix(%d %d)\n", ii, jj);
                    fclose(pFile);
                    fclose(pSolutionFile);
                    return -1;
                }
                Solution[ii * size + jj] = temp;
            }
            if (fscanf_s(pSolutionFile, "%lf\n", &temp) != 1)
            {
                fprintf(stderr, "Error reading SolutionMatrix(%d %d)\n", ii, size-1);
                fclose(pFile);
                fclose(pSolutionFile);
                return -1;
            }
            Solution[ii * size + size-1] = temp;
        }
        fclose(pSolutionFile);
    }

    printf("Closing file \"%s\"!\n", argv[fileNameIdx]);
    if (fclose(pFile) != 0) {
        fprintf(stderr, "Unable to close file \"%s\"!\n", argv[fileNameIdx]);
        return -1;
    }
    
    
    printf("Entry matrix values:\n");
    print(Matrix, size, size);

    // Inversion
    inversionCUDA(Matrix, size, InvMatrix);

    printf("\nInverse matrix values:\n");
    print(InvMatrix, size, size);

    // Testing the punctuality of the results
    if (givenSolution)
    {
        double sumError = 0.0;
        double maxError = 0.0;

        for (i = 0; i < size * size; i++)
        {
            double erroruPercentage = abs(((double)InvMatrix[i] - Solution[i]) / Solution[i]) * 100000000;
            sumError += erroruPercentage;
            if (erroruPercentage > maxError)
                maxError = erroruPercentage;
        }
        sumError /= size * size;
        printf("Average difference from correct solution: %.12f u%%\n", sumError);
        printf("Max difference from correct solution: %.12f u%%\n", maxError);
    }

    getchar();
    free(Matrix);
    free(InvMatrix);

}

__host__ cudaError_t inversionCUDA(DATATYPE* Matrix, int size, DATATYPE* InvMatrix) {

    DATATYPE det;
    cudaError_t cudaStatus = DeterminantWithCUDA(Matrix, size, &det);
    if (cudaStatus != cudaSuccess)
    {
        return cudaStatus;
    }
    if (det == 0)
    {
        printf("Non-invertable matrix!\n");
        return cudaSuccess;
    }

    // Device pointerek
    DATATYPE *d_Matrix = NULL, *d_InvMatrix = NULL;
    DATATYPE* d_L = NULL; DATATYPE* d_U = NULL; DATATYPE* d_Z = NULL;

    // Adathalmaz mérete, amit lefoglalunk
    size_t bytes = size * size * sizeof(DATATYPE);

    // Adatfoglalás
    cudaStatus = cudaMalloc((void**)&d_Matrix, bytes);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        // Ideiglenes adattárolók felszabadítása
        inversionCudaFree(d_Matrix, d_InvMatrix, d_L, d_U, d_Z);
        return cudaStatus;
    }
    cudaStatus = cudaMalloc((void**)&d_InvMatrix, bytes);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        // Ideiglenes adattárolók felszabadítása
        inversionCudaFree(d_Matrix, d_InvMatrix, d_L, d_U, d_Z);
        return cudaStatus;
    }
    cudaStatus = cudaMalloc((void**)&d_L, bytes);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        // Ideiglenes adattárolók felszabadítása
        inversionCudaFree(d_Matrix, d_InvMatrix, d_L, d_U, d_Z);
        return cudaStatus;
    }
    cudaStatus = cudaMalloc((void**)&d_U, bytes);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        // Ideiglenes adattárolók felszabadítása
        inversionCudaFree(d_Matrix, d_InvMatrix, d_L, d_U, d_Z);
        return cudaStatus;
    }
    cudaStatus = cudaMalloc((void**)&d_Z, bytes);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        // Ideiglenes adattárolók felszabadítása
        inversionCudaFree(d_Matrix, d_InvMatrix, d_L, d_U, d_Z);
        return cudaStatus;
    }

    // Adatok másolása
    cudaStatus = cudaMemcpy(d_Matrix, Matrix, bytes, cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        // Ideiglenes adattárolók felszabadítása
        inversionCudaFree(d_Matrix, d_InvMatrix, d_L, d_U, d_Z);
        return cudaStatus;
    }

    // Kernel hívás
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);

    int dimGridx = my_ceil(size, dimBlock.x),
        dimGridy = my_ceil(size, dimBlock.y);


    dim3 dimGrid(dimGridx, dimGridy);
    if (dimGridx == 1 && dimGridy == 1) {
        printf("\nLefutott fuggveny: inversionKernel_1Block\n");
        inversionKernel_1Block <<< dimGrid, dimBlock >>> (d_Matrix, d_InvMatrix, d_L, d_U, d_Z, size);
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
        if (supportsCoopLaunch != 1) {
            fprintf(stderr, "Cooperative Launch is not supported on this machine configuration.");
            // Ideiglenes adattárolók felszabadítása
            inversionCudaFree(d_Matrix, d_InvMatrix, d_L, d_U, d_Z);
            return cudaStatus;
        }

        // launch
        void* kernelArgs[] = { &d_Matrix, &d_InvMatrix, &d_L, &d_U, &d_Z, &size }; // add kernel args 
        printf("\nLefutott fuggveny: inversionKernel_multiBlock\n");
        cudaLaunchCooperativeKernel((void*)inversionKernel_multiBlock, dimGrid, dimBlock, kernelArgs);

    }

    // Hibakeresés kernel lauch közben
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "detKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        // Ideiglenes adattárolók felszabadítása
        inversionCudaFree(d_Matrix, d_InvMatrix, d_L, d_U, d_Z);
        return cudaStatus;
    }

    // cudaDeviceSynchronize vár arra, hogy befejeződjön a kernel, utána visszatér
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching detKernel!\n", cudaStatus);
        // Ideiglenes adattárolók felszabadítása
        inversionCudaFree(d_Matrix, d_InvMatrix, d_L, d_U, d_Z);
        return cudaStatus;
    }

    // Feldolgozott adat átvitele a GPU-ról
    cudaStatus = cudaMemcpy(InvMatrix, d_InvMatrix, bytes, cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        // Ideiglenes adattárolók felszabadítása
        inversionCudaFree(d_Matrix, d_InvMatrix, d_L, d_U, d_Z);
        return cudaStatus;
    }

    // Ideiglenes adattárolók felszabadítása
    inversionCudaFree(d_Matrix, d_InvMatrix, d_L, d_U, d_Z);
    return cudaStatus;
}

void inversionCudaFree(DATATYPE* d_Matrix, DATATYPE* d_InvMatrix, DATATYPE* d_L, DATATYPE* d_U, DATATYPE* d_Z) {
    if (d_Matrix != NULL) cudaFree(d_Matrix);
    if (d_InvMatrix != NULL) cudaFree(d_InvMatrix);
    if (d_L != NULL) cudaFree(d_L);
    if (d_U != NULL) cudaFree(d_U);
    if (d_Z != NULL) cudaFree(d_Z);

}

__global__ void inversionKernel_1Block(DATATYPE* Matrix, DATATYPE* InvMatrix, 
                                       DATATYPE* L, DATATYPE* U, DATATYPE* Z, int size)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;  // oszlopváltozó
    int j = blockIdx.y * blockDim.y + threadIdx.y;  // sorváltozó
    if (i >= size || j >= size)
        return;

    thread_block block = this_thread_block();

    // 1. lepes: LU dekompozíció

    // Előformázás
    U[i * size + j] = Matrix[i * size + j];
    if (i < j)
        L[i * size + j] = 0;
    else if (i == j)
        L[i * size + i] = 1;

    block.sync();

    // Gauss elimináció az U mátrixban
    for (int m = 0; m < size; m++)
    {
        if (U[m * size + m] == 0)
        {
            if (i == 0 && j == 0) 
            {
                printf("Inversion not calculatable with LU Decomposition!\n"); 
            }
            return;
        }
        DATATYPE x = U[i * size + m] / U[m * size + m];    // Ez soronként ugyanaz
        if (j == m && i > m)
        {
            // Ha a null�z�d� oszlopban vagyunk
            U[i * size + j] = 0;
            L[i * size + j] = x;
        }
        else if (i > m && j > m) {
            U[i * size + j] -= x * U[m * size + j];
        }
        block.sync();
    }

    // Felbontottuk a mátrixot egy felsőháromszög-, és egy alsóháromszögmátrixra
    int k;
    int l;

    Z[i * size + j] = 0;
    block.sync();

    // 1 dimenzios párhuzamosítás (szinkronizálások mindenkire vonatkoznak)
    // size db teljesen független müveletvégzés Matrix[][i] oszlopvektorokon

    // 2. lepes: = (oszlopvektor, azert van i j forditva) meghatarozasa
    if (j == 0)
    {
        Z[i * size + i] = 1;
    }
    block.sync();

    if (j == 0)
    {
        for (k = i + 1; k < size; k++)
        {
            for (l = 0; l < k; l++)
            {
                Z[k * size + i] -= Z[l * size + i] * L[k * size + l];
            }
        }
    }
    block.sync();

    // Elkeszult Z

    // 3. lepes: InvMatrix[][i] oszlopvektor meghat., módszer backpropagation
    InvMatrix[j * size + i] = Z[j * size + i];
    block.sync();
    if (j == 0)
    {
        for (k = size - 1; k >= 0; k--)
        {
            for (l = k + 1; l < size; l++)
            {
                InvMatrix[k * size + i] -= U[k * size + l] * InvMatrix[l * size + i];
            }
            InvMatrix[k * size + i] /= U[k * size + k];
        }
    }
}

__global__ void inversionKernel_multiBlock(DATATYPE* Matrix, DATATYPE* InvMatrix,
    DATATYPE* L, DATATYPE* U, DATATYPE* Z, int size)
{
    // Szinkronizációs változó a teljes griden belül
    // Synchronization variable across the whole grid
    grid_group grid = this_grid();
    if (!grid.is_valid())
        return;
    grid.sync();
    int i = blockIdx.x * blockDim.x + threadIdx.x;  // oszlopváltozó
    int j = blockIdx.y * blockDim.y + threadIdx.y;  // sorváltozó

    // 1. lepes: LU dekompozíció

    // Előformázás
    if (i < size && j < size)
    {
        U[i * size + j] = Matrix[i * size + j];
        if (i < j)
            L[i * size + j] = 0;
        else if (i == j)
            L[i * size + i] = 1;
    }
    grid.sync();

    // Gauss elimináció az U mátrixban
    for (int m = 0; m < size; m++)
    {
        if (U[m * size + m] == 0)
        {
            if (i == 0 && j == 0)
            {
                printf("Inversion not calculatable with LU Decomposition!\n");
            }
            return;
        }
        if (i < size && j < size)
        {
            DATATYPE x = U[i * size + m] / U[m * size + m];    // Ez soronként ugyanaz
            if (j == m && i > m)
            {
                // Ha a null�z�d� oszlopban vagyunk
                U[i * size + j] = 0;
                L[i * size + j] = x;
            }
            else if (i > m && j > m) {
                U[i * size + j] -= x * U[m * size + j];
            }
        }
        grid.sync();
    }

    // Felbontottuk a mátrixot egy felsőháromszög-, és egy alsóháromszögmátrixra
    int k;
    int l;
    if (i < size && j < size)
        Z[i * size + j] = 0;
    grid.sync();

    // 1 dimenzios párhuzamosítás (szinkronizálások mindenkire vonatkoznak)
    // size db teljesen független müveletvégzés Matrix[][i] oszlopvektorokon

    // 2. lepes: = (oszlopvektor, azert van i j forditva) meghatarozasa
    if (j == 0 && i < size)
    {
        Z[i * size + i] = 1;
    }
    grid.sync();

    if (j == 0 && i < size)
    {
        for (k = i + 1; k < size; k++)
        {
            for (l = 0; l < k; l++)
            {
                Z[k * size + i] -= Z[l * size + i] * L[k * size + l];
            }
        }
    }
    grid.sync();

    // Elkeszult Z

    // 3. lepes: InvMatrix[][i] oszlopvektor meghat., módszer backpropagation
    if (i < size && j < size)
    {
        InvMatrix[j * size + i] = Z[j * size + i];
    }
    grid.sync();
    if (j == 0 && i < size)
    {
        for (k = size - 1; k >= 0; k--)
        {
            for (l = k + 1; l < size; l++)
            {
                InvMatrix[k * size + i] -= U[k * size + l] * InvMatrix[l * size + i];
            }
            InvMatrix[k * size + i] /= U[k * size + k];
        }
    }
}


// Megkeresi a vezérelem alatti első nem nulla elemet az oszlopban
__device__ bool firstNotZero(DATATYPE* Matrix, int size, int k, int* idx) {
    int i;
    for (i = k + 1; i < size; ++i) {
        if (Matrix[i * size + k]) {
            *idx = i;
            return true;
        }
    }
    return false;
}


cudaError_t DeterminantWithCUDA(DATATYPE* Matrix, int size, DATATYPE* det) {
    cudaError_t cudaStatus;

    // Kiválasztjuk a GPU-t, multi-GPU rendszerben lényeges lehet.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        return cudaStatus;
    }

    // Device pointerek
    DATATYPE* d_Matrix = NULL, * d_det = NULL;

    // Adathalmaz mérete, amit lefoglalunk
    size_t bytes = size * size * sizeof(DATATYPE);

    // Adatfoglalás
    cudaStatus = cudaMalloc((void**)&d_Matrix, bytes);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        // Ideiglenes adattárolók felszabadítása
        determinantCudaFree(d_Matrix, d_det);
        return cudaStatus;
    }
    cudaStatus = cudaMalloc((void**)&d_det, sizeof(DATATYPE));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        // Ideiglenes adattárolók felszabadítása
        determinantCudaFree(d_Matrix, d_det);
        return cudaStatus;
    }

    // Adatok másolása
    cudaStatus = cudaMemcpy(d_Matrix, Matrix, bytes, cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        // Ideiglenes adattárolók felszabadítása
        determinantCudaFree(d_Matrix, d_det);
        return cudaStatus;
    }

    // Kernel hívás
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);

    int dimGridx = my_ceil(size, dimBlock.x),
        dimGridy = my_ceil(size, dimBlock.y);


    dim3 dimGrid(dimGridx, dimGridy);
    if (dimGridx == 1 && dimGridy == 1) {
        printf("\nLefutott fuggveny: detKernel_1Block\n");
        detKernel_1Block <<< dimGrid, dimBlock >>> (d_Matrix, size, d_det);
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
        if (supportsCoopLaunch != 1) {
            fprintf(stderr, "Cooperative Launch is not supported on this machine configuration.");
            // Ideiglenes adattárolók felszabadítása
            determinantCudaFree(d_Matrix, d_det);
            return cudaStatus;
        }

        // launch
        void* kernelArgs[] = { &d_Matrix, &size, &d_det }; // add kernel args 
        printf("\nLefutott fuggveny: detKernel_multiBlock\n");
        cudaLaunchCooperativeKernel((void*)detKernel_multiBlock, dimGrid, dimBlock, kernelArgs);

    }

    // Hibakeresés kernel lauch közben
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "detKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        // Ideiglenes adattárolók felszabadítása
        determinantCudaFree(d_Matrix, d_det);
        return cudaStatus;
    }

    // cudaDeviceSynchronize vár arra, hogy befejeződjön a kernel, utána visszatér
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching detKernel!\n", cudaStatus);
        // Ideiglenes adattárolók felszabadítása
        determinantCudaFree(d_Matrix, d_det);
        return cudaStatus;
    }

    // Feldolgozott adat átvitele a GPU-ról
    cudaStatus = cudaMemcpy(det, d_det, sizeof(DATATYPE), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        // Ideiglenes adattárolók felszabadítása
        determinantCudaFree(d_Matrix, d_det);
        return cudaStatus;
    }

    // Ideiglenes adattárolók felszabadítása
    determinantCudaFree(d_Matrix, d_det);
    return cudaStatus;
};

void determinantCudaFree(DATATYPE* d_Matrix,DATATYPE* d_det) {
    if (d_Matrix != NULL) cudaFree(d_Matrix);
    if (d_det != NULL) cudaFree(d_det);
}

// Grid: 1x1
__global__ void detKernel_1Block(DATATYPE* Matrix, int size, DATATYPE* det)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;  // oszlopváltozó
    int j = blockIdx.y * blockDim.y + threadIdx.y;  // sorváltozó
    if (i >= size || j >= size)
        return;
    int k;
    int idx;    // Sorcserénél használt változó
    DATATYPE temp1, temp2;

    __shared__ int sign;
    __shared__ bool fullZeroColoumn;

    // Kezdeti érték adás a 0. thread által
    if (i == 0 && j == 0) {
        sign = 1;
        fullZeroColoumn = false;
    }

    // Gauss elimináció
    for (k = 0; k < size - 1; ++k) {
        //a már nem kellő szálak kiléphetnek
        if (i < k || j < k)
            return;
        __syncthreads();

        // Mi van akkor  amikor a vezérelem 0?
        if (Matrix[k * size + k] == 0) {
            // Keresünk másik sort  ahol nem 0 vezérelem van  különben det=0
            // Mindig csak az egyik szálon teszteljük a vezérelem 0 voltát  mert megosztott változót állítunk
            if (i == k && j == k)
            {
                fullZeroColoumn = !firstNotZero(Matrix, size, k, &idx);

                if (fullZeroColoumn) {
                    // Csupa 0 oszlopot találtunk
                    *det = 0;
                }
                sign = -sign;
            }

            // Bevárjuk a [k][k] threadet  hogy megfelelőre állítsa a változót
            __syncthreads();
            // A többi threadet is értesítjük arról  ha kész vagyunk; értéket már nem kell állítaniuk
            if (fullZeroColoumn)
                return;

            // Kicseréljük a  k. és idx. sort: ilyenkor a determináns a -1 -szeresére változik

            // 1 dimenziós párhuzam  mert vektorművelet
            __syncthreads();
            if (i == k) {
                temp1 = Matrix[k * size + j];
                temp2 = Matrix[idx * size + j];
                Matrix[k * size + j] = temp2;
                Matrix[idx * size + j] = temp1;
            }
            // Sorcsere után újabb szinkronizáció
            __syncthreads();
        }

        // Nem nulla a vezérelem, kezdődhet a Gauss elimináció,
        // a k-adik oszlopot felesleges kinullázni, többet nem kellenek
        if (i > k && j > k)
            Matrix[i * size + j] -= Matrix[i * size + k] / Matrix[k * size + k] * Matrix[k * size + j];
    }
    // Deternináns a főátlóbeli elemek szorzata alapján
    if (i == size - 1 && j == size - 1)
    {
        *det = sign;
        for (k = 0; k < size; ++k)
            *det *= Matrix[k * size + k];
    }
};

// Számoljuk a sorcserék paritását: 
__device__ int global_sign;
// Ha bármikor csupa 0 oszlopot találunk, tudjuk, hogy 0 a determináns
__device__ bool global_fullZeroColoumn;

// Grid: >1x1
__global__ void detKernel_multiBlock(DATATYPE* Matrix, int size, DATATYPE* det) {
    // Szinkronizációs változó a teljes griden belül
    cooperative_groups::grid_group grid = cooperative_groups::this_grid();
    int i = blockIdx.x * blockDim.x + threadIdx.x;  // oszlopváltozó
    int j = blockIdx.y * blockDim.y + threadIdx.y;  // sorváltozó

    int k;
    int idx;    // Sorcserénél használt változó
    DATATYPE temp1, temp2;
    if (!grid.is_valid())
        return;
    grid.sync();

    // Kezdeti érték adás a 0. thread által
    if (i == 0 && j == 0) {
        global_sign = 1;
        global_fullZeroColoumn = false;
    }

    for (k = 0; k < size - 1; ++k) {
        grid.sync();

        // Mi van akkor  amikor a vezérelem 0?
        if (Matrix[k * size + k] == 0) {
            // Keresünk másik sort, ahol nem 0 vezérelem van  különben det=0
            // Mindig csak az egyik szálon teszteljük a vezérelem 0 voltát, mert megosztott változót állítunk
            if (i == k && j == k) {

                global_fullZeroColoumn = !firstNotZero(Matrix, size, k, &idx);

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

            grid.sync();
            if (i == k) {
                temp1 = Matrix[k * size + j];
                temp2 = Matrix[idx * size + j];
                Matrix[k * size + j] = temp2;
                Matrix[idx * size + j] = temp1;
            }
            // Sorcsere után újabb szinkronizáció
            grid.sync();
        }
        // Nem nulla a vezérelem, kezdődhet a Gauss elimináció 
        // a k-adik oszlopot felesleges kinullázni, többet nem kellenek
        if (i > k && j > k && i < size && j < size) // diagnosztika végett nem j>=k lehetséges
            Matrix[i * size + j] -= Matrix[i * size + k] / Matrix[k * size + k] * Matrix[k * size + j];
    }
    if (i == size - 1 && j == size - 1) {
        *det = global_sign;
        for (k = 0; k < size; ++k)
            *det *= Matrix[k * size + k];

    }

};

