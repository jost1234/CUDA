

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

// Megkeresi a vezérelem alatti első nem nulla elemet az oszlopban
__device__ bool firstNotZero(DATATYPE* Matrix, int size, int k, int* idx) 
{
    int i;
    for (i = k + 1; i < size; ++i) {
        if (Matrix[i * size + k]) {
            *idx = i;
            return true;
        }
    }
    return false;
}

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

void determinantCudaFree(DATATYPE* d_Matrix, DATATYPE* d_det) {
    if (d_Matrix != NULL) cudaFree(d_Matrix);
    if (d_det != NULL) cudaFree(d_det);
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
    DATATYPE* d_Matrix = NULL, *d_det = NULL;

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

    int dimGridx = my_ceil(size, dimBlock.x) ,
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
        cudaLaunchCooperativeKernel((void*)detKernel_multiBlock , dimGrid , dimBlock , kernelArgs);

    }

    // Hibakeresés kernel lauch közben
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr , "detKernel launch failed: %s\n" , cudaGetErrorString(cudaStatus));
        // Ideiglenes adattárolók felszabadítása
        determinantCudaFree(d_Matrix, d_det);
        return cudaStatus;
    }

    // cudaDeviceSynchronize vár arra, hogy befejeződjön a kernel, utána visszatér
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr , "cudaDeviceSynchronize returned error code %d after launching detKernel!\n" , cudaStatus);
        // Ideiglenes adattárolók felszabadítása
        determinantCudaFree(d_Matrix, d_det);
        return cudaStatus;
    }

    // Feldolgozott adat átvitele a GPU-ról
    cudaStatus = cudaMemcpy(det , d_det , sizeof(DATATYPE) , cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr , "cudaMemcpy failed!");
        // Ideiglenes adattárolók felszabadítása
        determinantCudaFree(d_Matrix, d_det);
        return cudaStatus;
    }

    //// For Debug purposes
    //cudaStatus = cudaMemcpy(Matrix, d_Matrix, bytes, cudaMemcpyDeviceToHost);
    //if (cudaStatus != cudaSuccess) {
    //    fprintf(stderr, "cudaMemcpy failed!");
    //    // Ideiglenes adattárolók felszabadítása
    //    determinantCudaFree(d_Matrix, d_det);
    //    return cudaStatus;
    //}

    // Ideiglenes adattárolók felszabadítása
    determinantCudaFree(d_Matrix, d_det);
    return cudaStatus;
};

int main(int argc, char* argv[]) {

    // Variables used for reading from txt file
    FILE* pFile;    // File pointer
    int fileNameIdx;
    bool foundMatrix = false;   // Error handling
    bool givenSolution = false;
    double solution;
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
            if (sscanf(argv[++i],"%lf", &solution) != 1) {
                fprintf(stderr, "Unable to read given solution!\n");
            }
            else {
                printf("Given solution : %f\n", solution);
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
    if (fscanf_s(pFile, "%d\n" , &size) != 1) {
        fprintf(stderr, "Unable to read Size!\n Make sure you have the right file syntax!\n");
        fclose(pFile);
        return -1;
    }


    DATATYPE* Matrix = (DATATYPE*)calloc(size * size , sizeof(DATATYPE));

    // Reading Matrix values
    for (int ii = 0; ii < size; ++ii) {
        double temp;

        for (int jj = 0; jj < size; ++jj) {
            if (fscanf_s(pFile, "%lf ", &temp) != 1) {
                fprintf(stderr, "Error reading file \"%s\" Matrix(%d %d)\n", argv[fileNameIdx], ii, jj);
                fclose(pFile);
                return -1;
            }
            Matrix[ii * size + jj] = temp;
        }
        //fscanf_s(pFile  "\n");
    }

    printf("Closing file \"%s\"!\n", argv[fileNameIdx]);
    if (fclose(pFile) != 0) {
        fprintf(stderr, "Unable to close file \"%s\"!\n", argv[fileNameIdx]);
        return -1;
    }
    print(Matrix, size, size);

    DATATYPE det;
    
    if (DeterminantWithCUDA(Matrix, size, &det) == cudaSuccess)
        printf( "Determinant: %.10e \n", det);
    // Csak azért roncsolja az eredeti mátrixot, hogy láthatóak legyenek az esetleges hibák
    //printf("After Gauss elimination:\n");
    //print(Matrix, size, size);

    if (givenSolution) {
        double erroruPercentage = abs(((double)det - solution) / solution) * 100000000;
        printf("Difference from correct solution: %.12f u%%\n", erroruPercentage);
    }

    getchar();
    free(Matrix);
}