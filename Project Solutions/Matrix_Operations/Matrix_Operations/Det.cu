#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cooperative_groups.h>
#include <math.h>
#include <iostream>
#include <cassert>

#include "Matrix_plusfunctions.cuh" 

using namespace cooperative_groups;


// Megkeresi az "első" nem nulla elemet az oszlopban
__device__ bool firstNotZero(DATATYPE(*Matrix)[N], int k, int* idx) {
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

__global__ void detKernel_multiBlock(DATATYPE(*Matrix)[N], DATATYPE* det) {
    // Szinkronizációs változó a teljes griden belül
    grid_group grid = this_grid();
    int i = blockIdx.x * blockDim.x + threadIdx.x;  // oszlopváltozó
    int j = blockIdx.y * blockDim.y + threadIdx.y;  // sorváltozó
    int tr = grid.thread_rank();
    //if (i >= N || j >= N)
    //    return;
    int k;
    int idx;    // Sorcserénél használt változó
    DATATYPE temp;
    if (!grid.is_valid())
        return;
    grid.sync();

    if (i == 0 && j == 0) {
        sign = 1;
        fullZeroColoumn = false;

    }

    for (k = 0; k < N - 1; ++k) {

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
            grid.sync();
            // A többi threadet is értesítjük arról, ha kész vagyunk; értéket már nem kell állítaniuk


            // Kicseréltük a két sort: ilyenkor a determináns a -1 -szeresére változik

            // 1 Dimenziós párhuzam, mert vektorművelet
            // !!! Helyett egy szálas mert valamiért nem akar működni a párhuzamos cserélés
            grid.sync();
            if (i == k && j == k) {    
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

__global__ void detKernel_1Block(DATATYPE(*Matrix)[N], DATATYPE* det) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;  // oszlopváltozó
    int j = blockIdx.y * blockDim.y + threadIdx.y;  // sorváltozó
    if (i >= N || j >= N)
        return;
    int k;
    int idx;    // Sorcserénél használt változó
    DATATYPE temp;



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

void determinantCudaFree(DATATYPE (*d_Matrix)[N], DATATYPE* d_det) {
    if (d_Matrix != NULL) cudaFree(d_Matrix);
    if (d_det != NULL) cudaFree(d_det);
}

cudaError_t DeterminantWithCUDA(DATATYPE(*Matrix)[N], int size, DATATYPE* det) {
    cudaError_t cudaStatus;
    
    // Kiválasztjuk a GPU-t, multi-GPU rendszerben lényeges lehet.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        return cudaStatus;
    }

    // Ideiglenes változó
    DATATYPE(*d_Matrix)[N] = NULL, * d_det = NULL;
    //DATATYPE* d_temp;

    // Adathalmaz mérete, amit lefoglalunk
    size_t bytes = size * size * sizeof(DATATYPE);

    // Adatfoglalás
    cudaMalloc((void**)&d_Matrix, bytes);

    cudaMalloc((void**)&d_det, sizeof(DATATYPE));

    // Adatok másolása
    cudaMemcpy(d_Matrix, Matrix, bytes, cudaMemcpyHostToDevice);

    // Kernel hívás
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);

    int dimGridx = my_ceil(size, dimBlock.x),
        dimGridy = my_ceil(size, dimBlock.y);


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
        {
            fprintf(stderr, "Cooperative Launch is not supported on this machine configuration.");
            // Ideiglenes adattárolók felszabadítása
            determinantCudaFree(d_Matrix, d_det);
            return cudaStatus;
        }
        // launch
        void* kernelArgs[2] = { &d_Matrix, &d_det }; // add kernel args 
        printf("\nLefutott fuggveny: detKernel_multiBlock\n");
        cudaLaunchCooperativeKernel((void*)detKernel_multiBlock, dimGrid, dimBlock, kernelArgs);


    }

    // Hibakeresés kernel lauch közben
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) 
    {
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
}


int main(int argc, char* argv[]) {

    DATATYPE Matrix[N][N];

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

    // Reading Matrix values
    for (int ii = 0; ii < size; ++ii) {
        double temp;

        for (int jj = 0; jj < size; ++jj) {
            if (fscanf_s(pFile, "%lf ", &temp) != 1) {
                fprintf(stderr, "Error reading file \"%s\" Matrix(%d %d)\n", argv[fileNameIdx], ii, jj);
                fclose(pFile);
                return -1;
            }
            Matrix[ii][jj] = temp;
        }
        //fscanf_s(pFile  "\n");
    }

    printf("Closing file \"%s\"!\n", argv[fileNameIdx]);
    if (fclose(pFile) != 0) {
        fprintf(stderr, "Unable to close file \"%s\"!\n", argv[fileNameIdx]);
        return -1;
    }
    print(Matrix);

    DATATYPE det;
    
    if (DeterminantWithCUDA(Matrix, size, &det) == cudaSuccess)
        printf( "Determinant: %.10e \n", det);
    // Csak azért roncsolja az eredeti mátrixot, hogy láthatóak legyenek az esetleges hibák
    //printf("After Gauss elimination:\n");
    //print(Matrix, size, size);

    if (givenSolution) {
        double errorPercentage = abs(((double)det - solution) / solution) * 100000000;
        printf("Difference from correct solution: %.12f u%%\n", errorPercentage);
    }

    getchar();
}
