/*
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <device_functions.h>
#include <iostream>
#include <cstdlib>
#include <vector>


// Thread block size
#define BLOCK_SIZE 16

template <typename T>
class Matrix {
    int width;
    int height;
    std::vector<T> elements;
public:
    // Konstruktor
    __device__ __host__ Matrix() : width(0), height(0) {elements.clear() }
    __device__ __host__ Matrix(int pwidth, int pheight, const T& pelement)
        : width(pwidth), height(pheight)
    {
        elements.resize(width * height, pelement);
    }
    // Copy konstruktor
    __device__ __host__ Matrix(const Matrix& other)
        : width(other.width), height(other.height)
    {
        elements = other.elements;
    }

    // Destruktor
    __host__ __device__ ~Matrix() {
        elements.clear();
    }
     

    // Getterek és setterek
    __device__ __host__ int getWidth() const { return width; }
    __device__ __host__ int getHeight() const { return height; }
    __device__ __host__ std::vector<T>& getElements() { return elements; }

    // Egy mátrixelem mérete
    __device__ __host__ inline size_t Stride() const {
        return sizeof(T);
    }

    // Visszaad egy elemet
    __device__ __host__ float GetElement(int row, int col) const
    {
        return elements[row * Stride() + col];
    }

    // Beállít egy elemet
    __device__ __host__ void SetElement(int row, int col, float value)
    {
        elements[row * Stride() + col] = value;
    }

    // Visszaad BLOCK_SIZExBLOCK_SIZE almátrixot (Asub) amely
    // bal felsõ sarka a megadott indexen van
    //              VESZÉLYES
    __device__ Matrix GetSubMatrix(int row, int col) const
    {
        Matrix Sub(BLOCK_SIZE,BL)
        Sub.width = BLOCK_SIZE;
        Sub.height = BLOCK_SIZE;
        Sub.elements = &elements[Stride() * BLOCK_SIZE * row + BLOCK_SIZE * col];
        return Sub;
    }

    // Cuda copy létrehozása
    void CUDA_copy_of(const Matrix& original) {
        free(elements);
        width = original.width;
        height = original.height;
        size_t size = width * height * sizeof(T);
        cudaMalloc(&elements, size);
        cudaMemcpy(elements, original.elements, size, cudaMemcpyHostToDevice);
    }

    void CUDA_wo_copying_elements(const Matrix& original) {
        width = original.width; height = original.height;
        size_t size = width * height * sizeof(T);
        cudaMalloc(&elements, size);
    }

    // Diagnosztikai kiíró függvény
    void print() {
        for (int row = 0; row < height; row++)
        {
            for (int col = 0; col < width; col++)
                std::cout << GetElement(row, col) << " ";
            std::cout << std::endl;
        }
    }

};

// Kernel elõdeklaráció
__global__ void MatMulKernel(const Matrix<float>&, const Matrix<float>&, Matrix<float>&);

// Matrix multiplication - Host code
// Matrix dimensions are assumed to be multiples of BLOCK_SIZE
Matrix<float>* MatMul(const Matrix<float>& A, const Matrix<float>& B)
{
    if (A.getWidth() != B.getHeight()) {
        std::cout << "A matrixszorzas nem vegezheto el!" << std::endl;
        return nullptr;
    }
    Matrix<float>* C = new Matrix<float>(A.getHeight(), B.getWidth(), 0);
    // Load A and B to device memory

    Matrix<float> d_A;
    d_A.CUDA_copy_of(A);

    Matrix<float> d_B;
    d_B.CUDA_copy_of(B);

    // Allocate C in device memory
    Matrix<float> d_C;
    d_C.CUDA_wo_copying_elements(*C);
    

    // Invoke kernel
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid(B.getWidth() / dimBlock.x, A.getHeight() / dimBlock.y);
    MatMulKernel <<< dimGrid, dimBlock >>> (d_A, d_B, d_C);

    size_t size = C->getWidth() * C->getHeight() * C->Stride();
    // Read C from device memory
    cudaMemcpy(C->getElements(), d_C.getElements(), size, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_A.getElements());
    cudaFree(d_B.getElements());
    cudaFree(d_C.getElements());

    return C;
}

__global__ void MatMulKernel(const Matrix<float>& A, const Matrix<float>& B, Matrix<float>& C) {
    // Block row and column
    int blockRow = blockIdx.y;
    int blockCol = blockIdx.x;

    // Each thread block computes one sub-matrix Csub of C
    Matrix<float> Csub = C.GetSubMatrix(blockRow, blockCol);

    // Each thread computes one element of Csub
    // by accumulating results into Cvalue
    float Cvalue = 0;

    // Thread row and column within Csub
    int row = threadIdx.y;
    int col = threadIdx.x;

    // Loop over all the sub-matrices of A and B that are
    // required to compute Csub
    // Multiply each pair of sub-matrices together
    // and accumulate the results
    for (int m = 0; m < (A.getWidth() / BLOCK_SIZE); ++m) {

        // Get sub-matrix Asub of A
        Matrix<float> Asub = A.GetSubMatrix(blockRow, m);

        // Get sub-matrix Bsub of B
        Matrix<float> Bsub = B.GetSubMatrix(m, blockCol);

        // Shared memory used to store Asub and Bsub respectively
        __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
        __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

        // Load Asub and Bsub from device memory to shared memory
        // Each thread loads one element of each sub-matrix
        As[row][col] = Asub.GetElement(row, col);
        Bs[row][col] = Bsub.GetElement(row, col);

        // Synchronize to make sure the sub-matrices are loaded
        // before starting the computation
        __syncthreads();
        // Multiply Asub and Bsub together
        for (int e = 0; e < BLOCK_SIZE; ++e)
            Cvalue += As[row][e] * Bs[e][col];

        // Synchronize to make sure that the preceding
        // computation is done before loading two new
        // sub-matrices of A and B in the next iteration
        __syncthreads();
    }

    // Write Csub to device memory
    // Each thread writes one element
    Csub.SetElement(row, col, Cvalue);
}

int main(void) {
    Matrix<float> A(160, 160, 2);
    Matrix<float> B(160, 160, 3);
    Matrix<float>* C = MatMul(A,B);

    if(C != nullptr)
    C->print();

    delete(C);

    return 0;
}
*/