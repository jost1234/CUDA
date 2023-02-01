
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include <cusolverDn.h>

/*
*	Elmélet:	
*		A = {{5,5},{-1,7}}
*		Cél : A = U * Σ * (V^T)	
* 
*		A^T * A = V * Σ^T * Σ * V^T
*		A * V = U * Σ
* 
*		A^T * A = {{5,-1},{5,7}} * {{5,5},{-1,7}} = {{26,18},{18,74}}
* 
*		det(A^T*A-λ*I) = det({{26-λ,18},{18,74-λ}})	= λ^2 - 100λ + 1600 = (λ-20) * (λ-80)
* 
*		λ1 = 20 : {{6,18},{18,54}} => v1 = {-3, 1} * 1/sqrt(10)
*		λ2 = 80 : {{-54,18},{18,-6}} => v2 = {1,3} * 1/sqrt(10)
* 
*		V = (v1, v2) = {{-3/sqrt10 , 1/sqrt10} , {1/sqrt10, 3/sqrt10}}
*		Σ = {{sqrt(λ1),0},{0,sqrt(λ2)}} = {{2*sqrt5,0},{0,4*sqrt5}}
*		U = A * V * Σ^-1
*/

const int m = 3;
const int lda = m;

int main(void) {
	cusolverDnHandle_t cusolverH = NULL;
	cusolverStatus_t cusolver_status = CUSOLVER_STATUS_SUCCESS;
	cudaError_t cudaStat1 = cudaSuccess;
	cudaError_t cudaStat2 = cudaSuccess;  
	cudaError_t cudaStat3 = cudaSuccess;
	const int m = 3;
	const int lda = m;
	//		| 3.5 0.5 0 |
	// A =	| 0.5 3.5 0 |
	//		| 0	  0   2 |
	double A[lda * m] = { 3.5, 0.5, 0, 0.5, 3.5, 0, 0, 0, 2.0 };
	double lambda[m] = { 2.0, 3.0, 4.0 };
	double U[lda * m];
	double Sig[lda * m];
	double V[lda * m];
	double* d_A = NULL;

	return 0;
}