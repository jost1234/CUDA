#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cooperative_groups.h>
#include <curand_kernel.h>
#include <iostream>


// Thread block size
#define BLOCK_SIZE 1024

// Define for used data types
#define DATATYPE float

///
/// CONTROL PANEL
///

// Number of threads = number of ants
// Default value: 1024
int ants = 16384;

// Repetition constants
#define REPETITIONS 10
#define RANDOM_GENERATIONS 20
#define FOLLOWER_GENERATIONS 500

// Pheromone matrix constants
#define RHO 0.75  // Reduction ratio of previous pheromon value
#define REWARD_MULTIPLIER 10   // Reward multiplier after finding a shortest path until then
#define INITIAL_PHEROMONE_VALUE 1000    // Initial value of elements in the Pheromone matrix

#define SERIALMAXTRIES 10 // Number of serial processes (for debug purposes)

namespace TSP {

	// Struct for Main CUDA function call
	typedef struct {
		DATATYPE* Dist;
		DATATYPE* Pheromone;
		int* route;
		bool* foundRoute;
		int antNum;
		size_t size;
	} TSP_AntCUDA_ParamTypedef;


	// Struct for kernel call
	typedef struct {
		DATATYPE* Dist;     // Cost function input
		DATATYPE* Pheromone;
		int* route;         // Sequence output
		int size;        // Number of graph vertices
		bool* foundRoute;   // Existence output
		int* antRoute;      // Temp array
		int antNum;         // Number of ants
		curandState* state; // CURAND random state
	} TSP_AntKernel_ParamTypedef;


	// Variables allocated in global memory for communication between different thread blocks
	typedef struct {
		bool invalidInput;   // Variable used to detecting invalid input
		bool isolatedVertex;  // Variable used to detecting isolated vertex (for optimization purposes)
		DATATYPE averageDist;
		DATATYPE multiplicationConst; 
		DATATYPE minRes;    // Minimal found Route distance
	} TSP_AntKernel_Global_ParamTypedef;

	
	typedef struct {
		// Repetition constants
		unsigned Repetitions;
		unsigned Random_Generations;
		unsigned Follower_Generations;
		int maxTryNumber;   // Follower ants use this to stop weighted roulette
		// Pheromone matrix constants
		float Rho;
		float Reward_Multiplier;
		DATATYPE Initial_Pheromone_Value;
	} TSP_AntKernel_Config_ParamTypedef;
}