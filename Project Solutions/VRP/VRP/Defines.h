#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cooperative_groups.h>
#include <iostream>


// Thread block size
#define BLOCK_SIZE 1024

// Define for used data types
#define DATATYPE double

///
/// CONTROL PANEL
///

// Number of threads = number of ants
int ants = 1024;

// Repetition constants
#define REPETITIONS 10
#define RANDOM_GENERATIONS 20
#define FOLLOWER_GENERATIONS 500

// Pheromone matrix constants
#define ALPHA 0.75  // Reduction ratio of previous pheromone value
#define REWARD_MULTIPLIER 100   // Rewart multiplier after finding a shortest path until then
#define INITIAL_PHEROMONE_VALUE 1000    // Initial value of elements in the Pheromone matrix

#define SERIALMAXTRIES 10    // Number of serial processes (for debug purposes)

// Struct for function call
// Naming convention:   firstSecond skalar and vector variables
//                      FirstSecond Matrices 
typedef struct {
    bool* foundRoute;
    unsigned int antNum;
    int size;
    int maxVehicles;
    DATATYPE* Dist;
    int* route;
    DATATYPE* Pheromone;
} VRP_AntCUDA_ParamTypedef;


typedef struct {
    DATATYPE* Dist;     // Cost function input
    DATATYPE* Pheromone;
    int* route;         // Sequence output
    bool* foundRoute;   // Existence output
    int size;        // Number of graph vertices
    int maxVehicles; // Maximum Number of Routes
    int* antRoute;      // Temp array
    int antNum;         // Number of ants
    curandState* state; // CURAND random state
} VRP_AntKernel_ParamTypedef;


// Variables allocated in global memory for communication between different thread blocks
typedef struct {
    bool* invalidInput;   // Variable used to detecting invalid input
    bool* isolatedVertex;  // Variable used to detecting isolated vertex (for optimization purposes)
    DATATYPE* averageDist;
    DATATYPE minRes;    // Minimal found Route distance
} VRP_AntKernel_Global_ParamTypedef;


typedef struct {
    // Repetition constants
    unsigned Repetitions;
    unsigned Random_Generations;
    unsigned Follower_Generations;
    int maxTryNumber;   // Follower ants use this to stop weighted roulette
    // Pheromone matrix constants
    float Alpha;
    float Reward_Multiplier;
    DATATYPE Initial_Pheromone_Value;
} VRP_AntKernel_Config_ParamTypedef;