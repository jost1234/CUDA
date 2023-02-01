#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>
#include <cstdlib>
#include <ctime>

// Sajátértékszámításra alkalmas könyvtár
#include "cuSolverSp.h"
#include "Header.cuh"

cusolverEigMode_t a = CUSOLVER_EIG_MODE_NOVECTOR;
