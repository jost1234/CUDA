// Special CUDA API headers
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "curand.h"
#include "curand_kernel.h"

// Custom header containing Control Panel
#include "TSP_v3.cuh"

// General purpose headers
#include <iostream>
#include <stdbool.h>
#include <stdlib.h>
#include <assert.h>
#include <float.h>

// Main function
int main(int argc, char* argv[])
{
    // Variables used for reading from txt file
    FILE* pfile;    // File pointer
    int fileNameIdx;
    bool foundDistFile = false;   // Error handling
    bool foundRoute;
    int size;    // Number of graph vertices
    int i;  // Iterator
    srand(time(0)); // Need seeds for random solutions
    
    // Processing command line arguments
    for (i = 1; i < argc; ++i) 
    {  
        /// Distance file: REQUIRED
        // Command Line Syntax: ... --dist [file_name]
        if ((strcmp(argv[i], "-d") == 0) || (strcmp(argv[i], "--dist") == 0)) 
        {
            pfile = fopen(argv[++i], "r");
            if (pfile == NULL) {
                fprintf(stderr, "Unable to open file \"%s\"", argv[i]);
                return -1;
            }
            fileNameIdx = i;
            printf("Opening file \"%s\"!\n", argv[fileNameIdx]);
            foundDistFile = true;
        }

        /// Number of threads: OPTIONAL (default: 1024)
        // Command Line Syntax: ... --ants [number of ants]
        else if ((strcmp(argv[i], "--a") == 0) || (strcmp(argv[i], "--ants") == 0))
        {
            if (sscanf(argv[++i], "%d", &ants) != 1) {
                fprintf(stderr, "Unable to read ant number!\n");
            }
            else {
                printf("Given ant number : %d\n", ants);
            }
        }
    }

    // Checking required elements
    if (!foundDistFile) 
    {
        fprintf(stderr, "Please give a file in command line arguments to set the Distance Matrix!\n");
        fprintf(stderr, "Command Line Syntax:\n\t--dist [data_file].txt\n");
        fprintf(stderr, "File Syntax:\n\t[Number of Nodes]\n\tdist11, dist12, ...\n\tdist21 ... \n");
        return -1;
    }

    // File syntax : 1st row must contain graph size in decimal
    // Following rows: graph edge values separated with comma (,)
    if (fscanf_s(pfile, "%d \n", &size) == 0) {
        fprintf(stderr, "Unable to read Size!\n Make sure you have the right file syntax!\n");
        fprintf(stderr, "File Syntax:\n\t[Number of Nodes]\n\tdist11, dist12, ...\n\tdist21 ... \n");
        fclose(pfile);
        return -1;
    }

    // Distance matrix
    // Store type: adjacency matrix format
    float* Dist = (float*)calloc(size * size, sizeof(float));

    // Reading distance values from dist file
    for (int ii = 0; ii < size; ++ii) {
        float temp;

        for (int jj = 0; jj < size; ++jj) {
            if (fscanf_s(pfile, "%f", &temp) == 0) {
                fprintf(stderr, "Error reading file \"%s\" distance(%d,%d)\n", argv[fileNameIdx], ii, jj);
                fclose(pfile);
                return -1;
            }
            Dist[ii * size + jj] = temp;
        }
        fscanf_s(pfile, "\n");
    }

    // Closing dist file
    printf("Closing file \"%s\"!\n", argv[fileNameIdx]);
    if (fclose(pfile) != 0) {
        fprintf(stderr, "Unable to close file \"%s\"!\n", argv[fileNameIdx]);
        return -1;
    }

    // Printing Matrix
    printf("Given Dist matrix:\n");
    print(Dist, size);

    // Host Variables

    TSP::CUDA_Main_ParamTypedef params;
    params.foundRoute = &foundRoute;
    params.antNum = ants;
    params.size = size;
}