
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>

// Cooperative groups includes
#include "cooperative_groups.h"
#include "cooperative_groups/scan.h"
#include "cooperative_groups/reduce.h"
#include "cooperative_groups/memcpy_async.h"
#include "cooperative_groups/details/async.h"
#include "cooperative_groups/details/coalesced_reduce.h"
#include "cooperative_groups/details/coalesced_scan.h"
#include "cooperative_groups/details/driver_abi.h"


using namespace cooperative_groups;
__device__ int reduce_sum(thread_group g, int* temp, int val)
{
    int lane = g.thread_rank();
    this_thread_block().sync();
    // Each iteration halves the number of active threads
    // Each thread adds its partial sum[i] to sum[lane+i]
    for (int i = g.size() / 2; i > 0; i /= 2)
    {
        temp[lane] = val;
        g.sync(); // wait for all threads to store
        if (lane < i) val += temp[lane + i];
        g.sync(); // wait for all threads to load
    }
    return val; // note: only thread 0 will return full sum
}

__global__ void addKernel(int *c, const int *a, const int *b)
{
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}

int main()
{
    
    return 0;
}
