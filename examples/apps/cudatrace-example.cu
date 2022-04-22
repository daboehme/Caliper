#include <caliper/CUDATrace.cuh>

#include <caliper/cali.h>

#include <iostream>

using namespace cali;


__global__ void kernel(int n, int* res)
{
    auto inner = cudatrace::register_region(5, "inner");
    auto outer = cudatrace::begin_region(5, "outer");

    for (int i = 0; i < n; ++i) {
        cudatrace::begin_region(inner);
        res[i*gridDim.x*blockDim.x + blockIdx.x*blockDim.x+threadIdx.x] = threadIdx.x;
        cudatrace::end_region(inner);
    }

    cudatrace::end_region(outer);
}


int main(int argc, char* argv[])
{
    const int blocks = 2;
    const int threads = 32;
    const int n = 4;
    int size = n * blocks * threads;
    int* d_res = nullptr;

    CALI_MARK_FUNCTION_BEGIN;

    cudaMalloc(&d_res, size);

    CALI_MARK_BEGIN("aaaa");
    kernel<<<blocks, threads>>>(n, d_res);
    cudaDeviceSynchronize();
    CALI_MARK_END("aaaa");

    CALI_MARK_BEGIN("bbbb");
    kernel<<<blocks, threads>>>(n, d_res);
    cudaDeviceSynchronize();
    CALI_MARK_END("bbbb");

    cudaFree(d_res);
    
    CALI_MARK_FUNCTION_END;
}
