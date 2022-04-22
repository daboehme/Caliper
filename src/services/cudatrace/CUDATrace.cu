#include <caliper/CUDATrace.cuh>

#include <algorithm>
#include <iostream>

using namespace cali;
using namespace cali::cudatrace;

namespace
{

__device__ __forceinline__ unsigned long long get_global_time() {
	unsigned long long globaltimer;
	asm volatile ("mov.u64 %0, %globaltimer;"   : "=l"(globaltimer));
	return globaltimer;
}

__device__ inline int get_thread_id() {
    return threadIdx.z * blockDim.x * blockDim.y
        + threadIdx.y * blockDim.x
        + threadIdx.x;
}

__device__ inline int get_block_id() {
    return blockIdx.z * gridDim.x * gridDim.y
        + blockIdx.y * gridDim.x
        + blockIdx.x;
}

__device__ inline int get_blocksize() {
    return blockDim.x * blockDim.y * blockDim.z;
}

}

__host__ CUDATrace::CUDATrace(unsigned trace_capacity, trace_entry_t* trace, unsigned region_capacity, region_entry_t* regions)
    : trace_capacity_  { trace_capacity },
        trace_pos_       { 0 },
        trace_           { trace },
        region_capacity_ { region_capacity },
        region_pos_      { 0 },
        regions_         { regions },
        correlation_id_  { -1 }
{
    if (region_capacity > 0) {
        std::fill_n(regions_[0].name, sizeof(regions_[0].name), 0);
        std::copy_n("UNKNOWN", 8, regions_[0].name);
        ++region_pos_;
    }
}

__device__ void CUDATrace::push_event(region_handle region, event_t event) {
    int block = get_block_id();

    if (get_thread_id() == 0) {
        auto p = atomicAdd(&trace_pos_, 1);

        if (p < trace_capacity_) {
            uint64_t timestamp = get_global_time();
            trace_[p] = { block, correlation_id_, region, event, timestamp };
        }
    }
}

__device__ CUDATrace::region_handle CUDATrace::find_region(int len, const char* name) {
    unsigned end = min(region_pos_, region_capacity_);

    int blocksize = get_blocksize();
    int t = get_thread_id();

    __shared__ int found;

    for (unsigned p = 0; p < end; p += blocksize) {
        int pos = p + t;

        if (t == 0)
            found = blocksize;

        __syncthreads();

        if (pos < end) {
            int c = 0;
            for ( ; c < len && regions_[pos].name[c] == name[c]; ++c)
                ;
            if (c == len)
                atomicMin(&found, t);
        }

        __syncthreads();

        if (found < blocksize)
            return p + found;
    }

    return 0;
}

__device__ CUDATrace::region_handle CUDATrace::register_region(int len, const char* name) {
    auto p = find_region(len, name);

    if (p > 0)
        return p;

    //   Add region only on one thread. This still executes on all blocks,
    // so we can get multiple entries for the same region.
    if (get_thread_id() == 0) {
        p = atomicAdd(&region_pos_, 1);

        if (p < region_capacity_) {
            memset(regions_[p].name, 0, MAX_NAME_LEN);
            memcpy(regions_[p].name, name, min(len, MAX_NAME_LEN-1));
        }
    }

    __syncthreads();

    return p < region_capacity_ ? p : 0;
}

__host__ void CUDATrace::print(std::ostream& os) {
    auto endp = std::min(trace_pos_, trace_capacity_);

    for (unsigned p = 0; p < endp; ++p) {
        const auto &e = trace_[p];

        os << e.timestamp << ' '
            << e.block_id  << ' '
            << e.correlation_id << ' '
            << (e.event == event_t::BEGIN ? "BEGIN " : "END ")
            << regions_[e.region_id].name
            << " (" << e.region_id << ")\n";
    }

    unsigned skipped = trace_pos_ > trace_capacity_ ? trace_pos_ - trace_capacity_ : 0;
    auto regions = std::min(region_pos_, region_capacity_);

    os << regions << " regions, "
        << endp    << " events, "
        << skipped << " skipped.\n";
}

__host__ unsigned CUDATrace::flush(CUDATrace::FlushFn f)
{
    unsigned count = 0;
    auto endp = std::min(trace_pos_, trace_capacity_);

    for (unsigned p = 0; p < endp; ++p) {
        const auto& e = trace_[p];
        const auto& r = regions_[e.region_id];
        f(r, e);
        ++count;
    }

    return count;
}

__host__ void CUDATrace::clear()
{
    trace_pos_ = 0;
}

CUDATrace* CUDATrace::create(size_t trace_capacity, size_t region_capacity) {
    CUDATrace::region_entry_t* reg = nullptr;
    CUDATrace::trace_entry_t* trace = nullptr;

    cudaMallocManaged(&trace, trace_capacity * sizeof(CUDATrace::trace_entry_t));
    cudaMallocManaged(&reg, region_capacity * sizeof(CUDATrace::region_entry_t));

    CUDATrace* obj = nullptr;
    cudaMallocManaged(&obj, sizeof(CUDATrace));
    new (obj) CUDATrace(trace_capacity, trace, region_capacity, reg);

    return obj;
}

void CUDATrace::release(CUDATrace** ptr) {
    CUDATrace* obj = *ptr;
    cudaFree(obj->trace_);
    cudaFree(obj->regions_);
    cudaFree(obj);
    *ptr = nullptr;
}

namespace 
{

__device__ __managed__ CUDATrace* tracer;

}

namespace cali
{

namespace cudatrace
{

__device__ CUDATrace::region_handle register_region(int len, const char* name)
{
    return ::tracer ? ::tracer->register_region(len, name) : 0;
}

__device__ CUDATrace::region_handle begin_region(int len, const char* name)
{
    if (::tracer) {
        return ::tracer->begin(len, name);
    }
    return 0;
}

__device__ void begin_region(CUDATrace::region_handle region)
{
    if (::tracer) {
        ::tracer->begin(region);
    }
}

__device__ void end_region(CUDATrace::region_handle region)
{
    if (::tracer) {
        ::tracer->end(region);
    }
}

__host__ CUDATrace* cudatrace_init(size_t trace_capacity, size_t region_capacity)
{
    ::tracer = CUDATrace::create(trace_capacity, region_capacity);
    return ::tracer;
}

__host__ void cudatrace_finalize()
{
    CUDATrace::release(&::tracer);
}

} // namespace cudatrace

} // namespace cali