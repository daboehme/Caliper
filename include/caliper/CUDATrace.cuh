#include <iostream>
#include <functional>

namespace cali 
{

namespace cudatrace
{

class CUDATrace {
public:

    using region_handle = unsigned;

    enum event_t {
        BEGIN, END
    };

    struct trace_entry_t {
        int      block_id;
        int      correlation_id;
        unsigned region_id;
        event_t  event;
        uint64_t timestamp;
    };

    static const int MAX_NAME_LEN = 48;

    struct region_entry_t {
        char name[MAX_NAME_LEN];
    };

    using FlushFn = std::function<void(const region_entry_t&,const trace_entry_t&)>;

private:

    unsigned trace_capacity_;
    unsigned trace_pos_;

    trace_entry_t* trace_;
    
    unsigned region_capacity_;
    unsigned region_pos_;

    region_entry_t* regions_;

    int      correlation_id_;

    __host__ CUDATrace(unsigned trace_capacity, trace_entry_t* trace, unsigned region_capacity, region_entry_t* regions);

    __device__ void push_event(region_handle region, event_t event);

public:

    CUDATrace()
        : trace_capacity_  { 0 },
          trace_pos_       { 0 },
          trace_           { nullptr },
          region_capacity_ { 0 },
          region_pos_      { 0 },
          regions_         { nullptr },
          correlation_id_  { -1 }
    { }

    __device__ region_handle find_region(int len, const char* name);

    __device__ region_handle register_region(int len, const char* name);

    __device__ void begin(region_handle region) {
        push_event(region, event_t::BEGIN);
    }

    __device__ region_handle begin(int len, const char* name) {
        auto r = register_region(len, name);
        begin(r);
        return r;
    }

    __device__ void end(region_handle region) {
        push_event(region, event_t::END);
    }

    __device__ region_handle end(int len, const char* name) {
        auto r = register_region(len, name);
        end(r);
        return r;
    }

    __host__ __device__ void set_correlation(int corr) {
        correlation_id_ = corr;
    }

    __host__ void print(std::ostream& os);

    __host__ unsigned flush(FlushFn f);

    __host__ void clear();

    static CUDATrace* create(size_t trace_capacity, size_t region_capacity);

    static void release(CUDATrace** ptr);
};

__host__ CUDATrace* cudatrace_init(size_t trace_capacity, size_t region_capacity);
__host__ void cudatrace_finalize();

__device__ CUDATrace::region_handle register_region(int len, const char* name);

__device__ CUDATrace::region_handle begin_region(int len, const char* name);
__device__ void begin_region(CUDATrace::region_handle region);
__device__ void end_region(CUDATrace::region_handle region);

} // namespace cudatrace

} // namespace cali
