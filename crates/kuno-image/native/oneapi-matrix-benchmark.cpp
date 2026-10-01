#include <sycl/sycl.hpp>
#include <oneapi/mkl/blas.hpp>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

// Synthetic arithmetic only. No model files, content, or inference requests.
struct Shape { int64_t rows, k, n; };
constexpr std::array<Shape, 4> shapes{{{17,256,33}, {4056,4096,4096},
                                    {4056,4096,24576}, {4056,12288,4096}}};

static int64_t padded_rows(int64_t rows) { return (rows + 3) & ~int64_t(3); }
static size_t count(int64_t a, int64_t b) {
    if (a <= 0 || b <= 0 || uint64_t(a) > SIZE_MAX / uint64_t(b)) throw std::runtime_error("size");
    return size_t(a) * size_t(b);
}
static int8_t pattern(size_t i, uint32_t salt) {
    uint32_t x = uint32_t(i) * 747796405u + salt;
    x ^= x >> 16; x *= 2246822519u; x ^= x >> 13;
    return int8_t(int(x % 15) - 7);
}
static int64_t reference_cell(const std::vector<int8_t>& weights,
                              const std::vector<int8_t>& input,
                              Shape shape, int64_t row, int64_t output) {
    int64_t sum = 0;
    for (int64_t k=0; k<shape.k; ++k)
        sum += int64_t(input[size_t(row)*shape.k+k]) * weights[size_t(output)*shape.k+k];
    return sum;
}
static void inputs(Shape shape, bool edges, std::vector<int8_t>& weights,
                   std::vector<int8_t>& input) {
    weights.resize(count(shape.n,shape.k));
    input.resize(count(padded_rows(shape.rows),shape.k));
    for (size_t i=0; i<weights.size(); ++i) weights[i]=pattern(i,19);
    for (int64_t r=0; r<shape.rows; ++r)
        for (int64_t k=0; k<shape.k; ++k) input[size_t(r)*shape.k+k]=pattern(size_t(r)*shape.k+k,41);
    // Padded rows and these zero controls remain zero.
    if (edges) {
        std::fill_n(weights.data(),size_t(shape.k),int8_t(127));
        std::fill_n(input.data(),size_t(shape.k),int8_t(127));
        if (shape.n>1) std::fill_n(weights.data()+shape.k,size_t(shape.k),int8_t(0));
        if (shape.n>2) std::fill_n(weights.data()+2*shape.k,size_t(shape.k),int8_t(-128));
        if (shape.rows>2) std::fill_n(input.data()+2*shape.k,size_t(shape.k),int8_t(127));
        if (shape.rows>3) std::fill_n(input.data()+3*shape.k,size_t(shape.k),int8_t(0));
    }
}
static void cpu_oracle() {
    const Shape shape{3,4,2};
    const std::vector<int8_t> weights{1,2,3,4,-4,3,-2,1};
    const std::vector<int8_t> input{5,6,7,8,-1,0,1,2,0,0,0,0,0,0,0,0};
    constexpr int64_t expected[]{70,-8,10,4,0,0,0,0};
    for (int64_t r=0; r<padded_rows(shape.rows); ++r)
        for (int64_t n=0; n<shape.n; ++n)
            if (reference_cell(weights,input,shape,r,n)!=expected[r*shape.n+n]) throw std::runtime_error("oracle");
    std::printf("phase=1 cpu_reference_cases=8 status=passed\n");
}
static std::string normalize_pci(const std::string& value) {
    unsigned domain=0,bus=0,device=0,function=0; int used=0;
    bool good=std::sscanf(value.c_str(),"%x:%x:%x.%x%n",&domain,&bus,&device,&function,&used)==4 && used==int(value.size());
    if (!good) {
        domain=0; used=0;
        good=std::sscanf(value.c_str(),"%x:%x.%x%n",&bus,&device,&function,&used)==3 && used==int(value.size());
    }
    if (!good || domain>65535 || bus>255 || device>31 || function>7) throw std::runtime_error("pci");
    char normalized[20];std::snprintf(normalized,sizeof(normalized),"%04x:%02x:%02x.%x",domain,bus,device,function);
    return normalized;
}
static sycl::device select_device(const std::string& wanted) {
    std::vector<sycl::device> matches;
    for (const auto& device:sycl::device::get_devices(sycl::info::device_type::gpu)) {
        if (device.get_backend()!=sycl::backend::ext_oneapi_level_zero || !device.has(sycl::aspect::ext_intel_pci_address)) continue;
        if (normalize_pci(device.get_info<sycl::ext::intel::info::device::pci_address>())==wanted) matches.push_back(device);
    }
    if (matches.size()!=1) throw std::runtime_error("selection");
    return matches.front();
}
template <class T> struct DeviceBuffer {
    sycl::queue& queue; T* data=nullptr; size_t size;
    DeviceBuffer(sycl::queue& q,size_t n):queue(q),size(n) {
        data=sycl::malloc_device<T>(n,q);if (!data) throw std::bad_alloc();
    }
    ~DeviceBuffer() { if(data) sycl::free(data,queue); }
    DeviceBuffer(const DeviceBuffer&)=delete; DeviceBuffer& operator=(const DeviceBuffer&)=delete;
};
struct Timings { double first, minimum, median, maximum; };
static void wait_checked(sycl::queue& queue, const std::atomic<bool>& failed) {
    queue.wait_and_throw();if(failed.load())throw std::runtime_error("async");
}
struct DrainOnExit {
    sycl::queue& queue;
    ~DrainOnExit() { try { queue.wait_and_throw(); } catch(...) {} }
};
template <class Call> static Timings time_calls(sycl::queue& queue,
        const std::atomic<bool>& failed, Call call, int iterations) {
    auto once=[&]() {
        const auto begin=std::chrono::steady_clock::now();
        call().wait_and_throw(); wait_checked(queue,failed);
        return std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-begin).count();
    };
    Timings result{};result.first=once();std::vector<double> warm;
    for(int i=0;i<iterations;++i) warm.push_back(once());
    std::sort(warm.begin(),warm.end());result.minimum=warm.front();
    result.maximum=warm.back();result.median=warm[warm.size()/2];return result;
}
static void run_matrix(sycl::queue& queue, const std::atomic<bool>& failed, Shape shape, int shape_id,
                       bool oracle, int iterations) {
    const auto rows=padded_rows(shape.rows);
    std::vector<int8_t> weights,input;inputs(shape,oracle,weights,input);
    std::vector<sycl::half> half_weights(weights.size()),half_input(input.size());
    for(size_t i=0;i<weights.size();++i)half_weights[i]=sycl::half(float(weights[i]));
    for(size_t i=0;i<input.size();++i)half_input[i]=sycl::half(float(input[i]));
    const size_t output_count=count(rows,shape.n);
    const uint64_t bytes=3ull*(weights.size()+input.size())+8ull*output_count;
    if(bytes>2ull*1024*1024*1024)throw std::runtime_error("budget");
    // All copied host storage outlives the drain guard and the device buffers.
    std::vector<int32_t> output(output_count);std::vector<float> half_output(output_count);
    DeviceBuffer<int8_t> w(queue,weights.size()),x(queue,input.size());
    DeviceBuffer<sycl::half> hw(queue,half_weights.size()),hx(queue,half_input.size());
    DeviceBuffer<int32_t> y(queue,output_count);DeviceBuffer<float> hy(queue,output_count);
    DrainOnExit drain{queue};
    queue.memcpy(w.data,weights.data(),weights.size());queue.memcpy(x.data,input.data(),input.size());
    queue.memcpy(hw.data,half_weights.data(),half_weights.size()*sizeof(sycl::half));
    queue.memcpy(hx.data,half_input.data(),half_input.size()*sizeof(sycl::half));wait_checked(queue,failed);
    auto int8_call=[&]() {
        return oneapi::mkl::blas::column_major::gemm(queue,oneapi::mkl::transpose::trans,
            oneapi::mkl::transpose::nontrans,shape.n,rows,shape.k,1.f,w.data,shape.k,
            x.data,shape.k,0.f,y.data,shape.n,oneapi::mkl::blas::compute_mode::standard);
    };
    auto half_call=[&]() {
        return oneapi::mkl::blas::column_major::gemm(queue,oneapi::mkl::transpose::trans,
            oneapi::mkl::transpose::nontrans,shape.n,rows,shape.k,1.f,hw.data,shape.k,
            hx.data,shape.k,0.f,hy.data,shape.n,oneapi::mkl::blas::compute_mode::standard);
    };
    const auto integer=time_calls(queue,failed,int8_call,iterations);
    const auto floating=time_calls(queue,failed,half_call,iterations);
    queue.memcpy(output.data(),y.data,output_count*sizeof(int32_t));
    queue.memcpy(half_output.data(),hy.data,output_count*sizeof(float));wait_checked(queue,failed);
    double max_error=0;uint64_t reference_checks=0;
    for(int64_t r=0;r<rows;++r)for(int64_t n=0;n<shape.n;++n) {
        const size_t i=size_t(r)*shape.n+n;
        if(!std::isfinite(half_output[i]))throw std::runtime_error("nonfinite");
        max_error=std::max(max_error,std::abs(double(half_output[i])-output[i]));
        if(oracle) {
            const auto expected=reference_cell(weights,input,shape,r,n);
            if(expected!=output[i])throw std::runtime_error("integer");
            ++reference_checks;
        }
        if(r>=shape.rows && (output[i]!=0 || half_output[i]!=0.f))throw std::runtime_error("padding");
    }
    if(!oracle) {
        // Independent large-shape checks across edges and interior, bounded to
        // at most64*K scalar products instead of exhaustive O(rows*N*K).
        for(int row_index=0;row_index<8;++row_index)for(int col_index=0;col_index<8;++col_index) {
            const int64_t r=(shape.rows-1)*row_index/7,n=(shape.n-1)*col_index/7;
            if(reference_cell(weights,input,shape,r,n)!=output[size_t(r)*shape.n+n])throw std::runtime_error("sample");
            ++reference_checks;
        }
    }
    // Integer-valued half inputs and these bounded sums make the two arithmetic
    // paths exactly comparable. This does not compare real ConvRot vs dense models.
    if(max_error!=0)throw std::runtime_error("dense");
    std::printf("phase=10 shape=%d rows=%lld padded_rows=%lld k=%lld n=%lld device_bytes=%llu cpu_reference_checks=%llu max_abs_error=%.9g status=passed\n",
        shape_id,(long long)shape.rows,(long long)rows,(long long)shape.k,(long long)shape.n,
        (unsigned long long)bytes,(unsigned long long)reference_checks,max_error);
    auto print=[&](int type,const Timings& t) {
        std::printf("phase=11 shape=%d type=%d iterations=%d first_ms=%.6f min_ms=%.6f median_ms=%.6f max_ms=%.6f\n",shape_id,type,iterations,t.first,t.minimum,t.median,t.maximum);
    };
    print(1,integer);print(2,floating);std::fflush(stdout);
}
int main(int argc,char** argv) {
    try {
        if(argc==2 && std::string(argv[1])=="--cpu-oracle") {cpu_oracle();return 0;}
        std::string pci;int shape_id=2,iterations=5;
        for(int i=1;i<argc;++i) {
            const std::string arg=argv[i];if(i+1>=argc)throw std::runtime_error("arg");
            if(arg=="--pci")pci=normalize_pci(argv[++i]);
            else if(arg=="--shape")shape_id=std::stoi(argv[++i]);
            else if(arg=="--iterations")iterations=std::stoi(argv[++i]);
            else throw std::runtime_error("arg");
        }
        if(pci.empty() || shape_id<0 || shape_id>=int(shapes.size()) || iterations<1 || iterations>20)throw std::runtime_error("arg");
        cpu_oracle();const auto device=select_device(pci);std::atomic<bool> failed{false};
        auto handler=[&](sycl::exception_list errors) {if(errors.begin()!=errors.end())failed.store(true);};
        sycl::queue queue(device,handler,sycl::property_list{sycl::property::queue::in_order{}});
        std::printf("phase=2 physical_pci=%s backend=1 fp16=%d device_memory_bytes=%llu\n",pci.c_str(),int(device.has(sycl::aspect::fp16)),
                    (unsigned long long)device.get_info<sycl::info::device::global_mem_size>());std::fflush(stdout);
        if(!device.has(sycl::aspect::fp16))throw std::runtime_error("feature");
        const std::array<Shape,4> oracle_shapes{{{1,4,1},{3,16,7},{17,256,33},{33,1024,65}}};
        for(size_t i=0;i<oracle_shapes.size();++i)run_matrix(queue,failed,oracle_shapes[i],-1-int(i),true,1);
        if(shape_id!=0)run_matrix(queue,failed,shapes[size_t(shape_id)],shape_id,false,iterations);
        queue.wait_and_throw();if(failed.load())throw std::runtime_error("async");
        std::printf("phase=99 status=passed\n");return 0;
    } catch(...) {std::fprintf(stderr,"phase=99 status=failed error_code=1\n");return 1;}
}
