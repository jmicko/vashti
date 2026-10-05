// Synthetic arithmetic/routing oracle. No model or request content is loaded.
#include <array>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <sys/resource.h>
#include <vector>
#include "ggml-alloc.h"
#include "ggml-cpu.h"
#include "model/common/rope.hpp"
#include "whole-rope-policy.hpp"
#include "whole-rope-test.h"

namespace {
struct Failure { int phase; };
bool uncertain = false;
bool device_mode = false;
unsigned current_case = 0;
unsigned comparisons = 0;
unsigned cases = 0;
unsigned total_selected=0,total_launches=0;
#ifdef KUNO_WHOLE_ROPE_FAILURE_TEST
unsigned injected_failure = 0;
unsigned sync_attempts = 0;
std::array<unsigned,5> release_attempts{};
std::array<unsigned,5> released{};
#endif
[[noreturn]] void device_failure(int phase) noexcept {
    // A throwing device call/status may leave DMA using host vectors. Terminate
    // before C++ stack/vector/resource destruction; never claim a successful
    // drain from that uncertain state. The protected parent owns fault recovery.
    std::fprintf(stderr,"whole_rope_error case=%u phase=%d cleanup_failed=1\n",current_case,phase);
    std::fflush(stderr);
    std::_Exit(1);
}
template<class Call> decltype(auto) checked(Call&& call) {
    if (uncertain) {
        if(device_mode) device_failure(20);
        throw Failure{20};
    }
    try { return call(); }
    catch (...) {
        uncertain = true;
        if(device_mode) device_failure(19);
        throw;
    }
}
void require(bool condition, int phase) { if (!condition) throw Failure{phase}; }
void backend_result(bool condition,int phase) {
    if(condition)return;
    // Backend allocation/init false or null is distinct from a pure shape,
    // memory-budget or unsupported-op prevalidation decline. Device state is
    // uncertain; leave process cleanup to the protected parent without unwind.
    uncertain=true;
    if(device_mode)device_failure(phase);
    throw Failure{phase};
}
void status_checked(enum ggml_status status,int phase) {
    if(status==GGML_STATUS_SUCCESS) return;
    uncertain=true;
    if(device_mode) device_failure(phase);
    throw Failure{phase};
}
void compute_checked(ggml_backend_t backend,ggml_cgraph* graph,int phase) {
    auto status=checked([&] { return ggml_backend_graph_compute(backend,graph); });
#ifdef KUNO_WHOLE_ROPE_FAILURE_TEST
    if(injected_failure==6) status=GGML_STATUS_FAILED;
#endif
    status_checked(status,phase);
}
void synchronize(ggml_backend_t backend) {
    checked([&] {
#ifdef KUNO_WHOLE_ROPE_FAILURE_TEST
        if(++sync_attempts==1&&injected_failure==2) throw 1;
#endif
        ggml_backend_synchronize(backend);
    });
}
template<class Call> void release(unsigned resource,Call&& call) {
    checked([&] {
#ifdef KUNO_WHOLE_ROPE_FAILURE_TEST
        ++release_attempts[resource];
        if((resource==1&&injected_failure==4)||(resource==4&&injected_failure==5)) throw 1;
#else
        (void)resource;
#endif
        call();
#ifdef KUNO_WHOLE_ROPE_FAILURE_TEST
        ++released[resource];
#endif
    });
}
uint32_t word(float value) { uint32_t result; std::memcpy(&result,&value,4); return result; }
float value(uint32_t bits) { float result; std::memcpy(&result,&bits,4); return result; }
bool finite(uint32_t bits) { return (bits & 0x7f800000u) != 0x7f800000u; }
bool nan(uint32_t bits) { return !finite(bits) && (bits & 0x007fffffu); }
bool zero(uint32_t bits) { return (bits & 0x7fffffffu) == 0; }
bool subnormal(uint32_t bits) { return !(bits & 0x7f800000u) && !zero(bits); }
void same(const std::vector<float>& a,const std::vector<float>& b,int phase) {
    require(a.size()==b.size(),phase);
    std::array<unsigned long long,6> mismatch{};
    for (size_t i=0;i<a.size();++i) {
        const auto x=word(a[i]), y=word(b[i]);
        if (x==y) continue;
        ++mismatch[nan(x)&&nan(y) ? 0 : nan(x)!=nan(y) ? 1 :
                   !finite(x)||!finite(y) ? 2 : zero(x)&&zero(y) ? 3 :
                   subnormal(x)||subnormal(y) ? 4 : 5];
    }
    ++comparisons;
    if (mismatch != std::array<unsigned long long,6>{}) {
        std::printf("whole_rope_mismatch case=%u phase=%d nan_bits=%llu nan_class=%llu infinity=%llu signed_zero=%llu subnormal=%llu finite=%llu\n",
            current_case,phase,mismatch[0],mismatch[1],mismatch[2],mismatch[3],mismatch[4],mismatch[5]);
        throw Failure{phase};
    }
}
static uint32_t mix(uint32_t i){i^=i>>16;i*=0x7feb352du;i^=i>>15;i*=0x846ca68bu;return i^(i>>16);}
static std::vector<float> pattern(size_t count,unsigned kind,unsigned shift) {
    const uint32_t special[]={0,0x80000000u,0x3f800000u,0xbf800000u,0x3f800001u,0x3f7fffffu,
        0x00800000u,1,0x807fffffu,0x7f800000u,0xff800000u,0x7fc01234u,0xffc05678u,0x7fa02468u,0xffa13579u};
    std::vector<float> result(count);
    for(size_t i=0;i<count;++i) {const auto bits=mix(uint32_t(i)+shift);
        result[i]=kind==1?value(special[(i+shift)%15]):kind==3?
            value((bits&0x807fffffu)|(uint32_t((i+shift)%255)<<23)):
            value((bits&0x807fffffu)|((126+(bits%3))<<23));}
    return result;
}
static std::vector<float> coefficients(int length,unsigned kind,unsigned changed) {
    if(kind!=2&&kind!=4)return pattern(256*size_t(length),kind,3+23*changed);
    // Public synthetic Qwen multi-axis geometry/positions, independent of user
    // content. Model theta/axes match the pinned Qwen2.1 configuration.
    auto ids=Rope::gen_qwen_image_ids(1,2,2,2,1,length-1,{},Rope::RefIndexMode::INCREASE);
    require(ids.size()==size_t(length),5);
    if(changed)for(auto& position:ids)for(float& axis:position)axis+=1.25f;
    auto raw=Rope::embed_nd(ids,1,10000.f,{16,56,56});
    std::vector<float> prepared(raw.size());
    for(int t=0;t<length;++t)for(int j=0;j<64;++j)for(int c=0;c<2;++c)for(int r=0;r<2;++r)
        prepared[r+2*j+128*t+128*length*c]=raw[c+2*r+4*j+256*t];
    return prepared;
}
struct BackendOwner {
    ggml_backend_t value = nullptr;
    void release() {
        if (!value || uncertain) return;
        synchronize(value);
        ::release(4,[&] { ggml_backend_free(value); });
        value=nullptr;
    }
    ~BackendOwner() noexcept { try { release(); } catch (...) { uncertain=true; } }
};
struct Resources {
    ggml_backend_t backend;
    ggml_context* context = nullptr;
    ggml_backend_buffer_t inputs = nullptr;
    ggml_gallocr_t allocator = nullptr;
    explicit Resources(ggml_backend_t b):backend(b) {}
    void release() {
        if (uncertain || (!allocator&&!inputs&&!context)) return;
        // This retains the backend's existing wait() synchronization contract.
        synchronize(backend);
        if (allocator) { ::release(1,[&] { ggml_gallocr_free(allocator); }); allocator=nullptr; }
        if (inputs) { ::release(2,[&] { ggml_backend_buffer_free(inputs); }); inputs=nullptr; }
        if (context) { ::release(3,[&] { ggml_free(context); }); context=nullptr; }
    }
    ~Resources() noexcept { try { release(); } catch (...) { uncertain=true; } }
};
std::vector<float> read(ggml_tensor* tensor) {
    std::vector<float> result(size_t(ggml_nelements(tensor)));
    checked([&] { ggml_backend_tensor_get(tensor,result.data(),0,ggml_nbytes(tensor)); });
    return result;
}
struct Spec {int length;unsigned kind,produced,escape,branches;bool strided,interleaved;};
struct Output {std::vector<std::vector<float>> data;std::array<unsigned long long,9> routes{};};
void mode(bool enabled,bool device) {
#ifdef KUNO_SYCL_WHOLE_ROPE_TEST
    if(device)kuno_sycl_whole_rope_test_enable(enabled);
#else
    (void)enabled;require(!device,2);
#endif
}
std::array<unsigned long long,9> routes(bool device) {
    std::array<unsigned long long,9> result{};
#ifdef KUNO_SYCL_WHOLE_ROPE_TEST
    if(device)kuno_sycl_whole_rope_test_counts(result.data());
#else
    (void)device;
#endif
    return result;
}
struct GraphFixture {
    Resources owner;
    ggml_tensor *input_owner=nullptr,*input=nullptr,*pe=nullptr,*weight=nullptr;
    ggml_cgraph* graph=nullptr;
    std::vector<ggml_tensor*> outputs;
    std::vector<float> xd,pd,wd;
    unsigned eligible=0,expected=0;
    bool device;
    Spec spec;
    GraphFixture(ggml_backend_t backend,bool on_device,const Spec& request,unsigned changed)
        :owner(backend),device(on_device),spec(request) {
        owner.context=ggml_init({16*1024*1024,nullptr,true});require(owner.context,3);
        auto* ctx=owner.context;
        const size_t head_stride=640,token_stride=32*head_stride+128;
        const size_t extent=128+token_stride*size_t(spec.length-1)+31*head_stride+512;
        input_owner=spec.strided?ggml_new_tensor_1d(ctx,GGML_TYPE_F32,int64_t(extent/4)):
            ggml_new_tensor_4d(ctx,GGML_TYPE_F32,128,32,spec.length,1);
        input=spec.strided?ggml_view_4d(ctx,input_owner,128,32,spec.length,1,head_stride,token_stride,token_stride*spec.length,128):input_owner;
        pe=ggml_new_tensor_4d(ctx,GGML_TYPE_F32,2,64,spec.length,2);
        // Exact prepared-PE leaf topology used by the steady cached segment;
        // inherited VIEW name suffixes are deliberately part of this fixture.
        ggml_set_name(pe,"ggml_runner_cut:synthetic|prepared|");
        weight=ggml_new_tensor_1d(ctx,GGML_TYPE_F32,128);
        xd=pattern(size_t(ggml_nelements(input_owner)),spec.kind==1?1:spec.kind>=3?3:0,17);
        pd=coefficients(spec.length,spec.kind,changed);
        wd=pattern(128,0,71);
        if(spec.kind==0||spec.kind==3) {
            const size_t first=spec.strided?32:0;
            // FMA-distinct cancellation survives only two F32 roundings.
            xd[first]=value(0x3f800001u);xd[first+1]=value(0xbf800000u);
            pd[0]=value(0x3f7ffffeu);pd[128*spec.length]=1.f;
            pd[1]=value(0x3f800001u);pd[1+128*spec.length]=1.f;
        }
        if(spec.kind>=3) {
            std::array<bool,255> seen{};
            for(float x:xd){const auto bits=word(x);require(finite(bits),12);seen[(bits>>23)&255]=true;}
            for(bool found:seen)require(found,12);
            for(float coefficient:pd)require(finite(word(coefficient)),12);
            if(spec.kind==3){seen.fill(false);for(float coefficient:pd)seen[(word(coefficient)>>23)&255]=true;
                for(bool found:seen)require(found,12);}
        }
        constexpr size_t slack=64ull*1024*1024;
        ggml_backend_dev_props props{};
        const size_t input_bytes=ggml_nbytes(input_owner)+ggml_nbytes(pe)+ggml_nbytes(weight);
        if(device) {
            checked([&]{ggml_backend_dev_get_props(ggml_backend_get_device(backend),&props);});
            require(props.memory_free>=slack&&input_bytes<=props.memory_free-slack,7);
        }
        owner.inputs=checked([&]{return ggml_backend_alloc_ctx_tensors(ctx,backend);});backend_result(owner.inputs!=nullptr,4);
        auto* source=input;
        if(spec.produced==1||spec.produced==2)source=ggml_dup(ctx,input);
        if(spec.produced==2)ggml_set_output(source);
        if(spec.produced==3)source=ggml_rms_norm(ctx,input,1.e-6f);
        if(spec.produced==4)source=ggml_mul(ctx,ggml_rms_norm(ctx,input,1.e-6f),weight);
        if(spec.produced==5)source=ggml_mul_inplace(ctx,ggml_rms_norm(ctx,input,1.e-6f),weight);
        if(spec.produced==5)require(source->op==GGML_OP_MUL&&source->view_src==source->src[0]&&
            source->view_src&&source->view_src->op==GGML_OP_RMS_NORM,5);
        graph=ggml_new_graph_custom(ctx,16384,false);
        for(unsigned branch=0;branch<spec.branches;++branch) {
            auto* rope=Rope::apply_rope_with_prepared_pe(ctx,source,pe,spec.interleaved);
            auto* output=ggml_dup(ctx,rope);ggml_set_output(output);outputs.push_back(output);
            ggml_build_forward_expand(graph,output);
            auto* add=rope->src[0];
            if(spec.interleaved) {
                require(add&&add->op==GGML_OP_ADD,5);
                auto* mul=add->src[0];auto* repeat=mul->src[0];
                if(spec.escape==1) {ggml_set_output(repeat);outputs.push_back(repeat);}
                if(spec.escape==2) {auto* half=repeat->src[0];ggml_set_output(half);outputs.push_back(half);}
                if(spec.escape==3) {ggml_set_name(mul,"ggml_runner_cut:synthetic|mul|");ggml_set_output(mul);outputs.push_back(mul);}
                if(spec.escape==4) {auto* view=ggml_view_1d(ctx,repeat,128,0);auto* escaped=ggml_dup(ctx,view);
                    ggml_set_output(escaped);outputs.push_back(escaped);ggml_build_forward_expand(graph,escaped);}
                if(spec.escape==5) {auto* escaped=ggml_dup(ctx,repeat);ggml_set_output(escaped);
                    outputs.push_back(escaped);ggml_build_forward_expand(graph,escaped);}
                if(spec.escape==6)ggml_set_output(rope);
            }
        }
        for(int i=0;i<graph->n_nodes;++i)require(checked([&]{return ggml_backend_supports_op(backend,graph->nodes[i]);}),6);
        owner.allocator=checked([&]{return ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));});backend_result(owner.allocator!=nullptr,7);
        size_t bytes=0;checked([&]{ggml_gallocr_reserve_n_size(owner.allocator,graph,nullptr,nullptr,&bytes);});
        require(bytes<=1024ull*1024*1024,7);
        if(device) {
            checked([&]{ggml_backend_dev_get_props(ggml_backend_get_device(backend),&props);});
            require(props.memory_free>=slack&&bytes<=props.memory_free-slack,7);
        }
        backend_result(checked([&]{return ggml_gallocr_reserve(owner.allocator,graph);}),7);
        backend_result(checked([&]{return ggml_gallocr_alloc_graph(owner.allocator,graph);}),7);
        for(int i=0;i<graph->n_nodes;++i)if(ggml_sycl_whole_rope_policy::find(graph,i,device?128:1))++eligible;
        if(spec.escape||!spec.interleaved)require(eligible==0,8);
        if(!spec.escape&&spec.interleaved&&(!spec.produced||spec.produced==2))require(eligible==spec.branches,8);
        expected=spec.branches>ggml_sycl_whole_rope_policy::max_plans?0:eligible;
    }
    Output compute(bool enabled) {
        checked([&]{ggml_backend_tensor_set(input_owner,xd.data(),0,ggml_nbytes(input_owner));});
        checked([&]{ggml_backend_tensor_set(pe,pd.data(),0,ggml_nbytes(pe));});
        checked([&]{ggml_backend_tensor_set(weight,wd.data(),0,ggml_nbytes(weight));});
        mode(enabled,device);compute_checked(owner.backend,graph,9);synchronize(owner.backend);
        Output result;result.routes=::routes(device);
        for(auto* output:outputs)result.data.push_back(read(output));
        // Integer byte equality after confirmed synchronous readback. These
        // invariance checks are separate from the output comparison counter.
        const auto input_copy=read(input_owner),pe_copy=read(pe),weight_copy=read(weight);
        require(!std::memcmp(input_copy.data(),xd.data(),xd.size()*4)&&
            !std::memcmp(pe_copy.data(),pd.data(),pd.size()*4)&&!std::memcmp(weight_copy.data(),wd.data(),wd.size()*4),10);
        if(device) {
            require(result.routes[1]==(enabled?expected:0)&&result.routes[2]==(enabled?expected:0),11);
            if(enabled&&spec.branches>ggml_sycl_whole_rope_policy::max_plans)require(result.routes[8]>0&&result.routes[1]==0&&result.routes[2]==0,11);
        }
        if(spec.kind==0||spec.kind==2)for(const auto& out:result.data)for(float x:out)require(finite(word(x)),12);
        else if(spec.kind==1){bool found=false;for(const auto& out:result.data)for(float x:out)found|=nan(word(x));require(found,12);}
        // Finite wide-exponent inputs may overflow or form NaNs through the
        // ordered arithmetic. Compare their resulting bits without sanitizing.
#ifdef KUNO_WHOLE_ROPE_FAILURE_TEST
        if(injected_failure==1&&enabled)result.data.front().back()=value(0x7fc01234u);
#endif
        return result;
    }
};
void compare(const Output& old,const Output& candidate,const Output& replay) {
    require(old.data.size()==candidate.data.size()&&candidate.data.size()==replay.data.size(),13);
    for(size_t i=0;i<old.data.size();++i){same(old.data[i],candidate.data[i],13);same(candidate.data[i],replay.data[i],14);}
}
void fixture(ggml_backend_t backend,bool device,const Spec& spec,bool benchmark=false) {
    ++current_case;
    GraphFixture original(backend,device,spec,0);
    const auto old=original.compute(false),candidate=original.compute(true),replay=original.compute(true);
    compare(old,candidate,replay);
    // Keep the first allocation alive while binding the fresh graph to changed
    // PE. The original pointer cannot be recycled into the second allocation.
    GraphFixture rebound(backend,device,spec,1);
    require(original.pe->data!=rebound.pe->data,8);
    const auto changed_old=rebound.compute(false),changed_candidate=rebound.compute(true),changed_replay=rebound.compute(true);
    compare(changed_old,changed_candidate,changed_replay);
    bool changed=false;
    for(size_t i=0;i<old.data.size();++i)changed|=old.data[i].size()!=changed_old.data[i].size()||
        std::memcmp(old.data[i].data(),changed_old.data[i].data(),old.data[i].size()*4)!=0;
    require(changed,15);
    if(device)require(candidate.routes==replay.routes&&changed_candidate.routes==changed_replay.routes,14);
    if(benchmark)for(bool enabled:{false,true}) {
        mode(enabled,device);
        for(unsigned i=0;i<3;++i)compute_checked(backend,rebound.graph,16);
        synchronize(backend);
        for(unsigned trial=0;trial<5;++trial) {
            const auto start=std::chrono::steady_clock::now();
            for(unsigned i=0;i<4;++i)compute_checked(backend,rebound.graph,16);
            synchronize(backend);
            const auto ns=std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now()-start).count();
            std::printf("whole_rope_benchmark case=%u mode=%u trial=%u evaluations=4 nanoseconds=%lld\n",current_case,unsigned(enabled),trial,static_cast<long long>(ns));
        }
    }
    rebound.owner.release();original.owner.release();require(!uncertain,20);
    total_selected+=unsigned(candidate.routes[1]+changed_candidate.routes[1]);
    total_launches+=unsigned(candidate.routes[2]+changed_candidate.routes[2]);
    ++cases;
    std::printf("whole_rope_case case=%u kind=%u produced=%u escape=%u branches=%u eligible=%u attempts=%llu selected=%llu launches=%llu fallback=%llu bitwise=1 replay=1 changed_pe=1 pointer_rebound=1\n",
        current_case,spec.kind,spec.produced,spec.escape,spec.branches,original.eligible,candidate.routes[0],candidate.routes[1],candidate.routes[2],candidate.routes[8]);
}
bool pci(const char* text) {
    if(!text||std::strlen(text)!=12||text[4]!=':'||text[7]!=':'||text[10]!='.')return false;
    for(unsigned i=0;i<12;++i){if(i==4||i==7||i==10)continue;
        if(!((text[i]>='0'&&text[i]<='9')||(text[i]>='a'&&text[i]<='f')))return false;}
    return text[11]<='7';
}
} // namespace
int main(int argc,char** argv) {
    std::setvbuf(stdout,nullptr,_IOLBF,0);
    rlimit no_core{0,0};if(setrlimit(RLIMIT_CORE,&no_core)){std::fputs("whole_rope_error case=0 phase=1 cleanup_failed=0\n",stderr);return 2;}
    const bool device=argc==4||argc==5;device_mode=device;
    const bool benchmark=argc==5;
    if(argc!=1&&(!device||std::strcmp(argv[1],"--device")||std::strcmp(argv[2],"SYCL")||!pci(argv[3])||(benchmark&&std::strcmp(argv[4],"--benchmark")))) {
        std::fputs("whole_rope_error case=0 phase=1 cleanup_failed=0\n",stderr);return 2;
    }
    try {
        ggml_log_set([](ggml_log_level,const char*,void*){},nullptr);
        BackendOwner backend;
#ifdef KUNO_WHOLE_ROPE_FAILURE_TEST
        if(injected_failure==3)checked([]{throw 1;});
#endif
        if(device) {
            checked([]{ggml_backend_load_all();});ggml_backend_dev_t selected=nullptr;
            const auto device_count=checked([]{return ggml_backend_dev_count();});
            for(size_t i=0;i<device_count;++i){auto dev=checked([&]{return ggml_backend_dev_get(i);});ggml_backend_dev_props props{};
                backend_result(dev!=nullptr,2);
                checked([&]{ggml_backend_dev_get_props(dev,&props);});
                if(props.type==GGML_BACKEND_DEVICE_TYPE_GPU&&props.device_id&&!std::strcmp(props.device_id,argv[3])&&props.name&&!std::strncmp(props.name,"SYCL",4)) {
                    require(!selected,2);selected=dev;
                }}
            require(selected,2);backend.value=checked([&]{return ggml_backend_dev_init(selected,nullptr);});
        } else {backend.value=checked([]{return ggml_backend_cpu_init();});backend_result(backend.value!=nullptr,2);
            checked([&]{ggml_backend_cpu_set_n_threads(backend.value,1);});}
        backend_result(backend.value!=nullptr,2);
        for(int length:{1,3,17,65})for(unsigned kind=0;kind<3;++kind)for(unsigned produced=0;produced<3;++produced)
            fixture(backend.value,device,{length,kind,produced,0,1,false,true});
        for(int length:{1,3,17,65})for(unsigned produced:{3u,4u})
            fixture(backend.value,device,{length,2,produced,0,1,false,true});
        for(unsigned escape=1;escape<=5;++escape)for(unsigned kind:{0u,1u})
            fixture(backend.value,device,{3,kind,0,escape,1,false,true});
        for(int length:{1,3,17,65})for(unsigned kind:{0u,1u})
            fixture(backend.value,device,{length,kind,0,0,1,false,false});
        for(int length:{3,17,65})for(unsigned kind=0;kind<3;++kind)
            fixture(backend.value,device,{length,kind,0,0,1,true,true});
        for(unsigned kind:{0u,1u})fixture(backend.value,device,{3,kind,0,6,1,false,true});
        for(unsigned branches:{64u,65u,128u})fixture(backend.value,device,{1,2,0,0,branches,false,true});
        for(unsigned kind:{3u,4u})fixture(backend.value,device,{65,kind,0,0,1,false,true});
        if(device) {
            for(unsigned kind=0;kind<3;++kind)fixture(backend.value,true,{4096,kind,0,0,1,false,true},benchmark&&kind==2);
            for(unsigned kind:{0u,2u})fixture(backend.value,true,{8214,kind,0,0,1,false,true},benchmark&&kind==2);
            fixture(backend.value,true,{4096,3,0,0,1,false,true});
            require(total_selected>0&&total_launches>0,11);
        }
        // Append actual Qwen RMSNorm's affine in-place path without renumbering
        // the established 84-device/78-CPU corpus. The larger case is GPU-only.
        fixture(backend.value,device,{65,2,5,0,1,false,true});
        if(device)fixture(backend.value,true,{8214,2,5,0,1,false,true});
        require(cases==(device?86u:79u)&&comparisons==(device?1400u:1372u),5);
        backend.release();require(!uncertain,20);
        std::printf("whole_rope_completed cases=%u comparisons=%u bitwise=1 replay=1 changed_pe=1 pointer_rebound=1\n",cases,comparisons);return 0;
    } catch(const Failure& failure) {
        std::fprintf(stderr,"whole_rope_error case=%u phase=%d cleanup_failed=%u\n",current_case,failure.phase,unsigned(uncertain));return 1;
    } catch(...) {
        std::fprintf(stderr,"whole_rope_error case=%u phase=19 cleanup_failed=%u\n",current_case,unsigned(uncertain));return 1;
    }
}
