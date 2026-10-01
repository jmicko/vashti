// Numeric-only standalone IM2COL acceptance; default CPU, explicit physical SYCL GPU.
#ifdef NDEBUG
#undef NDEBUG
#endif
#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <cstdint>
#include <initializer_list>
#include <set>
#include <sys/resource.h>
#include <vector>
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "im2col-launch-policy.hpp"
#include <sycl/sycl.hpp>
#include <sycl/detail/id_queries_fit_in_int.hpp>
static_assert(GGML_MAX_NAME == 160);
struct Shape { int w,h,c,n,kw,kh,sx,sy,px,py,dx,dy; ggml_type type; };
static float value(uint64_t i) { return float((i*37+17)%257)/64.f-2.f; }
static float expected(uint64_t at, const Shape & s, int ow, int oh) {
    const uint64_t k=uint64_t(s.c)*s.kw*s.kh;
    const int ic=(at%k)/(s.kw*s.kh), kh=(at%k/s.kw)%s.kh, kw=at%s.kw;
    const uint64_t p=at/k; const int x=p%ow,y=(p/ow)%oh,n=p/(uint64_t(ow)*oh);
    const int ix=x*s.sx+kw*s.dx-s.px,iy=y*s.sy+kh*s.dy-s.py;
    if(ix<0||ix>=s.w||iy<0||iy>=s.h) return 0.f;
    return value(((uint64_t(n)*s.c+ic)*s.h+iy)*s.w+ix);
}
static void run(ggml_backend_t backend,int id,const Shape & s) {
    auto ctx=ggml_init({4*1024*1024,nullptr,true}); assert(ctx);
    auto input=ggml_new_tensor_4d(ctx,GGML_TYPE_F32,s.w,s.h,s.c,s.n);
    auto weight=ggml_new_tensor_4d(ctx,GGML_TYPE_F16,s.kw,s.kh,s.c,1);
    auto inputs=ggml_backend_alloc_ctx_tensors(ctx,backend); assert(inputs);
    // At most 1 MiB host input staging, even for the full-resolution GPU case.
    std::vector<float> chunk(256*1024);
    const uint64_t elements=ggml_nelements(input);
    for(uint64_t begin=0;begin<elements;begin+=chunk.size()) {
        const size_t count=std::min<uint64_t>(chunk.size(),elements-begin);
        for(size_t i=0;i<count;++i)chunk[i]=value(begin+i);
        ggml_backend_tensor_set(input,chunk.data(),begin*sizeof(float),count*sizeof(float));
    }
    auto output=ggml_im2col(ctx,weight,input,s.sx,s.sy,s.px,s.py,s.dx,s.dy,true,s.type);
    assert(ggml_backend_supports_op(backend,output));
    const int ow=output->ne[1],oh=output->ne[2];
    auto graph=ggml_new_graph(ctx);ggml_build_forward_expand(graph,output);
    auto alloc=ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));assert(ggml_gallocr_alloc_graph(alloc,graph));
    printf("im2col_start shape=%d input_bytes=%zu output_bytes=%zu w=%d h=%d c=%d n=%d ow=%d oh=%d type=%d\n",id,ggml_nbytes(input),ggml_nbytes(output),s.w,s.h,s.c,s.n,ow,oh,int(s.type));fflush(stdout);
    const auto start=std::chrono::steady_clock::now();
    assert(ggml_backend_graph_compute(backend,graph)==GGML_STATUS_SUCCESS);ggml_backend_synchronize(backend);
    const double ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
    const uint64_t total=ggml_nelements(output),k=uint64_t(s.c)*s.kh*s.kw;
    std::set<uint64_t> checks;
    if(total<=200000)for(uint64_t i=0;i<total;++i)checks.insert(i);
    else {
        for(uint64_t p:std::initializer_list<uint64_t>{0ull,1ull,uint64_t(ow-1),uint64_t(ow),uint64_t(ow)*(oh-1),uint64_t(ow)*oh-1})
            for(uint64_t col:std::initializer_list<uint64_t>{0ull,1ull,8ull,9ull,255ull,256ull,k/2,k-1})if(col<k && p*k+col<total)checks.insert(p*k+col);
        for(uint64_t i=0;i<192;++i)checks.insert((i*6364136223846793005ull+1442695040888963407ull)%total);
        checks.insert(total-1);
    }
    std::vector<ggml_fp16_t> all_half;
    std::vector<float> all_float;
    if(total<=200000) {
        if(s.type==GGML_TYPE_F16){all_half.resize(total);ggml_backend_tensor_get(output,all_half.data(),0,total*2);}
        else{all_float.resize(total);ggml_backend_tensor_get(output,all_float.data(),0,total*4);}
    } else {
        for(uint64_t boundary:{uint64_t(1)<<31,uint64_t(1)<<32}) {
            const uint64_t index=boundary/ggml_element_size(output);
            for(uint64_t i=index-1;i<=index+1;++i)if(i<total)checks.insert(i);
        }
    }
    for(uint64_t i:checks) {
        float actual;
        if(s.type==GGML_TYPE_F16){ggml_fp16_t half;if(!all_half.empty())half=all_half[i];else ggml_backend_tensor_get(output,&half,i*2,2);actual=ggml_fp16_to_fp32(half);}
        else if(!all_float.empty())actual=all_float[i];else ggml_backend_tensor_get(output,&actual,i*4,4);
        const float wanted=expected(i,s,ow,oh);
        if(actual!=wanted)fprintf(stderr,"im2col_mismatch shape=%d index=%llu expected=%.9g actual=%.9g\n",id,(unsigned long long)i,wanted,actual);
        assert(actual==wanted);
    }
    printf("im2col_pass shape=%d checks=%zu ms=%.6f\n",id,checks.size(),ms);fflush(stdout);
    ggml_gallocr_free(alloc);ggml_backend_buffer_free(inputs);ggml_free(ctx);
}
static void run3d(ggml_backend_t backend,ggml_type type) {
    constexpr int w=7,h=5,d=3,c=2,n=2,kw=3,kh=2,kd=2;
    auto ctx=ggml_init({4*1024*1024,nullptr,true});assert(ctx);
    auto input=ggml_new_tensor_4d(ctx,GGML_TYPE_F32,w,h,d,c*n);
    auto weight=ggml_new_tensor_4d(ctx,GGML_TYPE_F16,kw,kh,kd,c);
    auto inputs=ggml_backend_alloc_ctx_tensors(ctx,backend);assert(inputs);
    std::vector<float> source(ggml_nelements(input));for(size_t i=0;i<source.size();++i)source[i]=value(i);
    ggml_backend_tensor_set(input,source.data(),0,source.size()*4);
    auto out=ggml_im2col_3d(ctx,weight,input,c,2,1,1,2,1,1,2,2,1,type);
    assert(ggml_backend_supports_op(backend,out));
    auto graph=ggml_new_graph(ctx);ggml_build_forward_expand(graph,out);
    auto alloc=ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));assert(ggml_gallocr_alloc_graph(alloc,graph));
    assert(ggml_backend_graph_compute(backend,graph)==GGML_STATUS_SUCCESS);ggml_backend_synchronize(backend);
    const int ow=out->ne[1],oh=out->ne[2],od=out->ne[3]/n;const size_t total=ggml_nelements(out);
    std::vector<ggml_fp16_t> half(type==GGML_TYPE_F16?total:0);
    std::vector<float> actual(type==GGML_TYPE_F32?total:0);
    ggml_backend_tensor_get(out,type==GGML_TYPE_F16?static_cast<void*>(half.data()):static_cast<void*>(actual.data()),0,ggml_nbytes(out));
    for(size_t i=0;i<total;++i) {
        const int ic=(i%(c*kd*kh*kw))/(kd*kh*kw),z=(i%(kd*kh*kw))/(kh*kw),y=(i%(kh*kw))/kw,x=i%kw;
        const size_t pixel=i/(c*kd*kh*kw);const int ox=pixel%ow,oy=(pixel/ow)%oh,oz=(pixel/(ow*oh))%od,batch=pixel/(ow*oh*od);
        const int ix=ox*2+x*2-2,iy=oy+y*2-1,iz=oz+z-1;
        const float wanted=(ix<0||ix>=w||iy<0||iy>=h||iz<0||iz>=d)?0.f:value((((uint64_t(batch)*c+ic)*d+iz)*h+iy)*w+ix);
        assert((type==GGML_TYPE_F16?ggml_fp16_to_fp32(half[i]):actual[i])==wanted);
    }
    printf("im2col_3d_pass type=%d checks=%zu\n",int(type),total);fflush(stdout);
    ggml_gallocr_free(alloc);ggml_backend_buffer_free(inputs);ggml_free(ctx);
}
static void policy() {
#ifndef __SYCL_DEVICE_ONLY__
    static_assert(__SYCL_ID_QUERIES_FIT_IN_INT__==1,"Pinned prototype compiler range contract changed");
    const auto original=sycl::nd_range<3>(sycl::range<3>(832,1248,2816),sycl::range<3>(1,1,256));
    bool rejected=false;
    try{sycl::detail::checkValueRange(original);}catch(const sycl::exception&e){rejected=e.code()==sycl::make_error_code(sycl::errc::nd_range);}
    assert(rejected);
    assert(ggml_sycl_im2col_rows_per_launch(2816,1248)==611);
    assert(ggml_sycl_im2col_rows_per_launch(1,1)==65535);
    assert(ggml_sycl_im2col_rows_per_launch(0,1)==0);
    assert(ggml_sycl_im2col_rows_per_launch(1,0)==0);
    assert(ggml_sycl_im2col_rows_per_launch(-1,1)==0);
    assert(ggml_sycl_im2col_rows_per_launch(1,-1)==0);
    assert(ggml_sycl_im2col_rows_per_launch(INT32_MAX,1)==1);
    assert(ggml_sycl_im2col_rows_per_launch(1,INT32_MAX)==1);
    assert(ggml_sycl_im2col_rows_per_launch(INT64_MAX,1)==0);
    assert(ggml_sycl_im2col_rows_per_launch(INT64_MAX,INT64_MAX)==0);
    assert(ggml_sycl_im2col_rows_per_launch(INT32_MAX,2)==0);
    int accepted=0;
    for(auto shape:std::initializer_list<std::vector<int64_t>>{{2816,1248,832},{1536,1248,832},{10496,78,52},{1,1,131072}}) {
        const int64_t rows=ggml_sycl_im2col_rows_per_launch(shape[0],shape[1]);assert(rows>0);
        int64_t visited=0;
        for(int64_t begin=0;begin<shape[2];begin+=rows){const int64_t end=std::min(shape[2],begin+rows);sycl::detail::checkValueRange(sycl::range<3>(end-begin,shape[1],shape[0]));visited+=end-begin;++accepted;}
        assert(visited==shape[2]);
    }
    printf("im2col_policy original_rejected=1 chunks_accepted=%d\n",accepted);
#endif
}
int main(int argc,char**argv) {
    rlimit limit{0,0};assert(setrlimit(RLIMIT_CORE,&limit)==0);
    assert(argc==1 || argc==5);policy();ggml_backend_t backend=nullptr;
    if(argc==1){backend=ggml_backend_cpu_init();ggml_backend_cpu_set_n_threads(backend,2);}
    else {
        assert(!strcmp(argv[1],"--device")&&!strcmp(argv[2],"SYCL"));ggml_backend_load_all();
        for(size_t i=0;i<ggml_backend_dev_count();++i){auto d=ggml_backend_dev_get(i);ggml_backend_dev_props p{};ggml_backend_dev_get_props(d,&p);if(p.name&&!strncmp(p.name,"SYCL",4)&&p.device_id&&!strcmp(p.device_id,argv[3])){assert(!backend);backend=ggml_backend_dev_init(d,nullptr);}}
    }
    assert(backend);
    if(argc==1 || !strcmp(argv[4],"1")) {
        run(backend,1,{7,5,3,2,3,3,1,1,1,1,1,1,GGML_TYPE_F16});
        run(backend,2,{9,8,4,1,3,2,2,1,2,1,2,2,GGML_TYPE_F32});
        run(backend,3,{5,3,2,1,1,1,1,1,0,0,1,1,GGML_TYPE_F16});
        run(backend,7,{1,65536,1,2,1,1,1,1,0,0,1,1,GGML_TYPE_F16});
        run3d(backend,GGML_TYPE_F16);run3d(backend,GGML_TYPE_F32);
    } else if(!strcmp(argv[4],"2")) run(backend,4,{78,52,1152,1,3,3,1,1,1,1,1,1,GGML_TYPE_F16});
    else if(!strcmp(argv[4],"3")) run(backend,5,{1248,832,144,1,3,3,1,1,1,1,1,1,GGML_TYPE_F16});
    else if(!strcmp(argv[4],"4")) run(backend,6,{1248,832,288,1,3,3,1,1,1,1,1,1,GGML_TYPE_F16});
    else assert(false);
    ggml_backend_free(backend);puts("im2col_acceptance_passed");
}
