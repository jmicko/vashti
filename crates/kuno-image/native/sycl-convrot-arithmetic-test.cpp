// Independent synthetic oracle for direct and packed SYCL ConvRot GEMM.
// Default CPU only; explicit --device SYCL <PCI> is reserved for GPU acceptance.
#ifdef NDEBUG
#undef NDEBUG
#endif
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cfenv>
#include <cstdio>
#include <cstring>
#include <sys/resource.h>
#include <vector>
#include "ggml.h"
static_assert(GGML_MAX_NAME == 160, "Use the image engine GGML_MAX_NAME=160 ABI");
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

static void run(ggml_backend_t backend, int k, int rows, int n, int group, bool packed, bool bias, bool batched, bool ties = false, bool fractional_ties = false) {
    auto ctx = ggml_init({4*1024*1024,nullptr,true});
    auto x = ggml_new_tensor_4d(ctx,GGML_TYPE_F32,k,rows,batched?2:1,1);
    rows *= batched?2:1;
    auto w = ggml_new_tensor_2d(ctx,GGML_TYPE_I8,k,n);
    auto s = ggml_new_tensor_1d(ctx,GGML_TYPE_F32,n);
    auto b = ggml_new_tensor_1d(ctx,GGML_TYPE_F32,n);
    auto inputs = ggml_backend_alloc_ctx_tensors(ctx,backend); assert(inputs);
    std::vector<float> xd(k*rows),sd(n),bd(n);
    std::vector<int8_t> wd(k*n);
    for(int i=0;i<k*rows;++i) xd[i]= i<k?0.f:float((i*29+11)%63-31)/32.f;
    if (ties || fractional_ties) {
        constexpr float unit_controls[] = {127.f, .5f, 1.5f, 2.5f, -.5f, -1.5f, -2.5f, -127.f};
        // Regression: FP32(.40625 * FP32(1 / FP32(.8125 / 127))) is
        // exactly 63.5. Permitted device division error previously emitted 63.
        constexpr float fractional_controls[] = {.40625f, -.65625f, .25f, -.8125f, -.40625f, .65625f, -.25f, .8125f};
        const float * controls = fractional_ties ? fractional_controls : unit_controls;
        constexpr int matrix[4][4]={{1,1,1,-1},{1,1,-1,1},{1,-1,1,1},{-1,1,1,1}};
        for(int row=1;row<rows;++row) for(int i=0;i<k;++i) {
            if(!group) xd[row*k+i]=i<8?controls[i]:0.f;
            else {
                double input=0;
                for(int j=0;j<8;++j){int a=i%256,bb=j,sign=1;for(int digit=0;digit<4;++digit){sign*=matrix[a%4][bb%4];a/=4;bb/=4;}input+=controls[j]*sign/16.;}
                xd[row*k+i]=float(input);
            }
        }
    }
    for(int i=0;i<k*n;++i) wd[i]=int8_t((i*43+17)%256-128);
    for(int i=0;i<n;++i) {sd[i]=float(i%7+1)/256.f;bd[i]=float(i%5-2)/32.f;}
    ggml_backend_tensor_set(x,xd.data(),0,xd.size()*4);ggml_backend_tensor_set(w,wd.data(),0,wd.size());
    ggml_backend_tensor_set(s,sd.data(),0,sd.size()*4);ggml_backend_tensor_set(b,bd.data(),0,bd.size()*4);
    auto q = packed?ggml_quantize_i8_convrot(ctx,x,group):x;
    auto y = ggml_mul_mat_i8_tensorwise(ctx,w,q,s,bias?b:nullptr,group);
    assert(ggml_backend_supports_op(backend,y)); if(packed)assert(ggml_backend_supports_op(backend,q));
    auto graph=ggml_new_graph(ctx);ggml_build_forward_expand(graph,y);
    auto alloc=ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));assert(ggml_gallocr_alloc_graph(alloc,graph));
    assert(ggml_backend_graph_compute(backend,graph)==GGML_STATUS_SUCCESS);ggml_backend_synchronize(backend);
    std::vector<float> actual(rows*n);ggml_backend_tensor_get(y,actual.data(),0,actual.size()*4);
    std::vector<int8_t> bytes(k*rows);std::vector<float> scales(rows);
    if(packed) {
        ggml_backend_tensor_get(q,bytes.data(),0,bytes.size());
        const int padded=(rows+3)/4*4;
        ggml_backend_tensor_get(q,scales.data(),k*padded,rows*4);
        std::vector<int8_t> tail(k*(padded-rows));
        if(!tail.empty()){ggml_backend_tensor_get(q,tail.data(),k*rows,tail.size());for(auto v:tail)assert(v==0);}
    }
    constexpr int h[4][4]={{1,1,1,-1},{1,1,-1,1},{1,-1,1,1},{-1,1,1,1}};
    double max_error=0;
    for(int row=0;row<rows;++row) {
        std::vector<double> rotated(k);double maximum=0;
        for(int i=0;i<k;++i) {
            if(!group)rotated[i]=xd[row*k+i];
            else for(int j=0;j<256;++j){int a=i%256,bb=j,sign=1;for(int d=0;d<4;++d){sign*=h[a%4][bb%4];a/=4;bb/=4;}rotated[i]+=double(xd[row*k+(i/256)*256+j])*sign/16.;}
            maximum=std::max(maximum,std::abs(rotated[i]));
        }
        const float scale=float(maximum)/127.f; const float inverse=scale?1.f/scale:0.f;
        std::vector<int> quant(k);
        for(int i=0;i<k;++i)quant[i]=scale?std::clamp(int(std::nearbyint(float(rotated[i])*inverse)),-127,127):0;
        if(fractional_ties && row>0) {
            constexpr int expected_bytes[] = {64,-103,39,-127,-64,103,-39,127};
            for(int i=0;i<std::min(k,8);++i) assert(quant[i]==expected_bytes[i]);
        }
        if(packed){
            assert(std::abs(scales[row]-scale)<1e-7);
            for(int i=0;i<k;++i){
                if(quant[i]!=int(bytes[row*k+i]))fprintf(stderr,"quant_failure rows=%d k=%d group=%d row=%d col=%d expected=%d actual=%d scale=%.9g inverse=%.9g scaled=%.9g\n",rows,k,group,row,i,quant[i],int(bytes[row*k+i]),scale,inverse,float(rotated[i])*inverse);
                assert(quant[i]==int(bytes[row*k+i]));
            }
        }
        for(int col=0;col<n;++col){
            int64_t sum=0;for(int i=0;i<k;++i)sum+=int(wd[col*k+i])*(packed?int(bytes[row*k+i]):quant[i]);
            const double expected=double(sum)*(packed?double(scales[row]):scale)*sd[col]+(bias?bd[col]:0.f);
            const double error=std::abs(actual[row*n+col]-expected);max_error=std::max(max_error,error);
            assert(std::isfinite(actual[row*n+col]));if(error>3e-5+std::abs(expected)*2e-6)fprintf(stderr,"numeric_failure rows=%d k=%d n=%d group=%d packed=%d row=%d col=%d expected=%.9g actual=%.9g\n",rows,k,n,group,packed,row,col,expected,actual[row*n+col]);assert(error<=3e-5+std::abs(expected)*2e-6);
        }
    }
    printf("sycl_convrot_arithmetic rows=%d k=%d n=%d group=%d packed=%d bias=%d batch=%d max_error=%.9g\n",rows,k,n,group,packed,bias,batched,max_error);
    ggml_gallocr_free(alloc);ggml_backend_buffer_free(inputs);ggml_free(ctx);
}
int main(int argc,char**argv){
    rlimit limit{0,0};assert(setrlimit(RLIMIT_CORE,&limit)==0);assert(argc==1||argc==4);
    assert(std::fesetround(FE_TONEAREST)==0);
    ggml_backend_t backend=nullptr;
    if(argc==1){backend=ggml_backend_cpu_init();ggml_backend_cpu_set_n_threads(backend,2);}
    else{
        assert(strcmp(argv[1],"--device")==0);assert(strcmp(argv[2],"SYCL")==0);
        ggml_backend_load_all();
        for(size_t i=0;i<ggml_backend_dev_count();++i){auto d=ggml_backend_dev_get(i);ggml_backend_dev_props p{};ggml_backend_dev_get_props(d,&p);if(p.name&&strncmp(p.name,"SYCL",4)==0&&p.device_id&&strcmp(p.device_id,argv[3])==0){assert(!backend);backend=ggml_backend_dev_init(d,nullptr);}}
    }
    assert(backend);
    for(bool bias:{false,true})for(bool batch:{false,true}){
        run(backend,4,3,1,0,false,bias,batch);run(backend,12,5,7,0,false,bias,batch);
        run(backend,256,3,7,256,false,bias,batch);run(backend,512,5,65,256,false,bias,batch);
        run(backend,256,3,7,256,true,bias,batch);run(backend,512,5,65,256,true,bias,batch);
    }
    run(backend,12,3,7,0,false,true,false,true);
    run(backend,256,3,7,256,false,true,false,true);
    run(backend,256,3,7,256,true,true,false,true);
    run(backend,12,3,7,0,false,true,true,false,true);
    run(backend,256,3,7,256,false,true,true,false,true);
    run(backend,256,3,7,256,true,true,true,false,true);
    ggml_backend_free(backend);puts("sycl_convrot_arithmetic passed=30");
}
