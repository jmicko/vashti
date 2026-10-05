#include <array>
#include <cfenv>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>
#include <sys/resource.h>
#include "model/common/rope.hpp"
#include "whole-rope-policy.hpp"
namespace policy=ggml_sycl_whole_rope_policy;
static unsigned checks=0,graphs=0;
static void require(bool condition) {
    ++checks;
    if(!condition) {std::fprintf(stderr,"whole_rope_policy_error check=%u\n",checks);std::exit(1);}
}
struct Fixture {
    ggml_context* context=nullptr;
    ggml_cgraph* graph=nullptr;
    ggml_tensor *input=nullptr,*pe=nullptr,*output=nullptr,*add=nullptr;
    int terminal=-1;
    explicit Fixture(int length=3,bool interleaved=true) {
        context=ggml_init({4*1024*1024,nullptr,true});require(context);
        input=ggml_new_tensor_4d(context,GGML_TYPE_F32,128,32,length,1);
        pe=ggml_new_tensor_4d(context,GGML_TYPE_F32,2,64,length,2);
        output=Rope::apply_rope_with_prepared_pe(context,input,pe,interleaved);
        graph=ggml_new_graph_custom(context,256,false);
        ggml_build_forward_expand(graph,output);
        for(int i=0;i<graph->n_nodes;++i) if(graph->nodes[i]->op==GGML_OP_ADD) {add=graph->nodes[i];terminal=i;}
        require(add&&terminal>=0);
        std::vector<ggml_tensor*> tensors;
        for(int i=0;i<graph->n_leafs;++i)tensors.push_back(graph->leafs[i]);
        for(int i=0;i<graph->n_nodes;++i)tensors.push_back(graph->nodes[i]);
        uintptr_t next=0x100000000ull;
        for(auto* tensor:tensors) if(!tensor->view_src) {tensor->data=reinterpret_cast<void*>(next);next+=0x40000000ull;}
        for(auto* tensor:tensors) if(tensor->view_src) tensor->data=static_cast<char*>(tensor->view_src->data)+tensor->view_offs;
        ++graphs;
    }
    ~Fixture(){ggml_free(context);}
    policy::Plan plan(size_t alignment=128) const {return policy::find(graph,terminal,alignment);}
};
static void guards() {
    require(policy::capable(128,128));require(policy::capable(1024,1024));
    require(!policy::capable(0,128));require(!policy::capable(128,0));
    require(!policy::capable(127,1024));require(!policy::capable(1024,127));
    {policy::Plans plans;
     for(int i=0;i<64;++i){policy::Plan p;p.reason=policy::Reason::eligible;p.begin=32*i;p.end=p.begin+1;require(plans.append(p));
        if(i==62)require(plans.count==63&&!plans.discarded);}
     require(plans.count==64&&!plans.discarded);
     policy::Plan overflow;overflow.reason=policy::Reason::eligible;overflow.begin=2048;overflow.end=2049;
     require(!plans.append(overflow)&&plans.count==0&&plans.discarded);require(!plans.append(overflow));}
    {policy::Plans plans;policy::Plan p;p.reason=policy::Reason::eligible;p.begin=0;p.end=1;require(plans.append(p));
     require(!plans.append(p)&&plans.count==0&&plans.discarded);}
    {policy::Plans plans;policy::Plan p;p.reason=policy::Reason::eligible;p.begin=0;p.end=policy::max_window_nodes;
     require(!plans.append(p)&&plans.count==0);}
    for(int length:{1,3,17,65,policy::max_length}) {
        Fixture fixture(length);
        const auto p=fixture.plan();
        if(!p) std::fprintf(stderr,"whole_rope_policy_decline reason=%u\n",unsigned(p.reason));
        require(bool(p)&&p.source==fixture.input&&p.coefficient==fixture.pe&&p.destination==fixture.add&&p.output==fixture.output);
        unsigned computes=0;
        for(int i=p.begin;i<=p.end;++i) if(!policy::bytes::metadata(fixture.graph->nodes[i]->op)) ++computes;
        require(computes==7&&p.end==fixture.terminal);
    }
    {Fixture f(3,false);require(!f.plan());}
    {Fixture f;ggml_set_output(f.output);require(!f.plan());}
    {Fixture f;require(!policy::find(nullptr,0,128));require(!policy::find(f.graph,-1,128));require(!f.plan(0));require(!f.plan(96));}
    {Fixture f;f.graph->order=GGML_CGRAPH_EVAL_ORDER_RIGHT_TO_LEFT;require(!f.plan());}
    {Fixture f;f.graph->n_nodes=policy::max_graph_nodes+1;require(!f.plan());}
    {Fixture f;f.graph->n_leafs=policy::max_graph_nodes+1;require(!f.plan());}
    {Fixture f;f.graph->grad_accs=f.graph->nodes;require(!f.plan());}
    {Fixture f;f.input->nb[1]+=1;require(!f.plan());}
    {Fixture f;f.input->nb[2]=SIZE_MAX;require(!f.plan());}
    {Fixture f;f.input->ne[3]=2;require(!f.plan());}
    {Fixture f;f.pe->nb[1]+=4;require(!f.plan());}
    // Every eliminated tensor denies explicit outputs/cuts, source-slot
    // escapes, shape changes and physical view-offset changes independently.
    {Fixture baseline;const int count=baseline.graph->n_nodes;
     for(int i=0;i<count-1;++i) {
        {Fixture f;auto* n=f.graph->nodes[i];n->flags|=GGML_TENSOR_FLAG_OUTPUT;require(!f.plan());}
        {Fixture f;auto* n=f.graph->nodes[i];std::strcpy(n->name,"ggml_runner_cut:synthetic|");require(!f.plan());}
        {Fixture f;auto* n=f.graph->nodes[i];n->src[2]=f.input;require(!f.plan());}
        {Fixture f;auto* n=f.graph->nodes[i];n->type=GGML_TYPE_F16;require(!f.plan());}
        {Fixture f;auto* n=f.graph->nodes[i];if(n->view_src) {n->view_offs+=4;require(!f.plan());}}
        {Fixture f;auto* n=f.graph->nodes[i];if(!policy::bytes::metadata(n->op)) {n->flags&=~GGML_TENSOR_FLAG_COMPUTE;require(!f.plan());}}
     }
    }
    {Fixture f;auto* duplicate=f.graph->nodes[1];f.graph->nodes[f.graph->n_nodes++]=duplicate;require(!f.plan());}
    {Fixture f;std::swap(f.graph->nodes[f.terminal-1],f.graph->nodes[f.terminal]);require(!f.plan());}
    {Fixture f;auto* rogue=ggml_dup(f.context,f.input);rogue->data=reinterpret_cast<void*>(0x2000000000ull);
     for(int i=f.graph->n_nodes;i>2;--i) f.graph->nodes[i]=f.graph->nodes[i-1];
     f.graph->nodes[2]=rogue;++f.graph->n_nodes;++f.terminal;require(!f.plan());}
    for(int slot=0;slot<GGML_MAX_SRC;++slot) {
        Fixture f;auto* escape=ggml_new_tensor_1d(f.context,GGML_TYPE_F32,1);
        escape->src[slot]=f.add->src[0];f.graph->leafs[f.graph->n_leafs++]=escape;require(!f.plan());
    }
    {Fixture f;auto* view=ggml_view_1d(f.context,f.add->src[0],1,0);ggml_build_forward_expand(f.graph,view);require(!f.plan());}
    {Fixture f;auto* view=ggml_view_1d(f.context,f.output,1,0);ggml_build_forward_expand(f.graph,view);require(bool(f.plan()));}
    {Fixture f;auto* view=ggml_view_1d(f.context,f.output,1,0);ggml_build_forward_expand(f.graph,view);view->data=static_cast<char*>(f.output->data)-4;require(!f.plan());}
    for(ggml_tensor* Fixture::* member:{&Fixture::input,&Fixture::pe}) {
        Fixture f;auto* t=f.*member;t->data=f.add->data;
        for(int i=0;i<f.graph->n_nodes;++i) {auto* n=f.graph->nodes[i];if(n->view_src==t)n->data=static_cast<char*>(t->data)+n->view_offs;}
        const auto p=f.plan();require(!p&&p.reason==policy::Reason::overlap);
    }
    {Fixture f;auto* owner=ggml_new_tensor_4d(f.context,GGML_TYPE_F32,128,32,3,1);
     owner->data=static_cast<char*>(f.input->data)-128;owner->nb[2]=SIZE_MAX;f.input->view_src=owner;require(!f.plan());}
    {Fixture f;f.input->data=reinterpret_cast<void*>(UINTPTR_MAX-127);require(!f.plan());}
    {Fixture f;auto* pe0=f.add->src[0]->src[1];std::strcpy(pe0->name,"ggml_runner_cut:synthetic|pe| (view)");require(bool(f.plan()));
     std::strcpy(pe0->name,"ggml_runner_cut:synthetic|pe|");require(!f.plan());}
}
static uint32_t word(float value){uint32_t result;std::memcpy(&result,&value,4);return result;}
static float value(uint32_t bits){float result;std::memcpy(&result,&bits,4);return result;}
__attribute__((noinline)) static float multiply(float a,float b){volatile float result=a*b;return result;}
__attribute__((noinline)) static float add(float a,float b){volatile float result=a+b;return result;}
static uint32_t mix(uint32_t i){i^=i>>16;i*=0x7feb352du;i^=i>>15;i*=0x846ca68bu;return i^(i>>16);}
static std::vector<float> pattern(size_t count,unsigned kind,unsigned shift) {
    const uint32_t special[]={0,0x80000000u,0x3f800000u,0xbf800000u,0x3f800001u,0x3f7fffffu,
        0x00800000u,1,0x807fffffu,0x7f800000u,0xff800000u,0x7fc01234u,0xffc05678u,0x7fa02468u,0xffa13579u};
    std::vector<float> result(count);
    for(size_t i=0;i<count;++i) {const auto bits=mix(uint32_t(i)+shift);
        result[i]=kind==1?value(special[(i+shift)%15]):value((bits&0x807fffffu)|((126+(bits%3))<<23));}
    return result;
}
static std::vector<float> coefficients(int length,unsigned kind) {
    if(kind!=2)return pattern(256*size_t(length),kind,3);
    // Public synthetic Qwen multi-axis geometry/positions, independent of user
    // content. Model theta/axes match the pinned Qwen2.1 configuration.
    auto ids=Rope::gen_qwen_image_ids(1,2,2,2,1,length-1,{},Rope::RefIndexMode::INCREASE);
    require(ids.size()==size_t(length));
    auto raw=Rope::embed_nd(ids,1,10000.f,{16,56,56});
    std::vector<float> prepared(raw.size());
    for(int t=0;t<length;++t)for(int j=0;j<64;++j)for(int c=0;c<2;++c)for(int r=0;r<2;++r)
        prepared[r+2*j+128*t+128*length*c]=raw[c+2*r+4*j+256*t];
    return prepared;
}
static void address_arithmetic() {
    require(std::fegetround()==FE_TONEAREST);
    for(int length:{1,3,17,65})for(unsigned kind=0;kind<3;++kind)for(unsigned padded=0;padded<2;++padded) {
        const size_t nb1=512+128*padded,nb2=32*nb1+128*padded;
        auto x=pattern((size_t(length-1)*nb2+31*nb1+512)/4,kind==1?1:0,17);
        auto pe=coefficients(length,kind);
        const size_t count=4096*size_t(length);
        std::vector<float> cont(count),split(count),products0(count),products1(count);
        for(int h=0;h<32;++h)for(int t=0;t<length;++t)for(int d=0;d<128;++d)
            cont[d+128*(t+length*h)]=x[(h*nb1+t*nb2)/4+d];
        for(int h=0;h<32;++h)for(int t=0;t<length;++t)for(int j=0;j<64;++j)for(int e=0;e<2;++e)
            split[j+64*(t+length*h)+2048*length*e]=cont[2*j+e+128*(t+length*h)];
        for(int h=0;h<32;++h)for(int t=0;t<length;++t)for(int j=0;j<64;++j)for(int r=0;r<2;++r) {
            const size_t out=2*j+r+128*(t+length*h),coefficient=2*j+r+128*t;
            const size_t half=j+64*(t+length*h);
            products0[out]=multiply(split[half],pe[coefficient]);
            products1[out]=multiply(split[half+2048*length],pe[coefficient+128*length]);
        }
        for(int h=0;h<32;++h)for(int t=0;t<length;++t)for(int j=0;j<64;++j)for(int r=0;r<2;++r) {
            const size_t out=2*j+r+128*(t+length*h),coefficient=2*j+r+128*t;
            const size_t original=(h*nb1+t*nb2)/4+2*j;
            const auto p0=multiply(x[original],pe[coefficient]);
            const auto p1=multiply(x[original+1],pe[coefficient+128*length]);
            require(word(products0[out])==word(p0)&&word(products1[out])==word(p1));
            require(word(add(products0[out],products1[out]))==word(add(p0,p1)));
        }
    }
}
int main(){rlimit no_core{0,0};if(setrlimit(RLIMIT_CORE,&no_core))return 2;
    ggml_log_set([](ggml_log_level,const char*,void*){},nullptr);
    guards();address_arithmetic();
    std::printf("whole_rope_policy_completed checks=%u graphs=%u address_cases=24 device_proof=0\n",checks,graphs);
}
