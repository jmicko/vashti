// CPU-only allocator/real graph-cut binding checks. No GPU load/enumeration.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <sys/resource.h>
#include "ggml-alloc.h"
#include "ggml-cpu.h"
#include "core/segment_graph_bindings.h"
#include "core/runner_cache.h"
#include "model/common/rope.hpp"
#include "whole-rope-policy.hpp"
namespace policy=ggml_sycl_whole_rope_policy;
static unsigned checks=0,cases=0,eligible=0,alias_declines=0,produced_eligible=0,segmented=0,skipped_x_reuse=0;
static void require(bool value){++checks;if(!value){std::fprintf(stderr,"whole_rope_allocation_error case=%u check=%u\n",cases+1,checks);std::exit(1);}}
static int index(ggml_cgraph* graph,ggml_tensor* tensor){for(int i=0;i<graph->n_nodes;++i)if(graph->nodes[i]==tensor)return i;return -1;}
__attribute__((noinline)) static float multiply(float a,float b){volatile float result=a*b;return result;}
__attribute__((noinline)) static float ordered_add(float a,float b){volatile float result=a+b;return result;}
static void fixture(ggml_backend_t backend,int length,unsigned produced,bool segment) {
    auto* context=ggml_init({8*1024*1024,nullptr,true});require(context);
    auto* input=ggml_new_tensor_4d(context,GGML_TYPE_F32,128,32,length,1);
    auto* raw_pe=ggml_new_tensor_4d(context,GGML_TYPE_F32,2,2,64,length);
    auto* weight=ggml_new_tensor_1d(context,GGML_TYPE_F32,128);
    std::vector<float> w(128);for(unsigned i=0;i<128;++i)w[i]=float(i+1)/129.f;
    std::vector<float> x(size_t(ggml_nelements(input)));
    for(size_t i=0;i<x.size();++i)x[i]=float(int(i%37)-18)/19.f;
    auto ids=Rope::gen_qwen_image_ids(1,2,2,2,1,length-1,{},Rope::RefIndexMode::INCREASE);
    auto p=Rope::embed_nd(ids,1,10000.f,{16,56,56});
    auto inputs=ggml_backend_alloc_ctx_tensors(context,backend);require(inputs);
    ggml_backend_tensor_set(input,x.data(),0,ggml_nbytes(input));
    ggml_backend_tensor_set(raw_pe,p.data(),0,ggml_nbytes(raw_pe));
    ggml_backend_tensor_set(weight,w.data(),0,ggml_nbytes(weight));
    auto* prepared=Rope::prepare_rope_pe(context,raw_pe);
    if(segment){sd::ggml_graph_cut::mark_graph_cut(prepared,"synthetic","prepared");ggml_set_output(prepared);}
    auto* source=produced?ggml_dup(context,input):input;
    ggml_tensor* sibling=nullptr;
    if(produced==3||produced==4)source=ggml_norm(context,input,1.e-6f);
    if(produced==5)source=ggml_rms_norm(context,input,1.e-6f);
    if(produced==6)source=ggml_mul(context,ggml_rms_norm(context,input,1.e-6f),weight);
    if(produced==7){source=ggml_rms_norm(context,input,1.e-6f);sibling=ggml_mul(context,source,weight);ggml_set_output(sibling);}
    if(produced==2||produced==4)ggml_set_output(source);
    auto* rope=Rope::apply_rope_with_prepared_pe(context,source,prepared);
    // A real consumer makes RoPE's private MUL0 unprotected. Marking the
    // final reshape OUTPUT would propagate to MUL0 and is a separate decline.
    auto* output=ggml_dup(context,rope);ggml_set_output(output);
    auto* graph=ggml_new_graph_custom(context,256,false);ggml_build_forward_expand(graph,prepared);
    if(sibling)ggml_build_forward_expand(graph,sibling);
    ggml_build_forward_expand(graph,output);
    if(produced==5)require(!ggml_can_fuse(graph,index(graph,source),{GGML_OP_RMS_NORM,GGML_OP_MUL}));
    if(produced==6)require(ggml_can_fuse(graph,index(graph,source->src[0]),{GGML_OP_RMS_NORM,GGML_OP_MUL}));
    if(produced==7)require(!ggml_can_fuse(graph,index(graph,source),{GGML_OP_RMS_NORM,GGML_OP_MUL}));
    auto* allocator=ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));require(allocator);
    require(ggml_gallocr_alloc_graph(allocator,graph));
    require(ggml_backend_graph_compute(backend,graph)==GGML_STATUS_SUCCESS);ggml_backend_synchronize(backend);
    std::vector<float> original(size_t(ggml_nelements(output)));
    ggml_backend_tensor_get(output,original.data(),0,ggml_nbytes(output));
    sd::ggml_graph_cut::Plan cut_plan;
    sd::ggml_graph_cut::Segment prelude,steady;
    ggml_cgraph* executable=graph;ggml_context* segment_context=nullptr;
    sd::GraphCutTensorCache cache(backend);
    // This exercises the exact engine cache + SegmentGraphBindings path on
    // CPU, including topology removal/restoration and fresh cached PE data.
    sd::SegmentGraphBindings* binding=nullptr;
    std::array<ggml_tensor*,GGML_MAX_SRC> prepared_sources{};
    std::copy(std::begin(prepared->src),std::end(prepared->src),prepared_sources.begin());
    if(segment) {
        const int prepared_index=index(graph,prepared);require(prepared_index>=0);
        prelude.output_node_indices={prepared_index};prelude.future_cut_names.insert(prepared->name);
        require(cache.capture(graph,prelude,"synthetic")==GGML_STATUS_SUCCESS);
        for(int i=0;i<graph->n_nodes;++i)if(i>prepared_index)steady.internal_node_indices.push_back(i);
        sd::ggml_graph_cut::Segment::InputRef pe_ref;
        pe_ref.type=sd::ggml_graph_cut::Segment::INPUT_PREVIOUS_CUT;pe_ref.node_index=prepared_index;pe_ref.display_name=prepared->name;
        steady.input_refs.push_back(pe_ref);
        sd::ggml_graph_cut::Segment::InputRef input_ref;input_ref.type=sd::ggml_graph_cut::Segment::INPUT_EXTERNAL;
        for(int i=0;i<graph->n_leafs;++i)if(graph->leafs[i]==input)input_ref.leaf_index=i;
        require(input_ref.leaf_index>=0);steady.input_refs.push_back(input_ref);
        // DUP/NORM sources before prepared PE may precede this crude segment
        // split in left-to-right DFS; include every post-leaf executable source.
        for(int i=0;i<prepared_index;++i)if(graph->nodes[i]->op==GGML_OP_DUP||graph->nodes[i]->op==GGML_OP_NORM)
            steady.internal_node_indices.insert(steady.internal_node_indices.begin(),i);
        steady.output_node_indices={index(graph,output)};
        cut_plan.segments={prelude,steady};
        binding=new sd::SegmentGraphBindings(cache,cut_plan,graph);
        // reset() clears allocations, so restore these CPU bindings through a
        // fresh allocator after binding the real cached leaf.
        binding->reset(steady);require(binding->bind_cached_inputs(steady,"synthetic"));
        require(prepared->op==GGML_OP_NONE&&prepared->view_src==nullptr);
        for(auto* s:prepared->src)require(s==nullptr);
        executable=sd::ggml_graph_cut::build_segment_graph(graph,steady,&segment_context);
        ggml_gallocr_free(allocator);allocator=ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
        require(allocator&&ggml_gallocr_alloc_graph(allocator,executable));
        ++segmented;
    }
    unsigned routes=0;policy::Plan selected;
    for(int i=0;i<executable->n_nodes;++i)if(executable->nodes[i]->op==GGML_OP_ADD) {
        const auto plan=policy::find(executable,i,1);
        if(plan){++routes;++eligible;if(produced)++produced_eligible;selected=plan;}
        else if(plan.reason==policy::Reason::overlap)++alias_declines;
        if(produced==2||produced==4)require(bool(plan));
        if(plan) {
            policy::bytes::Span destination{},input_span{};
            require(policy::bytes::span(plan.destination,destination)&&policy::bytes::span(plan.source,input_span));
            require(!policy::bytes::overlaps(destination,input_span));
            unsigned ordered=0;for(int j=plan.begin;j<=plan.end;++j)if(!policy::bytes::metadata(executable->nodes[j]->op))++ordered;
            require(ordered==7);
            for(int j=plan.begin;j<plan.end;++j)if(!policy::bytes::metadata(executable->nodes[j]->op)) {
                policy::bytes::Span temporary{};require(policy::bytes::span(executable->nodes[j],temporary));
                if(policy::bytes::overlaps(temporary,input_span))++skipped_x_reuse;
            }
        }
    }
    if(!produced)require(routes==1);
    require(ggml_backend_graph_compute(backend,executable)==GGML_STATUS_SUCCESS);ggml_backend_synchronize(backend);
    std::vector<float> result(original.size());ggml_backend_tensor_get(output,result.data(),0,ggml_nbytes(output));
    require(result.size()==original.size()&&std::memcmp(result.data(),original.data(),result.size()*4)==0);
    if(selected) {
        // CPU simulation of the same terminal scheduling point using the real
        // gallocr binding. Earlier compute produces X/PE; all seven old ops
        // stay skipped, so intermediates allocated over X cannot clobber it.
        // This is address/lifetime evidence, not SYCL arithmetic/dispatch proof.
        auto prefix=ggml_graph_view(executable,0,selected.begin);
        require(ggml_backend_graph_compute(backend,&prefix)==GGML_STATUS_SUCCESS);ggml_backend_synchronize(backend);
        const auto* src=static_cast<const char*>(selected.source->data);
        const auto* coefficients=static_cast<const float*>(selected.coefficient->data);
        auto* destination=static_cast<float*>(selected.destination->data);
        for(int h=0;h<32;++h)for(int t=0;t<length;++t)for(int j=0;j<64;++j)for(int r=0;r<2;++r) {
            const auto* pair=reinterpret_cast<const float*>(src+h*selected.source->nb[1]+t*selected.source->nb[2]+8*j);
            const size_t pe_index=r+2*j+128*t;
            const auto p0=multiply(pair[0],coefficients[pe_index]);
            const auto p1=multiply(pair[1],coefficients[pe_index+128*length]);
            destination[r+2*j+128*(t+length*h)]=ordered_add(p0,p1);
        }
        auto suffix=ggml_graph_view(executable,selected.end+1,executable->n_nodes);
        require(ggml_backend_graph_compute(backend,&suffix)==GGML_STATUS_SUCCESS);ggml_backend_synchronize(backend);
        ggml_backend_tensor_get(output,result.data(),0,ggml_nbytes(output));
        require(std::memcmp(result.data(),original.data(),result.size()*4)==0);
    }
    if(binding){binding->restore();require(prepared->op==GGML_OP_CONT);
        for(unsigned i=0;i<GGML_MAX_SRC;++i)require(prepared->src[i]==prepared_sources[i]);
        delete binding;}
    ggml_backend_synchronize(backend);cache.clear();ggml_gallocr_free(allocator);
    if(segment_context)ggml_free(segment_context);
    ggml_backend_buffer_free(inputs);ggml_free(context);++cases;
}
int main(){rlimit no_core{0,0};if(setrlimit(RLIMIT_CORE,&no_core))return 2;
    ggml_log_set([](ggml_log_level,const char*,void*){},nullptr);
    auto backend=ggml_backend_cpu_init();require(backend);ggml_backend_cpu_set_n_threads(backend,1);
    for(int length:{1,3,17,65})for(unsigned produced=0;produced<8;++produced)for(bool segment:{false,true})fixture(backend,length,produced,segment);
    ggml_backend_synchronize(backend);ggml_backend_free(backend);
    require(eligible>0&&produced_eligible>0&&segmented==32&&skipped_x_reuse>0);
    std::printf("whole_rope_allocation_completed checks=%u cases=%u eligible=%u produced_eligible=%u alias_declines=%u skipped_x_reuse=%u segmented=%u gpu=0\n",checks,cases,eligible,produced_eligible,alias_declines,skipped_x_reuse,segmented);
}
