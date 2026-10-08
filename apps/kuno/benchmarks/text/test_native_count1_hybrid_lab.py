"""Host-only rejection controls. These synthetic documents are not device receipts."""
import copy
import importlib.util
import io
import json
import os
import pathlib
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

HERE = pathlib.Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("native_hybrid_lab", HERE / "run-native-count1-hybrid-lab.py")
LAB = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(LAB)


def receipt(card, cap):
    # Historical real native inventory, altered solely for validator tests.
    # No test output is saved as a hybrid device acceptance receipt.
    value = json.loads((HERE / f"receipts/rust-native-model-full-{card}-20261005.json").read_text())["report"]
    value.update(fixture=LAB.MODE, prompt_tokens=15, native_public_chunk_partition=[8, 7],
                 public_generated_ids=[9419, 0, 2500, 628, 353, 1438, 488, 3242],
                 native_decode=True, native_decode_steps=7, scalar_decode_steps=0,
                 native_decode_projection_tile=1, native_decode_kernels="single-q4-all",
                 observed_native_decode_kernels="single-q4-all", resident_private_owned_bytes=818413568,
                 count1_hybrid_cpu_miss_cap=cap, observed_count1_hybrid_cpu_miss_cap=cap,
                 completed_cpu_misses=17 if cap else 0, fresh_completed_cpu_misses=3 if cap else 0,
                 count1_hybrid_cpu_gpu_overlap=False, observed_count1_hybrid_cpu_gpu_overlap=False, count1_hybrid_cpu_workers="scoped", observed_count1_hybrid_cpu_workers="scoped")
    for audit in value["owned_zero_audits"]:
        audit.update(main_buffers=1378, main_bytes=818413568, source_buffers=0, source_bytes=0)
    return value


def completion(cap):
    return "\n".join((LAB.GUARD,
        "public_mid_token_cancel layer=0 erased=true",
        "public_mid_token_cancel layer=3 erased=true",
        "public_resident_cancel_erased=true poisoned_retry_rejected=true fresh_first_token_matches=true",
        "public_resident_owned_cleanup=true context_retired=false",
        *LAB.BASE.TILE_ONE_COMPLETION, *LAB.BASE.kernel_completion("single-q4-all"),
        *LAB.completion(cap)))


class HybridLabTests(unittest.TestCase):
    def test_source_advice_requires_actual_bounded_evidence_and_static_bank_mapping(self):
        value=receipt("intel",10)
        value.update(reclaim_gpu_sources=True,gpu_source_advised_bytes=19*1024**3,gpu_source_advice_ms=150,mapped_weights=True,observed_mapped_weights=True)
        LAB.validate_receipt(value,"intel",10,completion(10),mapped=True,reclaim_gpu_sources=True)
        for field,bads in [("reclaim_gpu_sources",[False,1,"true",None]),("gpu_source_advised_bytes",[0,True,33*1024**3,None]),("gpu_source_advice_ms",[-1,True,None])]:
            for bad in bads:
                wrong=copy.deepcopy(value);wrong[field]=bad
                with self.assertRaises(ValueError):LAB.validate_receipt(wrong,"intel",10,completion(10),mapped=True,reclaim_gpu_sources=True)
        with self.assertRaises(ValueError):LAB.validate_receipt(value,"intel",5,completion(5),mapped=True,reclaim_gpu_sources=True)
        with self.assertRaises(ValueError):LAB.worker_environment("/original","/receipt",reclaim_gpu_sources=True)
        env=LAB.worker_environment("/original","/receipt",mapped=True,reclaim_gpu_sources=True)
        self.assertEqual(env["KUNO_LAB_RECLAIM_GPU_SOURCES"],"1")
        with mock.patch.dict(os.environ,{"KUNO_LAB_RECLAIM_GPU_SOURCES":"1"}):
            self.assertEqual(LAB.worker_environment("/original","/receipt")["KUNO_LAB_RECLAIM_GPU_SOURCES"],"0")
    def test_cpu_fma_arithmetic_is_explicit_typed_and_separate_from_q8(self):
        value=receipt("intel",10)
        value.update(cpu_fma_arithmetic=True,count1_hybrid_cpu_workers="pooled",observed_count1_hybrid_cpu_workers="pooled")
        LAB.validate_receipt(value,"intel",10,completion(10),workers="pooled",cpu_fma=True)
        for bad in (None,False,1,"true"):
            wrong=copy.deepcopy(value);wrong["cpu_fma_arithmetic"]=bad
            with self.assertRaises(ValueError):LAB.validate_receipt(wrong,"intel",10,completion(10),workers="pooled",cpu_fma=True)
        with self.assertRaises(ValueError):LAB.validate_receipt(value,"intel",10,completion(10),workers="pooled")
        with self.assertRaises(ValueError):LAB.validate_receipt(value,"intel",10,completion(10),workers="pooled",cpu_fma=True,cpu_q8=True)
        with self.assertRaises(ValueError):LAB.validate_receipt(value,"intel",10,completion(10),cpu_fma=True)
    def test_inline_dispatch_requires_compiled_runtime_evidence(self):
        value=receipt("intel",10)
        value.update(device_dispatch="inline",observed_inline_device_dispatch=True)
        output=completion(10)+"\ngpu_device_dispatch mode=inline blocking=true"
        LAB.validate_receipt(value,"intel",10,output,dispatch="inline")
        for change in ({"device_dispatch":"threaded"},{"observed_inline_device_dispatch":False}):
            bad=dict(value,**change)
            with self.assertRaises(ValueError):
                LAB.validate_receipt(bad,"intel",10,output,dispatch="inline")
        with self.assertRaises(ValueError):
            LAB.validate_receipt(value,"intel",10,completion(10),dispatch="inline")
        with self.assertRaises(ValueError):
            LAB.validate_receipt(value,"intel",0,output,dispatch="inline")
        with self.assertRaises(ValueError):
            LAB.worker_environment("/original/model","/public/receipt",dispatch="unknown")
        env=LAB.worker_environment("/original/model","/public/receipt",dispatch="inline")
        self.assertEqual(env["KUNO_LAB_DEVICE_DISPATCH"],"inline")

    def test_batched_info_requires_compact_slabs_and_independent_runtime_marker(self):
        value=receipt("intel",10)
        value.update(compact_info=True,info_slab_bytes=4096,info_batch_upload=True)
        output=completion(10)+"\nprivate_info_batch_upload enabled=true"
        LAB.validate_receipt(value,"intel",10,output,compact_info=True,info_batch_upload=True)
        with self.assertRaises(ValueError):
            LAB.validate_receipt(value,"intel",10,completion(10),compact_info=True,info_batch_upload=True)
        with self.assertRaises(ValueError):
            LAB.worker_environment("/original/model","/public/receipt",info_batch_upload=True)
        env=LAB.worker_environment("/original/model","/public/receipt",compact_info=True,info_batch_upload=True)
        self.assertEqual(env["KUNO_LAB_INFO_BATCH_UPLOAD"],"1")

    def test_shared_kernels_require_original_numeric_probes_and_all_dense_calls(self):
        value = receipt("intel", 10)
        value.update(fast_gpu=True, observed_fast_dense_calls=5264,
                     gpu_router=True, observed_gpu_router_calls=336,
                     next_shared_kernels=True, observed_shared_dense_calls=5264)
        markers = ("rust_gpu_router cases=6 ranks=10 numeric_and_owning_checks=true",
                   "next_shared_original_rows cases=6 numeric_and_owning_checks=true",
                   "next_shared_gpu_router cases=8 numeric_and_owning_checks=true")
        output = completion(10) + "\n" + "\n".join(markers)
        LAB.validate_receipt(value, "intel", 10, output, fast=True, gpu_router=True, shared=True)
        for bad in (None, True, 0, 5263, 5265, -1):
            wrong = copy.deepcopy(value)
            wrong["observed_shared_dense_calls"] = bad
            with self.assertRaises(ValueError):
                LAB.validate_receipt(wrong, "intel", 10, output, fast=True, gpu_router=True, shared=True)
        for marker in markers:
            with self.assertRaises(ValueError):
                LAB.validate_receipt(value, "intel", 10, output.replace(marker, ""), fast=True, gpu_router=True, shared=True)
        for kwargs in ({"shared": True}, {"shared": True, "fast": True},
                       {"shared": True, "fast": True, "gpu_router": True, "fast_fault": True},
                       {"shared": 1, "fast": True, "gpu_router": True}):
            with self.assertRaises(ValueError):
                LAB.worker_environment("/original/model", "/public/receipt", **kwargs)
        env = LAB.worker_environment("/original/model", "/public/receipt", fast=True, gpu_router=True, shared=True)
        self.assertEqual(env["KUNO_LAB_NEXT_SHARED"], "1")
        value.update(next_shared_gdn=True, observed_shared_gdn_calls=252)
        output += "\nnext_shared_gdn cases=3 numeric_and_owning_checks=true"
        LAB.validate_receipt(value, "intel", 10, output, fast=True, gpu_router=True, shared=True, shared_gdn=True)
        for bad in (0, 251, 253, True, None):
            wrong=copy.deepcopy(value);wrong["observed_shared_gdn_calls"]=bad
            with self.assertRaises(ValueError):
                LAB.validate_receipt(wrong, "intel", 10, output, fast=True, gpu_router=True, shared=True, shared_gdn=True)
        with self.assertRaises(ValueError):
            LAB.worker_environment("/original/model", "/public/receipt", shared_gdn=True)

    def test_deferred_epochs_require_actual_bridges_late_cancel_and_five_owning_audits(self):
        value=receipt("intel",10)
        value.update(compact_info=True,info_slab_bytes=4096,defer_router_io=True,observed_defer_router_io=True,observed_deferred_router_reads=336,io_token_checkpoint=True)
        value["owned_zero_audits"].append(copy.deepcopy(value["owned_zero_audits"][-1]))
        output=completion(10)+"\npublic_count1_hybrid_mid_token_cancel cap=10 layer=47 erased=true"
        LAB.validate_receipt(value,"intel",10,output,compact_info=True,deferred=True)
        for field,bads in (("defer_router_io",(False,1,None)),("observed_defer_router_io",(False,1,None)),("observed_deferred_router_reads",(335,337,True,None)),("io_token_checkpoint",(False,1,None))):
            for bad in bads:
                wrong=copy.deepcopy(value);wrong[field]=bad
                with self.assertRaises(ValueError):LAB.validate_receipt(wrong,"intel",10,output,compact_info=True,deferred=True)
        wrong=copy.deepcopy(value);wrong["owned_zero_audits"].pop()
        with self.assertRaises(ValueError):LAB.validate_receipt(wrong,"intel",10,output,compact_info=True,deferred=True)
        with self.assertRaises(ValueError):LAB.validate_receipt(value,"intel",10,completion(10),compact_info=True,deferred=True)
        with self.assertRaises(ValueError):LAB.validate_receipt(value,"intel",0,output,compact_info=True,deferred=True)
        with self.assertRaises(ValueError):LAB.worker_environment("/original/model","/public/receipt",deferred=True)
        env=LAB.worker_environment("/original/model","/public/receipt",compact_info=True,deferred=True)
        self.assertEqual(env["KUNO_LAB_DEFER_ROUTER_IO"],"1")
        self.assertEqual(env["KUNO_LAB_IO_TOKEN_CHECKPOINT"],"0") # Original flag cannot silently select protected policy.

    def test_fused_finite_requires_every_dispatch_and_all_format_fault_probes(self):
        value=receipt("intel",10)
        value.update(fast_gpu=True,observed_fast_dense_calls=5264,fast_gpu_fault=True,observed_fast_dense_fault_calls=5264)
        output=completion(10)+"\n"+"\n".join(f"rust_fast_packed_fused_fault dtype={dtype} cases=4 passed=true" for dtype in (0,1,30,12,6,14,8))
        LAB.validate_receipt(value,"intel",10,output,fast=True,fast_fault=True)
        for bad in (True,0,5263,5265,None):
            wrong=copy.deepcopy(value);wrong["observed_fast_dense_fault_calls"]=bad
            with self.assertRaises(ValueError):LAB.validate_receipt(wrong,"intel",10,output,fast=True,fast_fault=True)
        for dtype in (0,1,30,12,6,14,8):
            wrong=output.replace(f"rust_fast_packed_fused_fault dtype={dtype} cases=4 passed=true","")
            with self.assertRaises(ValueError):LAB.validate_receipt(value,"intel",10,wrong,fast=True,fast_fault=True)
        with self.assertRaises(ValueError):LAB.worker_environment("/original/model","/public/receipt",fast_fault=True)

    def test_compact_info_and_larger_submission_groups_require_typed_matching_bounds(self):
        for compact in (False,True):
            for batch in (False,True):
                value=receipt("intel",10)
                value.update(compact_info=compact,info_slab_bytes=4096 if compact else 0,batch_tasks=batch,runtime_tasks_max=256 if batch else 32)
                LAB.validate_receipt(value,"intel",10,completion(10),compact_info=compact,batch_tasks=batch)
                for field in ("compact_info","info_slab_bytes","batch_tasks","runtime_tasks_max"):
                    for bad in (None,"1",True if type(value[field]) is int else 1):
                        wrong=copy.deepcopy(value);wrong[field]=bad
                        with self.assertRaises(ValueError):LAB.validate_receipt(wrong,"intel",10,completion(10),compact_info=compact,batch_tasks=batch)
                env=LAB.worker_environment("/original/model","/public/receipt",compact_info=compact,batch_tasks=batch)
                self.assertEqual(env["KUNO_LAB_INFO_SLAB_BYTES"],"4096" if compact else "0")
                self.assertEqual(env["KUNO_LAB_BATCH_TASKS"],"1" if batch else "0")

    def test_q8_requires_pooled_workers_and_exact_completed_arithmetic_evidence(self):
        value = receipt("intel",10)
        value.update(count1_hybrid_cpu_workers="pooled",observed_count1_hybrid_cpu_workers="pooled",cpu_q8_activations=True,observed_cpu_q8_activations=True)
        LAB.validate_receipt(value,"intel",10,completion(10),workers="pooled",cpu_q8=True)
        for field in ("cpu_q8_activations","observed_cpu_q8_activations"):
            for bad in (None,False,1,"true"):
                wrong=copy.deepcopy(value);wrong[field]=bad
                with self.assertRaises(ValueError):
                    LAB.validate_receipt(wrong,"intel",10,completion(10),workers="pooled",cpu_q8=True)
        with self.assertRaises(ValueError):
            LAB.worker_environment("/original/model","/public/receipt",cpu_q8=True)
        env=LAB.worker_environment("/original/model","/public/receipt",workers="pooled",cpu_q8=True)
        self.assertEqual(env["KUNO_LAB_CPU_Q8"],"1")

    def test_larger_model_cache_requires_mapping_and_actual_bounded_no_swap_cgroup(self):
        spec=importlib.util.spec_from_file_location("protected_lab",HERE/"protected-lab-exec.py")
        guard=importlib.util.module_from_spec(spec);spec.loader.exec_module(guard)
        env={"KUNO_LAB_LARGE_WEIGHT_CACHE":"1","KUNO_LAB_MAPPED_WEIGHTS":"1"}
        args=["/frozen/binary","intel",LAB.MODE,"10"]
        self.assertEqual(guard.memory_limit(env,args),48*1024**3)
        for cap in range(11):
            self.assertEqual(guard.memory_limit(env,[*args[:3],str(cap)]),48*1024**3)
        self.assertEqual(guard.memory_limit({},args),8*1024**3)
        for bad in ("auto","48","",True):
            with self.assertRaises(ValueError):guard.memory_limit(dict(env,KUNO_LAB_LARGE_WEIGHT_CACHE=bad),args)
        for badargs in (args[:-1],[*args,"arbitrary"],[args[0],"cpu",LAB.MODE,"10"],[args[0],"intel","arbitrary","10"],*([*args[:3],cap] for cap in ("11","-1","01"," 1",True))):
            with self.assertRaises(ValueError):guard.memory_limit(env,badargs)
        with self.assertRaises(ValueError):guard.memory_limit(dict(env,KUNO_LAB_MAPPED_WEIGHTS="0"),args)
        value=receipt("intel",10);value.update(mapped_weights=True,observed_mapped_weights=True,large_weight_cache=True)
        output=completion(10).replace("8589934592","51539607552")
        LAB.validate_receipt(value,"intel",10,output,mapped=True,large_cache=True)
        for wrong in (completion(10),output.replace("51539607552","max"),output.replace("51539607552","55834574848")):
            with self.assertRaises(ValueError):LAB.validate_receipt(value,"intel",10,wrong,mapped=True,large_cache=True)
        with self.assertRaises(ValueError):LAB.worker_environment("/original/model","/public/receipt",large_cache=True)

    def test_fast_gpu_requires_actual_dispatches_and_separate_long_selection(self):
        value = receipt("intel",10)
        value.update(fast_gpu=True,observed_fast_dense_calls=5264,long_fixture=False)
        LAB.validate_receipt(value,"intel",10,completion(10),fast=True)
        for field, bads in (("fast_gpu",(False,1,"true")),("observed_fast_dense_calls",(0,True,"5264",100001)),("long_fixture",(True,1,"false"))):
            for bad in bads:
                wrong=copy.deepcopy(value);wrong[field]=bad
                with self.assertRaises(ValueError):
                    LAB.validate_receipt(wrong,"intel",10,completion(10),fast=True)
        with mock.patch.dict(os.environ,{"KUNO_LAB_FAST_GPU":"auto","KUNO_LAB_LONG_FIXTURE":"auto"}):
            env=LAB.worker_environment("/original/model","/public/receipt",fast=True,long=False)
        self.assertEqual(env["KUNO_LAB_FAST_GPU"],"1")
        self.assertEqual(env["KUNO_LAB_LONG_FIXTURE"],"0")

    def test_mapping_and_profiling_are_explicit_observed_and_never_inherited(self):
        for mapped in (False, True):
            for profile in (False, True):
                value = receipt("intel", 10)
                value.update(mapped_weights=mapped, observed_mapped_weights=mapped, kernel_profile=profile)
                LAB.validate_receipt(value, "intel", 10, completion(10), mapped=mapped, profile=profile)
                for field, selected in (("mapped_weights", mapped), ("observed_mapped_weights", mapped), ("kernel_profile", profile)):
                    for bad in (0, "false", None, not selected):
                        wrong = copy.deepcopy(value)
                        wrong[field] = bad
                        with self.assertRaises(ValueError):
                            LAB.validate_receipt(wrong, "intel", 10, completion(10), mapped=mapped, profile=profile)
                with mock.patch.dict(os.environ, {"KUNO_LAB_MAPPED_WEIGHTS":"auto","KUNO_LAB_KERNEL_PROFILE":"auto"}):
                    env = LAB.worker_environment("/original/model", "/public/receipt", mapped=mapped, profile=profile)
                self.assertEqual(env["KUNO_LAB_MAPPED_WEIGHTS"],"1" if mapped else "0")
                self.assertEqual(env["KUNO_LAB_KERNEL_PROFILE"],"1" if profile else "0")
        for bad in (1,"1",None):
            with self.assertRaises(ValueError):
                LAB.worker_environment("/original/model", "/public/receipt", mapped=bad)

    def test_gpu_routing_pins_device_controls_actual_calls_and_finite_fixture(self):
        value=receipt("intel",10)
        value.update(gpu_router=True,observed_gpu_router_calls=336)
        output=completion(10)+"\nrust_gpu_router cases=6 ranks=10 numeric_and_owning_checks=true\n"
        LAB.validate_receipt(value,"intel",10,output,gpu_router=True)
        for field,bads in (("gpu_router",(False,1,"true")),("observed_gpu_router_calls",(0,True,"336",335,337))):
            for bad in bads:
                wrong=copy.deepcopy(value);wrong[field]=bad
                with self.assertRaises(ValueError):LAB.validate_receipt(wrong,"intel",10,output,gpu_router=True)
        with self.assertRaises(ValueError):LAB.validate_receipt(value,"intel",10,completion(10),gpu_router=True)
        with self.assertRaises(ValueError):LAB.validate_receipt(value,"intel",0,output,gpu_router=True)
        with mock.patch.dict(os.environ,{"KUNO_LAB_GPU_ROUTER":"auto"}):
            self.assertEqual(LAB.worker_environment("/original/model","/public/receipt",gpu_router=False)["KUNO_LAB_GPU_ROUTER"],"0")
            self.assertEqual(LAB.worker_environment("/original/model","/public/receipt",gpu_router=True)["KUNO_LAB_GPU_ROUTER"],"1")

    def test_metadata_requires_real_warm_lookups_protected_mode_and_explicit_policy(self):
        value = receipt("intel", 10)
        value.update(retained_metadata=True, observed_retained_metadata_calls=1200)
        LAB.validate_receipt(value, "intel", 10, completion(10), metadata=True)
        for field, bads in (("retained_metadata", (False, 1, "true")),
                            ("observed_retained_metadata_calls", (None, True, "1200", 0, -1, 1000001))):
            for bad in bads:
                wrong=copy.deepcopy(value);wrong[field]=bad
                with self.assertRaises(ValueError):
                    LAB.validate_receipt(wrong,"intel",10,completion(10),metadata=True)
        with self.assertRaises(ValueError):LAB.validate_receipt(value,"intel",0,completion(0),metadata=True)
        with mock.patch.dict(os.environ, {"KUNO_LAB_RETAINED_METADATA":"auto"}):
            for selected in (False,True):
                self.assertEqual(LAB.worker_environment("/original/model","/public/receipt",metadata=selected)["KUNO_LAB_RETAINED_METADATA"],"1" if selected else "0")

    def test_expanded_banks_pin_the_entire_device_budget_without_changing_other_settings(self):
        for banks in ("intel72","intel96","intel128"):
            value=receipt("intel",10)
            value.update(count1_bank_profile=banks, expert_bank_slots={"intel72":72,"intel96":96,"intel128":128}[banks],
                         expert_bank_segments=[72] if banks=="intel72" else [64,64] if banks=="intel128" else [64,32])
            if banks=="intel96":
                value.update(block_moe=True,explicit_static_cache=True,
                             expert_bank_accounted_limit_bytes=18*1024**3,static_cache_budget_bytes=6*1024**3)
            if banks=="intel128":
                value.update(block_moe=True,explicit_static_cache=True,expert_bank_accounted_limit_bytes=24*1024**3,static_cache_budget_bytes=5*1024**3,device_reserved_limit_bytes=31*1024**3)
            LAB.validate_receipt(value,"intel",10,completion(10),banks=banks)
            for field in ("expert_bank_slots","expert_bank_segments","expert_bank_accounted_limit_bytes","static_cache_budget_bytes","device_reserved_limit_bytes","explicit_static_cache","count1_bank_profile"):
                wrong=copy.deepcopy(value);wrong[field]=None
                with self.assertRaises(ValueError):LAB.validate_receipt(wrong,"intel",10,completion(10),banks=banks)
            for card,cap in (("nvidia",10),("intel",0)):
                with self.assertRaises(ValueError):LAB.validate_receipt(value,card,cap,completion(cap),banks=banks)
            env=LAB.worker_environment("/original/model","/public/receipt",banks=banks)
            self.assertEqual(env["KUNO_LAB_COUNT1_BANKS"],banks)
            self.assertEqual(env["KUNO_LAB_BLOCK_MOE"],"1" if banks in ("intel96","intel128") else "0")
            self.assertEqual(env.get("KUNO_LAB_STATIC_CACHE_BYTES"),"5368709120" if banks=="intel128" else "6442450944" if banks=="intel96" else None)

    def test_both_caps_and_cards_require_fixed_inventory_and_actual_cpu_work(self):
        for card in ("intel", "nvidia"):
            for cap in range(11):
                LAB.validate_receipt(receipt(card, cap), card, cap, completion(cap))

    def test_intermediate_caps_require_matching_actual_completed_work_within_their_own_bound(self):
        for cap in range(1, 10):
            value = receipt("intel", cap)
            value.update(device_dispatch="inline", observed_inline_device_dispatch=True)
            output = completion(cap) + "\ngpu_device_dispatch mode=inline blocking=true"
            LAB.validate_receipt(value, "intel", cap, output, dispatch="inline")
            for field, limit in (("completed_cpu_misses", 7 * 48 * cap),
                                 ("fresh_completed_cpu_misses", 48 * cap)):
                for bad in (0, limit + 1, True, None):
                    wrong = copy.deepcopy(value); wrong[field] = bad
                    with self.assertRaises(ValueError):
                        LAB.validate_receipt(wrong, "intel", cap, output, dispatch="inline")
            with self.assertRaises(ValueError):
                LAB.validate_receipt(value, "intel", (cap % 10) + 1, output, dispatch="inline")

    def test_overlap_policy_pins_both_observations_and_cannot_enable_scoped_workers(self):
        for overlap in (False, True):
            value = receipt("intel", 10)
            value.update(count1_hybrid_cpu_workers="pooled", observed_count1_hybrid_cpu_workers="pooled",
                         count1_hybrid_cpu_gpu_overlap=overlap, observed_count1_hybrid_cpu_gpu_overlap=overlap, observed_shared_overlap_calls=336 if overlap else 0)
            LAB.validate_receipt(value, "intel", 10, completion(10), "pooled", overlap)
            for bad in (None, True, "336", 335, 337, 0 if overlap else 336):
                wrong = copy.deepcopy(value)
                wrong["observed_shared_overlap_calls"] = bad
                with self.assertRaises(ValueError):
                    LAB.validate_receipt(wrong, "intel", 10, completion(10), "pooled", overlap)
            for field in ("count1_hybrid_cpu_gpu_overlap", "observed_count1_hybrid_cpu_gpu_overlap"):
                for bad in (None, 0, "false", not overlap):
                    wrong = copy.deepcopy(value)
                    wrong[field] = bad
                    with self.assertRaises(ValueError):
                        LAB.validate_receipt(wrong, "intel", 10, completion(10), "pooled", overlap)
            with mock.patch.dict(os.environ, {"KUNO_LAB_CPU_GPU_OVERLAP": "auto"}):
                env = LAB.worker_environment("/original/model", "/public/receipt", "pooled", overlap)
            self.assertEqual(env["KUNO_LAB_CPU_GPU_OVERLAP"], "1" if overlap else "0")
        for workers, overlap in (("scoped", True), ("pooled", 1), ("pooled", "1")):
            with self.assertRaises(ValueError):
                LAB.worker_environment("/original/model", "/public/receipt", workers, overlap)

    def test_cpu_worker_policy_cannot_be_forged_or_inherited(self):
        for selected in ("scoped", "pooled"):
            value = receipt("intel", 10)
            value.update(count1_hybrid_cpu_workers=selected, observed_count1_hybrid_cpu_workers=selected)
            LAB.validate_receipt(value, "intel", 10, completion(10), selected)
            for field in ("count1_hybrid_cpu_workers", "observed_count1_hybrid_cpu_workers"):
                for bad in (None, True, "auto", "pooled" if selected == "scoped" else "scoped"):
                    wrong = copy.deepcopy(value)
                    wrong[field] = bad
                    with self.assertRaises(ValueError):
                        LAB.validate_receipt(wrong, "intel", 10, completion(10), selected)
            with mock.patch.dict(os.environ, {"KUNO_LAB_CPU_WORKERS": "auto"}):
                env = LAB.worker_environment("/original/model", "/public/receipt", selected)
            self.assertEqual(env["KUNO_LAB_CPU_WORKERS"], selected)
        with self.assertRaises(ValueError):
            LAB.worker_environment("/original/model", "/public/receipt", "auto")

    def test_receipt_cannot_select_cap_or_claim_overlap(self):
        for cap in (0, 10):
            for field in ("count1_hybrid_cpu_miss_cap", "observed_count1_hybrid_cpu_miss_cap"):
                for wrong in (None, True, "10", 10.0, 1, 10 - cap):
                    value = receipt("intel", cap)
                    value[field] = wrong
                    with self.assertRaises(ValueError):
                        LAB.validate_receipt(value, "intel", cap, completion(cap))
            for bad in (None, 0, True, "false"):
                value = receipt("intel", cap)
                value["count1_hybrid_cpu_gpu_overlap"] = bad
                with self.assertRaises(ValueError):
                    LAB.validate_receipt(value, "intel", cap, completion(cap))
        for cap in (False, "0", 0.0, None, -1, 11):
            with self.assertRaises(ValueError):
                LAB.validate_receipt(receipt("intel", 0), "intel", cap, completion(0))

    def test_completed_counts_reject_attempt_only_missing_and_impossible_successes(self):
        for cap in (0, 10):
            for field, maximum in (("completed_cpu_misses", 3360), ("fresh_completed_cpu_misses", 480)):
                for bad in (None, True, "1", 1.0, -1, maximum + 1, 0 if cap else 1):
                    value = receipt("intel", cap)
                    value[field] = bad
                    with self.assertRaises(ValueError):
                        LAB.validate_receipt(value, "intel", cap, completion(cap))

    def test_every_actual_owner_and_output_and_lifecycle_proof_is_mandatory(self):
        original = receipt("intel", 10)
        for at in range(4):
            for field in ("main_buffers", "main_bytes", "source_buffers", "source_bytes"):
                for bad in (None, True, "0", -1, original["owned_zero_audits"][at][field] + 1):
                    value = copy.deepcopy(original)
                    value["owned_zero_audits"][at][field] = bad
                    with self.assertRaises(ValueError):
                        LAB.validate_receipt(value, "intel", 10, completion(10))
        for field, bad in (("fixture", "resident-chat"), ("public_generated_ids", [9419]),
                           ("native_decode_steps", 8), ("scalar_decode_steps", 1), ("temperature", False),
                           ("native_projection_tile", 1), ("native_decode_projection_tile", 8),
                           ("observed_native_decode_kernels", "chunk"), ("batch_experts", True),
                           ("resident_private_owned_bytes", 818249728), ("owned_zero_audits", [])):
            value = copy.deepcopy(original)
            value[field] = bad
            with self.assertRaises(ValueError):
                LAB.validate_receipt(value, "intel", 10, completion(10))
        for line in completion(10).splitlines():
            with self.assertRaises(ValueError):
                LAB.validate_receipt(original, "intel", 10, completion(10).replace(line, ""))

    def test_ordinary_validator_still_rejects_hybrid_inventory(self):
        value = receipt("intel", 0)
        value["fixture"] = "resident-chat"
        with self.assertRaises(ValueError):
            LAB.BASE.validate_receipt(value, "intel", "resident-chat", 8, completion(0),
                native_decode=True, native_decode_tile=1, native_decode_kernels="single-q4-all")

    def test_environment_does_not_inherit_ambient_policy(self):
        with mock.patch.dict(os.environ, {"KUNO_LAB_SMALL_CPU": "1", "KUNO_LAB_EXPERT_SLOTS": "96"}):
            env = LAB.worker_environment("/original/model", "/public/receipt")
        self.assertEqual(env["KUNO_LAB_SMALL_CPU"], "0")
        self.assertNotIn("KUNO_LAB_EXPERT_SLOTS", env)
        self.assertEqual(env["KUNO_LAB_NATIVE_BATCH_COUNT"], "8")
        self.assertEqual(env["KUNO_LAB_NATIVE_DECODE_KERNELS"], "single-q4-all")
        self.assertEqual(env["KUNO_LAB_CANCEL_CHECK"], "1")

    def test_live_runner_pins_cap_validates_before_recording_and_rejects_incomplete_proofs(self):
        for valid in (False, True):
            with tempfile.TemporaryDirectory() as directory:
                target = pathlib.Path(directory) / "receipt.json"
                build = pathlib.Path(directory) / "build.jsonl"
                build.write_text(json.dumps({"reason":"compiler-artifact","target":{"name":"kuno-text-lab"},"executable":"/frozen","features":[]})+"\n")
                command_seen = []
                def launch(command, **options):
                    command_seen.extend(command)
                    value = receipt("intel", 10)
                    if not valid:
                        value["completed_cpu_misses"] = 0
                    target.write_text(json.dumps(value))
                    return SimpleNamespace(stdout=io.StringIO(completion(10)), wait=lambda: 0)
                args = ["lab", "--cargo-artifacts", str(build), "--model", "/original/model",
                        "--card", "intel", "--cpu-miss-cap", "10", "--run-id", "test",
                        "--receipt", str(target)]
                with mock.patch.object(LAB.BASE, "cargo_binary", return_value=(pathlib.Path("/frozen"), "binary", "build")), \
                     mock.patch.object(LAB.subprocess, "Popen", side_effect=launch), \
                     mock.patch.object(LAB.sys, "argv", args), mock.patch("sys.stdout", new_callable=io.StringIO):
                    if valid:
                        LAB.main()
                    else:
                        with self.assertRaises(ValueError):
                            LAB.main()
                self.assertEqual(command_seen[-3:], ["intel", LAB.MODE, "10"])
                self.assertIn("--property=MemorySwapMax=0", command_seen)
                self.assertIn("--property=LimitCORE=0", command_seen)
                record = target.with_suffix(".json.run.json")
                self.assertEqual(record.exists(), valid)
                if valid:
                    data = json.loads(record.read_text())
                    self.assertFalse(data["matched_performance_measurement"])
                    self.assertFalse(data["cpu_gpu_overlap"])
                    self.assertEqual(data["cpu_miss_cap"], 10)


if __name__ == "__main__":
    unittest.main()
