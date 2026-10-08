"""Protected fixed-public whole-model CPU-miss acceptance, not production inference.

Requires externally coordinated exclusive GPU/admission access. Does not obtain
sudo, stop services, reset hardware or accept arbitrary prompts. Device output
and timings establish nothing until every output/lifecycle/owning audit passes.
"""
import argparse
import importlib.util
import json
import pathlib
import re
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("native_max8_lab", HERE / "run-native-max8-lab.py")
BASE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BASE)
MODE = "resident-count1-hybrid-chat"
GUARD = "verified_lab_limits swap_max=0 core_soft=0 core_hard=0 memory_max=8589934592"


def validate_cap(cap):
    if type(cap) is not int or not 0 <= cap <= 10:
        raise ValueError("CPU miss cap must be independently selected as integer 0 through 10")


def completion(cap, deferred=False):
    validate_cap(cap)
    return (
        f"public_count1_hybrid_fresh_continuation cap={cap} matches=true",
        f"public_count1_hybrid_mid_token_cancel cap={cap} layer=0 erased=true",
        f"public_count1_hybrid_mid_token_cancel cap={cap} layer=3 erased=true",
    ) + ((f"public_count1_hybrid_mid_token_cancel cap={cap} layer=47 erased=true",) if deferred else ())


def validate_request_io_reuse(receipt, selected, output):
    if not selected and "request_io_reuse" not in receipt:
        return
    calls = receipt.get("observed_request_io_checkpoints")
    close_ms = receipt.get("request_io_close_ms")
    reuses = receipt.get("observed_request_io_reuses")
    if (type(selected) is not bool or receipt.get("request_io_reuse") is not selected
        or receipt.get("request_io_closed_and_audited") is not selected
        or type(calls) is not int or not 0 <= calls <= 1000000 or (calls > 0) is not selected
        or type(reuses) is not int or not 0 <= reuses <= 10000000 or (reuses > 0) is not selected
        or type(close_ms) is not int or not 0 <= close_ms <= 900000
        or not selected and close_ms != 0):
        raise ValueError("Request I/O reuse requires actual checkpoints and full closing audit")
    if selected and "public_request_io_reuse closed=true full_audit=true" not in output.splitlines():
        raise ValueError("Completed request I/O closing audit missing")
    if selected or "warm_info_cache_stats" in receipt:
        stats = receipt.get("warm_info_cache_stats")
        if (type(stats) is not list or len(stats) != 3
            or any(type(n) is not int or not 0 <= n <= 10000000 for n in stats)
            or selected and not all(n > 0 for n in stats)
            or not selected and any(stats)):
            raise ValueError("Immutable INFO reuse requires actual bounded hits, misses and page pins")
    # Older frozen control artifacts predate this optional measurement.
    if "warm_upload_transfer_bytes" in receipt:
        stats = receipt["warm_upload_transfer_bytes"]
        if (type(stats) is not list or len(stats) != 3
            or any(type(n) is not int or not 0 <= n < 2**64 for n in stats)
            or stats[0] != stats[1] + stats[2]
            or selected and stats[0] == 0
            or not selected and any(stats)):
            raise ValueError("Scoped upload bytes must conserve copied and GPU-cleared spans")


def validate_receipt(receipt, card, cap, output, workers="scoped", overlap=False, mapped=False, profile=False, fast=False, long=False, cpu_q8=False, large_cache=False, compact_info=False, batch_tasks=False, fast_fault=False, deferred=False, gpu_router=False, metadata=False, banks="standard", shared=False, shared_gdn=False, info_batch_upload=False, dispatch="threaded", cpu_fma=False, reclaim_gpu_sources=False, native_scratch_clear=False, cpu_q8_residual=False, cpu_mapped_prefetch=False, pageout_gpu_sources=False, cpu_threads=16, cpu_q8_two_pass=False, cpu_buffered_weights=False, compute_scratch_clear=False, grouped_status=False, readback_reuse=False, gpu_upload_clear=False, inline_readback=False, cpu_work_stealing=False, request_io_reuse=False):
    validate_cap(cap)
    validate_request_io_reuse(receipt, request_io_reuse, output)
    if request_io_reuse and cap != 10:
        raise ValueError("Request I/O reuse requires cap10")
    if cpu_work_stealing or "cpu_work_stealing" in receipt:
        if type(cpu_work_stealing) is not bool or receipt.get("cpu_work_stealing") is not cpu_work_stealing or cpu_work_stealing and workers != "pooled":
            raise ValueError("CPU row scheduling policy mismatch")
        if cpu_work_stealing and "cpu_row_dispatch mode=work-stealing joined=true original_arithmetic=true" not in output.splitlines():
            raise ValueError("CPU row scheduling marker missing")
    if inline_readback or "inline_readback" in receipt:
        calls=receipt.get("observed_inline_readback_batches")
        if type(inline_readback) is not bool or receipt.get("inline_readback") is not inline_readback or type(calls) is not int or not 0<=calls<=1000000 or (calls>0) is not inline_readback:
            raise ValueError("Inline readback must match bounded completed evidence")
        if inline_readback and "private_inline_readback enabled=true checked_callbacks=true held_leases=true" not in output.splitlines():
            raise ValueError("Actual inline readback policy marker missing")
    if gpu_upload_clear or "gpu_upload_clear" in receipt:
        calls=receipt.get("observed_gpu_upload_clears")
        if type(gpu_upload_clear) is not bool or receipt.get("gpu_upload_clear") is not gpu_upload_clear or type(calls) is not int or not 0<=calls<=1000000 or (calls>0) is not gpu_upload_clear:
            raise ValueError("GPU upload clear must match bounded completed evidence")
        if gpu_upload_clear and "private_primary_upload_clear enabled=true two_gpu_passes=true direct_full_cpu_inspection=true" not in output.splitlines():
            raise ValueError("Actual primary upload clear policy marker missing")
    if readback_reuse or "readback_reuse" in receipt:
        calls=receipt.get("observed_clean_readback_reuses")
        if type(readback_reuse) is not bool or receipt.get("readback_reuse") is not readback_reuse or type(calls) is not int or not 0<=calls<=1000000 or (calls>0) is not readback_reuse:
            raise ValueError("Clean readback reuse must match bounded completed evidence")
        if readback_reuse and "private_clean_readback_reuse enabled=true max_buffers=64 max_bytes=8388608" not in output.splitlines():
            raise ValueError("Actual bounded readback policy marker missing")
    if type(cpu_threads) is not int or cpu_threads not in (4,8,16):
        raise ValueError("Explicit public CPU threads must be4,8,16")
    if cpu_threads!=16 or cpu_q8_two_pass or "cpu_threads" in receipt:
        if type(receipt.get("cpu_threads")) is not int or type(receipt.get("observed_cpu_threads")) is not int or receipt.get("cpu_threads")!=cpu_threads or receipt.get("observed_cpu_threads")!=cpu_threads:
            raise ValueError("Actual constructed CPU threads mismatch")
        if receipt.get("cpu_q8_two_pass") is not cpu_q8_two_pass or (cpu_q8_two_pass and (not cpu_q8 or not cpu_q8_residual or workers!="pooled")):
            raise ValueError("Compiled two-pass Q8 must match explicit residual choice")
        passes=receipt.get("observed_cpu_q8_residual_passes")
        if type(passes) is not int or passes!=(2 if cpu_q8_two_pass else 3 if cpu_q8_residual else 0):
            raise ValueError("Completed residual activation pass count mismatch")
    if profile or "kernel_profile_scope" in receipt:
        if receipt.get("kernel_profile_scope") != ("first-following-decode" if profile else "disabled") or type(receipt.get("profiled_native_decode_steps")) is not int or receipt.get("profiled_native_decode_steps") != (1 if profile else 0):
            raise ValueError("GPU timing must cover only one independently selected native decode step")
    if cpu_buffered_weights or "cpu_buffered_weights" in receipt:
        calls=receipt.get("warm_cpu_buffered_reads");misses=receipt.get("completed_cpu_misses")
        if type(cpu_buffered_weights) is not bool or (cpu_buffered_weights and (not mapped or workers!="pooled" or cpu_mapped_prefetch)) or receipt.get("cpu_buffered_weights") is not cpu_buffered_weights or receipt.get("observed_cpu_buffered_weights") is not cpu_buffered_weights or type(calls) is not int or type(misses) is not int or calls != (3*misses if cpu_buffered_weights else 0):
            raise ValueError("CPU buffered source requires actual original selected-span reads")
    if cpu_mapped_prefetch or "cpu_mapped_prefetch" in receipt:
        calls=receipt.get("warm_cpu_prefetch_calls");misses=receipt.get("completed_cpu_misses")
        if (cpu_mapped_prefetch and (not mapped or workers!="pooled")) or receipt.get("cpu_mapped_prefetch") is not cpu_mapped_prefetch or receipt.get("observed_cpu_mapped_prefetch") is not cpu_mapped_prefetch or type(calls) is not int or type(misses) is not int or calls != (3*misses if cpu_mapped_prefetch else 0):
            raise ValueError("CPU source prefetch requires actual mapped pooled selected-source counts")
    if cpu_q8_residual or "cpu_q8_residual" in receipt:
        if (cpu_q8_residual and not cpu_q8) or receipt.get("cpu_q8_residual") is not cpu_q8_residual or receipt.get("observed_cpu_q8_residual") is not cpu_q8_residual:
            raise ValueError("Residual CPU Q8 must match actual selected arithmetic")
    if grouped_status or "grouped_status" in receipt:
        calls=receipt.get("observed_grouped_status_calls")
        if receipt.get("grouped_status") is not grouped_status or type(calls) is not int or not 0<=calls<=1000000 or (calls>0) is not grouped_status:
            raise ValueError("Grouped status must match completed bounded evidence")
        if grouped_status and "public_grouped_status_fold groups=3,5,8 counts=1,3,8 active_and_sticky=true inactive_ignored=true" not in output.splitlines():
            raise ValueError("Grouped status active/sticky probes missing")
    if compute_scratch_clear or "compute_scratch_clear" in receipt:
        calls=receipt.get("observed_compute_scratch_clear_calls")
        if receipt.get("compute_scratch_clear") is not compute_scratch_clear or type(calls) is not int or not 0<=calls<=1000000 or (calls>0) is not compute_scratch_clear:
            raise ValueError("Compute full scratch clear must match completed bounded evidence")
        if compute_scratch_clear and "public_compute_scratch_clear_full_owners=2,3,4 unselected_preserved=true padding=true" not in output.splitlines():
            raise ValueError("Compute scratch full-owner probes missing")
    if native_scratch_clear or "native_scratch_clear" in receipt:
        calls=receipt.get("observed_native_scratch_clear_calls")
        if receipt.get("native_scratch_clear") is not native_scratch_clear or type(calls) is not int or not 0<=calls<=1000000 or (calls>0) is not native_scratch_clear:
            raise ValueError("Native full scratch clear must match completed bounded evidence")
    if "warm_cpu_operator_host_phase_ns" in receipt:
        phases=receipt["warm_cpu_operator_host_phase_ns"]
        if type(phases) is not list or len(phases)!=4 or any(type(n) is not int or not 0<=n<=900_000_000_000 for n in phases) or sum(phases[:3])>phases[3]:
            raise ValueError("Successful CPU phase counters require bounded numeric host intervals")
    if pageout_gpu_sources or "pageout_gpu_sources" in receipt:
        if receipt.get("pageout_gpu_sources") is not pageout_gpu_sources or (pageout_gpu_sources and (not reclaim_gpu_sources or receipt.get("gpu_source_mapping_advice") != "discard-clean-file-ptes")):
            raise ValueError("Mapped GPU source pageout requires explicit actual source reclaim")
    if reclaim_gpu_sources or "reclaim_gpu_sources" in receipt:
        if receipt.get("reclaim_gpu_sources") is not reclaim_gpu_sources or (reclaim_gpu_sources and (cap!=10 or not mapped)):
            raise ValueError("GPU source cache advice must match mapped static cap10 selection")
        advised=receipt.get("gpu_source_advised_bytes")
        elapsed=receipt.get("gpu_source_advice_ms")
        if type(advised) is not int or not 0 <= advised <= 32*1024**3 or (reclaim_gpu_sources and advised==0) or (not reclaim_gpu_sources and advised!=0) or type(elapsed) is not int or elapsed<0:
            raise ValueError("Missing bounded actual GPU source advice evidence")
    if cpu_fma or "cpu_fma_arithmetic" in receipt:
        if receipt.get("cpu_fma_arithmetic") is not cpu_fma or (cpu_fma and (workers != "pooled" or cpu_q8)):
            raise ValueError("Explicit CPU FMA arithmetic does not match selection")
    if dispatch not in ("threaded","inline"):
        raise ValueError("Inline dispatch requires compiled protected count1 mode")
    if dispatch=="inline" or "device_dispatch" in receipt:
        if receipt.get("device_dispatch")!=dispatch or receipt.get("observed_inline_device_dispatch") is not (dispatch=="inline"):
            raise ValueError("Compiled device dispatch does not match selection")
        marker=f"gpu_device_dispatch mode={dispatch} blocking={str(dispatch=='inline').lower()}"
        if marker not in output.splitlines():
            raise ValueError("Actual runtime device dispatch marker missing")
    if info_batch_upload and (not compact_info):
        raise ValueError("Batched INFO requires compact count1 mode")
    if info_batch_upload or "info_batch_upload" in receipt:
        if receipt.get("info_batch_upload") is not info_batch_upload:
            raise ValueError("Batched INFO selection mismatch")
    if info_batch_upload and "private_info_batch_upload enabled=true" not in output.splitlines():
        raise ValueError("Batched INFO runtime initialization missing")
    if shared_gdn and not shared:
        raise ValueError("Shared GDN requires shared Next kernels")
    if shared_gdn or "next_shared_gdn" in receipt:
        if receipt.get("next_shared_gdn") is not shared_gdn or type(receipt.get("observed_shared_gdn_calls")) is not int or receipt["observed_shared_gdn_calls"] != ((127 if long else 7)*36 if shared_gdn else 0):
            raise ValueError("Actual shared GDN calls must cover every completed recurrent layer")
    if shared_gdn and "next_shared_gdn cases=3 numeric_and_owning_checks=true" not in output.splitlines():
        raise ValueError("Independent original GDN numeric/owning probe missing")
    if shared and (not fast or not gpu_router):
        raise ValueError("Shared kernels require protected count1 GPU/router mode")
    if shared or "next_shared_kernels" in receipt:
        if receipt.get("next_shared_kernels") is not shared:
            raise ValueError("Shared-kernel selection mismatch")
        calls=receipt.get("observed_shared_dense_calls")
        if type(calls) is not int or not 0 <= calls <= 100000 or (calls>0) is not shared or calls!=receipt.get("observed_fast_dense_calls") and shared:
            raise ValueError("Complete shared dense dispatch evidence missing")
    if shared and any(marker not in output.splitlines() for marker in ("next_shared_original_rows cases=6 numeric_and_owning_checks=true", "next_shared_gpu_router cases=8 numeric_and_owning_checks=true", "next_shared_gr_rms counts=1,3,8 cases=6 numeric_finite_status_and_owning_checks=true")):
        raise ValueError("Shared original row/router probes missing")
    if shared and (type(receipt.get("observed_shared_gr_calls")) is not int or not 1 <= receipt["observed_shared_gr_calls"] <= 1_000_000):
        raise ValueError("Actual shared GR dispatch evidence missing")
    if banks not in ("standard","intel72","intel96","intel128","intel144","intel160") or banks!="standard" and (card!="intel"):
        raise ValueError("Expanded banks require an explicit protected Intel count1 profile")
    if (banks!="standard" or "count1_bank_profile" in receipt) and receipt.get("count1_bank_profile")!=banks:
        raise ValueError("Completed bank profile must match independent selection")
    for field, selected in (("mapped_weights", mapped), ("observed_mapped_weights", mapped), ("kernel_profile", profile), ("fast_gpu", fast), ("long_fixture", long), ("cpu_q8_activations", cpu_q8), ("observed_cpu_q8_activations", cpu_q8), ("large_weight_cache", large_cache), ("compact_info", compact_info), ("batch_tasks", batch_tasks), ("fast_gpu_fault", fast_fault), ("defer_router_io",deferred), ("observed_defer_router_io",deferred), ("gpu_router",gpu_router),("retained_metadata",metadata)):
        if type(selected) is not bool or (selected or field in receipt) and receipt.get(field) is not selected:
            raise ValueError("Explicit optimization selection and completed receipt must match")
    metadata_calls=receipt.get("observed_retained_metadata_calls",0)
    if type(metadata_calls) is not int or not (0<=metadata_calls<=1000000) or (metadata_calls>0) is not metadata:
        raise ValueError("Completed metadata-only calls must match explicit selection")
    shared_calls=receipt.get("observed_shared_overlap_calls",0)
    if type(shared_calls) is not int or shared_calls!=((127 if long else 7)*48 if overlap else 0):
        raise ValueError("Actual prepared shared branches must match explicit overlap selection")
    router_calls=receipt.get("observed_gpu_router_calls",0)
    if type(router_calls) is not int or router_calls!=((127 if long else 7)*48 if gpu_router else 0):
        raise ValueError("Actual GPU routing must cover every completed native layer")
    if gpu_router and "rust_gpu_router cases=6 ranks=10 numeric_and_owning_checks=true" not in output.splitlines():
        raise ValueError("Independent fixed GPU routing checks missing")
    if deferred and (not compact_info):
        raise ValueError("Deferred router IO requires compact protected count1 hybrid mode")
    reads=receipt.get("observed_deferred_router_reads",0)
    if type(reads) is not int or reads!=((127 if long else 7)*48 if deferred else 0):
        raise ValueError("Actual deferred router bridges must cover every completed native layer")
    if deferred and receipt.get("io_token_checkpoint") is not True:
        raise ValueError("Completed token IO cadence missing")
    for field, expected, selected in (("info_slab_bytes",4096 if compact_info else 0,compact_info),("runtime_tasks_max",256 if batch_tasks else 32,batch_tasks)):
        if (selected or field in receipt) and (type(receipt.get(field)) is not int or receipt[field]!=expected):
            raise ValueError("Completed bounded runtime configuration must match explicit selection")
    calls = receipt.get("observed_fast_dense_calls", 0)
    if type(calls) is not int or not 0 <= calls <= 100000 or (calls > 0) is not fast:
        raise ValueError("Actual count1 fast dense dispatch evidence must match explicit selection")
    fused_calls=receipt.get("observed_fast_dense_fault_calls",0)
    if type(fused_calls) is not int or fused_calls != (calls if fast_fault else 0) or fast_fault and not fast:
        raise ValueError("Actual fused dispatches must cover the requested fast dense path")
    if fast_fault and any(f"{'next_shared' if shared else 'rust_fast'}_packed_fused_fault dtype={dtype} cases={5 if shared else 4} passed=true" not in output.splitlines() for dtype in (0,1,30,12,6,14,8)):
        raise ValueError("All seven original formats require fixed fused numeric/tail/poison checks")
    if workers not in ("scoped", "pooled"):
        raise ValueError("Unknown CPU worker mode")
    if cpu_q8 and workers != "pooled":
        raise ValueError("Q8 CPU activation arithmetic requires pooled workers")
    for field in ("count1_hybrid_cpu_workers", "observed_count1_hybrid_cpu_workers"):
        if receipt.get(field) != workers:
            raise ValueError("Requested and completed CPU workers must match explicit selection")
    if reclaim_gpu_sources and not mapped:
        raise ValueError("GPU source cache advice requires mapped original weights")
    if large_cache and not mapped:
        raise ValueError("Larger public cache requires mapped weights")
    guard = GUARD.replace("8589934592", "51539607552") if large_cache else GUARD
    if guard not in output.splitlines():
        raise ValueError("Protected execution limits missing")
    for field in ("count1_hybrid_cpu_miss_cap", "observed_count1_hybrid_cpu_miss_cap"):
        if type(receipt.get(field)) is not int or receipt[field] != cap:
            raise ValueError("Requested and completed hybrid cap mismatch")
    if type(overlap) is not bool or (overlap and workers != "pooled"):
        raise ValueError("Overlap requires explicit pooled workers")
    for field in ("count1_hybrid_cpu_gpu_overlap", "observed_count1_hybrid_cpu_gpu_overlap"):
        if receipt.get(field) is not overlap:
            raise ValueError("Requested and completed overlap policy mismatch")
    for field, limit in (("completed_cpu_misses", (127 if long else 7) * 48 * cap),
                         ("fresh_completed_cpu_misses", 48 * cap)):
        value = receipt.get(field)
        if type(value) is not int or not 0 <= value <= limit:
            raise ValueError("Completed CPU work outside accepted fixture bounds")
        if cap == 0 and value != 0 or cap > 0 and value == 0:
            raise ValueError("Actual completed CPU-miss evidence missing or incompatible")
    if any(type(receipt.get(field)) is not int for field in (
        "resident_private_owned_bytes", "prompt_tokens", "seed", "temperature", "native_public_chunk_count"
    )) or not isinstance(receipt.get("public_generated_ids"), list) or any(
        type(token) is not int for token in receipt["public_generated_ids"]
    ):
        raise ValueError("Hybrid fixture and inventory must use integer counts")
    audits = receipt.get("owned_zero_audits")
    if not isinstance(audits, list) or len(audits) != (5 if deferred else 4) or any(
        not isinstance(audit, dict) or any(type(audit.get(field)) is not int
            for field in ("main_buffers", "main_bytes", "source_buffers", "source_bytes"))
        for audit in audits
    ):
        raise ValueError("Four typed complete hybrid owner audits required")
    BASE._validate_model_receipt(
        receipt, card, "resident-rainwater" if long else "resident-chat", 8, output,
        owner_inventory=(1378, 818413568), reported_fixture="resident-count1-hybrid-rainwater" if long else MODE,
        native_decode=True, native_decode_tile=1, native_decode_kernels="single-q4-all",
        audit_count=5 if deferred else 4,
        deferred_io=deferred, bank_profile=banks,
    )
    if any(marker not in output.splitlines() for marker in completion(cap,deferred)):
        raise ValueError("Hybrid continuation and both cancellation proofs required")


def worker_environment(model, receipt, workers="scoped", overlap=False, mapped=False, profile=False, fast=False, long=False, cpu_q8=False, large_cache=False, compact_info=False, batch_tasks=False, fast_fault=False, deferred=False, gpu_router=False, metadata=False, banks="standard", shared=False, shared_gdn=False, info_batch_upload=False, dispatch="threaded", reclaim_gpu_sources=False, native_scratch_clear=False, cpu_q8_residual=False, cpu_mapped_prefetch=False, pageout_gpu_sources=False, cpu_threads=16, cpu_q8_two_pass=False, compute_scratch_clear=False, grouped_status=False, readback_reuse=False, gpu_upload_clear=False, inline_readback=False, cpu_work_stealing=False, request_io_reuse=False):
    if type(cpu_threads) is not int or cpu_threads not in (4,8,16):
        raise ValueError("Explicit public CPU threads must be4,8,16")
    if cpu_threads!=16 and workers!="pooled":
        raise ValueError("CPU thread tuning requires pooled workers")
    if cpu_q8_two_pass and (not cpu_q8 or not cpu_q8_residual or workers!="pooled"):
        raise ValueError("Two-pass Q8 requires pooled explicit residual activations")
    if pageout_gpu_sources and not reclaim_gpu_sources:
        raise ValueError("Mapped GPU source pageout requires explicit source reclaim")
    if cpu_mapped_prefetch and (not mapped or workers!="pooled"):
        raise ValueError("CPU source prefetch requires mapped pooled workers")
    if cpu_q8_residual and not cpu_q8:
        raise ValueError("Residual CPU Q8 requires explicit Q8 selection")
    if reclaim_gpu_sources and not mapped:
        raise ValueError("GPU source cache advice requires mapped original weights")
    if workers not in ("scoped", "pooled"):
        raise ValueError("Unknown CPU worker mode")
    if type(overlap) is not bool or (overlap and workers != "pooled"):
        raise ValueError("Overlap requires explicit pooled workers")
    if any(type(value) is not bool for value in (mapped,profile,fast,long,cpu_q8,large_cache,compact_info,batch_tasks,fast_fault,deferred,gpu_router,metadata,shared,shared_gdn,info_batch_upload,reclaim_gpu_sources)):
        raise ValueError("Explicit boolean optimization selections required")
    if cpu_q8 and workers != "pooled":
        raise ValueError("Q8 CPU activation arithmetic requires pooled workers")
    if large_cache and not mapped:
        raise ValueError("Larger public cache requires mapped weights")
    if fast_fault and not fast:
        raise ValueError("Fused finite checks require fast GPU projections")
    if deferred and not compact_info:
        raise ValueError("Deferred router IO requires compact INFO")
    if banks not in ("standard","intel72","intel96","intel128","intel144","intel160"):
        raise ValueError("Unknown fixed bank profile")
    if dispatch not in ("threaded","inline"):
        raise ValueError("Unknown GPU device dispatch")
    env = BASE.worker_environment(model, receipt, 8, True, 1, "single-q4-all")
    env["KUNO_LAB_DEVICE_DISPATCH"] = dispatch
    env["KUNO_LAB_CPU_THREADS"]=str(cpu_threads)
    env["KUNO_LAB_CPU_Q8_TWO_PASS"]="1" if cpu_q8_two_pass else "0"
    env["KUNO_LAB_CPU_WORKERS"] = workers
    env["KUNO_LAB_CPU_GPU_OVERLAP"] = "1" if overlap else "0"
    env["KUNO_LAB_MAPPED_WEIGHTS"] = "1" if mapped else "0"
    env["KUNO_LAB_KERNEL_PROFILE"] = "1" if profile else "0"
    env["KUNO_LAB_FAST_GPU"] = "1" if fast else "0"
    if shared and (not fast or not gpu_router):
        raise ValueError("Shared kernels require fast GPU/router mode")
    env["KUNO_LAB_NEXT_SHARED"] = "1" if shared else "0"
    if shared_gdn and not shared:
        raise ValueError("Shared GDN requires shared Next kernels")
    env["KUNO_LAB_NEXT_SHARED_GDN"] = "1" if shared_gdn else "0"
    env["KUNO_LAB_LONG_FIXTURE"] = "1" if long else "0"
    env["KUNO_LAB_PAGEOUT_GPU_SOURCES"]="1" if pageout_gpu_sources else "0"
    env["KUNO_LAB_CPU_MAPPED_PREFETCH"]="1" if cpu_mapped_prefetch else "0"
    env["KUNO_LAB_CPU_Q8_RESIDUAL"]="1" if cpu_q8_residual else "0"
    env["KUNO_LAB_CPU_Q8"] = "1" if cpu_q8 else "0"
    env["KUNO_LAB_LARGE_WEIGHT_CACHE"] = "1" if large_cache else "0"
    env["KUNO_LAB_COMPACT_INFO"] = "1" if compact_info else "0"
    if info_batch_upload and not compact_info:
        raise ValueError("Batched INFO requires compact slabs")
    if type(readback_reuse) is not bool or readback_reuse and not info_batch_upload:
        raise ValueError("Clean readback reuse requires explicit batched INFO")
    if type(gpu_upload_clear) is not bool or gpu_upload_clear and not info_batch_upload:
        raise ValueError("GPU upload clear requires explicit batched INFO")
    env["KUNO_LAB_GPU_UPLOAD_CLEAR"] = "1" if gpu_upload_clear else "0"
    if type(inline_readback) is not bool or inline_readback and (not info_batch_upload or dispatch != "inline"):
        raise ValueError("Inline readback requires explicit batched INFO and inline dispatch")
    if type(cpu_work_stealing) is not bool or cpu_work_stealing and workers != "pooled":
        raise ValueError("CPU work stealing requires pooled workers")
    if type(request_io_reuse) is not bool or request_io_reuse and (not mapped or not deferred or not info_batch_upload or not inline_readback or gpu_upload_clear or profile):
        raise ValueError("Request I/O reuse requires mapped deferred batched inline I/O without GPU clear or profiling")
    env["KUNO_LAB_REQUEST_IO_REUSE"] = "1" if request_io_reuse else "0"
    env["KUNO_LAB_CPU_WORK_STEALING"] = "1" if cpu_work_stealing else "0"
    env["KUNO_LAB_INLINE_READBACK"] = "1" if inline_readback else "0"
    env["KUNO_LAB_READBACK_REUSE"] = "1" if readback_reuse else "0"
    env["KUNO_LAB_INFO_BATCH_UPLOAD"] = "1" if info_batch_upload else "0"
    env["KUNO_LAB_INFO_SLAB_BYTES"] = "4096" if compact_info else "0"
    env["KUNO_LAB_BATCH_TASKS"] = "1" if batch_tasks else "0"
    env["KUNO_LAB_FAST_GPU_FAULT"] = "1" if fast_fault else "0"
    if type(compute_scratch_clear) is not bool or compute_scratch_clear and (native_scratch_clear or not metadata or not shared or not compact_info or not batch_tasks):
        raise ValueError("Compute scratch clear requires retained shared compact task batches without native clear")
    if type(grouped_status) is not bool or grouped_status and (not metadata or not shared or not compact_info or not batch_tasks):
        raise ValueError("Grouped status requires retained shared compact task batches")
    env["KUNO_LAB_GROUPED_STATUS"] = "1" if grouped_status else "0"
    env["KUNO_LAB_COMPUTE_SCRATCH_CLEAR"] = "1" if compute_scratch_clear else "0"
    env["KUNO_LAB_NATIVE_SCRATCH_CLEAR"] = "1" if native_scratch_clear else "0"
    env["KUNO_LAB_RECLAIM_GPU_SOURCES"] = "1" if reclaim_gpu_sources else "0"
    env["KUNO_LAB_DEFER_ROUTER_IO"] = "1" if deferred else "0"
    env["KUNO_LAB_GPU_ROUTER"] = "1" if gpu_router else "0"
    env["KUNO_LAB_RETAINED_METADATA"] = "1" if metadata else "0"
    env["KUNO_LAB_COUNT1_BANKS"] = banks
    if banks in ("intel96","intel128","intel144","intel160"):
        env["KUNO_LAB_STATIC_CACHE_BYTES"] = "3221225472" if banks=="intel160" else "5368709120" if banks in ("intel128","intel144") else "6442450944"
        env["KUNO_LAB_BLOCK_MOE"] = "1"
    return env


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cargo-artifacts", type=pathlib.Path, required=True)
    parser.add_argument("--model", type=pathlib.Path, required=True)
    parser.add_argument("--card", choices=("intel", "nvidia"), required=True)
    parser.add_argument("--cpu-miss-cap", type=int, choices=range(11), required=True)
    parser.add_argument("--cpu-workers", choices=("scoped", "pooled"), default="scoped")
    parser.add_argument("--cpu-gpu-overlap", action="store_true")
    parser.add_argument("--mapped-weights", action="store_true")
    parser.add_argument("--kernel-profile", action="store_true")
    parser.add_argument("--fast-gpu", action="store_true")
    parser.add_argument("--next-shared-kernels", action="store_true")
    parser.add_argument("--next-shared-gdn", action="store_true")
    parser.add_argument("--long-fixture", action="store_true")
    parser.add_argument("--cpu-q8", action="store_true")
    parser.add_argument("--cpu-fma", action="store_true")
    parser.add_argument("--cpu-buffered-weights", action="store_true")
    parser.add_argument("--cpu-threads",type=int,choices=(4,8,16),default=16)
    parser.add_argument("--cpu-q8-two-pass",action="store_true")
    parser.add_argument("--cpu-mapped-prefetch",action="store_true")
    parser.add_argument("--cpu-q8-residual",action="store_true",help="Three-pass residual Q8 activations; requires explicit --cpu-q8")
    parser.add_argument("--grouped-status",action="store_true",help="Combine ready status folds at their original scratch-reuse boundary")
    parser.add_argument("--compute-scratch-clear",action="store_true",help="Group consecutive full-owner shader clears without ending the compute pass")
    parser.add_argument("--native-scratch-clear",action="store_true",help="Use ordered full-owner native clears at original scratch boundaries")
    parser.add_argument("--reclaim-gpu-sources", action="store_true")
    parser.add_argument("--pageout-gpu-sources",action="store_true")
    parser.add_argument("--large-weight-cache", action="store_true")
    parser.add_argument("--compact-info", action="store_true")
    parser.add_argument("--info-batch-upload", action="store_true")
    parser.add_argument("--device-dispatch", choices=("threaded","inline"), default="threaded")
    parser.add_argument("--batch-tasks", action="store_true")
    parser.add_argument("--fast-gpu-fault", action="store_true")
    parser.add_argument("--defer-router-io", action="store_true")
    parser.add_argument("--gpu-router", action="store_true")
    parser.add_argument("--retained-metadata", action="store_true")
    parser.add_argument("--request-io-reuse", action="store_true")
    parser.add_argument("--cpu-work-stealing", action="store_true")
    parser.add_argument("--inline-readback", action="store_true")
    parser.add_argument("--gpu-upload-clear", action="store_true")
    parser.add_argument("--readback-reuse", action="store_true")
    parser.add_argument("--bank-profile", choices=("standard","intel72","intel96","intel128","intel144","intel160"), default="standard")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--receipt", type=pathlib.Path, required=True)
    args = parser.parse_args()
    if args.bank_profile!="standard" and (args.card!="intel"):
        parser.error("Expanded banks require --card intel and a bounded --cpu-miss-cap")
    if args.cpu_work_stealing and args.cpu_workers != "pooled":
        parser.error("CPU work stealing requires pooled workers")
    if args.request_io_reuse and (args.cpu_miss_cap != 10 or not args.mapped_weights or not args.defer_router_io or not args.info_batch_upload or not args.inline_readback or args.gpu_upload_clear or args.kernel_profile):
        parser.error("Request I/O reuse requires mapped cap10 with deferred batched inline I/O without GPU clear or profiling")
    if args.cpu_gpu_overlap and args.cpu_workers != "pooled":
        parser.error("CPU/GPU overlap requires --cpu-workers pooled")
    if args.cpu_q8 and args.cpu_workers != "pooled":
        parser.error("Q8 CPU activation arithmetic requires --cpu-workers pooled")
    if args.large_weight_cache and not args.mapped_weights:
        parser.error("Larger public cache requires --mapped-weights")
    if args.inline_readback and (not args.info_batch_upload or args.device_dispatch != "inline"):
        parser.error("Inline readback requires --info-batch-upload and inline dispatch")
    if args.gpu_upload_clear and not args.info_batch_upload:
        parser.error("GPU upload clear requires --info-batch-upload")
    if args.readback_reuse and not args.info_batch_upload:
        parser.error("Readback reuse requires --info-batch-upload")
    if args.info_batch_upload and (not args.compact_info):
        parser.error("Batched INFO requires --compact-info in protected count1 mode")
    if args.next_shared_gdn and not args.next_shared_kernels:
        parser.error("Shared GDN requires --next-shared-kernels")
    if args.next_shared_kernels and (not args.fast_gpu or not args.gpu_router):
        parser.error("Shared kernels require --fast-gpu --gpu-router")
    if args.fast_gpu_fault and not args.fast_gpu:
        parser.error("Fused finite checks require --fast-gpu")
    if args.defer_router_io and (not args.compact_info):
        parser.error("Deferred router IO requires --compact-info and a bounded --cpu-miss-cap")
    if not re.fullmatch(r"[a-z0-9-]{1,48}", args.run_id):
        parser.error("run-id must be 1..48 lowercase letters, digits or hyphens")
    if args.pageout_gpu_sources and not args.reclaim_gpu_sources:
        parser.error("Mapped GPU source pageout requires --reclaim-gpu-sources")
    if args.cpu_buffered_weights and (not args.mapped_weights or args.cpu_workers!="pooled" or args.cpu_mapped_prefetch):
        parser.error("Buffered CPU weights require mapped pooled workers without mapped read-ahead")
    if args.cpu_mapped_prefetch and (not args.mapped_weights or args.cpu_workers!="pooled"):
        parser.error("CPU source prefetch requires mapped pooled workers")
    if args.cpu_q8_two_pass and (not args.cpu_q8 or not args.cpu_q8_residual or args.cpu_workers!="pooled"):
        parser.error("Two-pass Q8 requires pooled residual Q8")
    if args.cpu_q8_residual and not args.cpu_q8:
        parser.error("Residual Q8 requires --cpu-q8")
    if args.cpu_fma and (args.cpu_workers != "pooled" or args.cpu_q8):
        parser.error("CPU FMA requires pooled original-float arithmetic")
    if args.reclaim_gpu_sources and (args.cpu_miss_cap!=10 or not args.mapped_weights):
        parser.error("GPU source cache advice requires mapped static cap10")
    target = args.receipt.resolve()
    metadata = target.with_suffix(target.suffix + ".run.json")
    if target.exists() or metadata.exists():
        parser.error("Refusing to replace an existing receipt or run record")
    frozen, identity, artifacts_identity = BASE.cargo_binary(args.cargo_artifacts)
    artifacts = [json.loads(line) for line in args.cargo_artifacts.read_text().splitlines()]
    compiled = next(r for r in artifacts if r.get("reason")=="compiler-artifact" and r.get("target",{}).get("name")=="kuno-text-lab" and r.get("executable"))
    if ("cpu-work-stealing" in compiled.get("features", [])) is not args.cpu_work_stealing:
        parser.error("CPU work stealing must match its compiled artifact")
    if ("cpu-q8-two-pass" in compiled.get("features",[])) is not args.cpu_q8_two_pass:
        parser.error("Two-pass Q8 must match independently selected compiled artifact")
    if ("cpu-buffered-weights" in compiled.get("features",[])) is not args.cpu_buffered_weights:
        parser.error("Buffered CPU weights must match independently selected compiled artifact")
    if ("cpu-fma" in compiled.get("features",[])) is not args.cpu_fma:
        parser.error("CPU FMA must match the independently selected compiled artifact")
    command = [
        "systemd-run", "--user", "--unit=kuno-user-" + args.run_id,
        "--collect", "--wait", "--pipe", "--property=MemorySwapMax=0",
        "--property=LimitCORE=0", "--property=MemoryMax=" + ("48G" if args.large_weight_cache else "8G"),
        "--property=RuntimeMaxSec=900",
        "--property=UnsetEnvironment=KUNO_LAB_BLOCK_KERNEL KUNO_LAB_EXPERT_SLOTS " +
        ("KUNO_LAB_STATIC_CACHE_BYTES " if args.bank_profile not in ("intel96","intel128","intel144","intel160") else "") + "KUNO_LAB_EXPERT_BATCH_BYTES KUNO_LAB_EXPERT_SCATTER",
    ]
    command.extend("--setenv=" + key + "=" + value
                   for key, value in worker_environment(args.model.resolve(), target, args.cpu_workers, args.cpu_gpu_overlap, args.mapped_weights, args.kernel_profile, args.fast_gpu, args.long_fixture, args.cpu_q8, args.large_weight_cache, args.compact_info, args.batch_tasks, args.fast_gpu_fault,args.defer_router_io,args.gpu_router,args.retained_metadata,args.bank_profile,args.next_shared_kernels,args.next_shared_gdn,args.info_batch_upload,args.device_dispatch,args.reclaim_gpu_sources,args.native_scratch_clear,args.cpu_q8_residual,args.cpu_mapped_prefetch,args.pageout_gpu_sources,args.cpu_threads,args.cpu_q8_two_pass,compute_scratch_clear=args.compute_scratch_clear,grouped_status=args.grouped_status,readback_reuse=args.readback_reuse,gpu_upload_clear=args.gpu_upload_clear,inline_readback=args.inline_readback,cpu_work_stealing=args.cpu_work_stealing,request_io_reuse=args.request_io_reuse).items())
    command.extend([sys.executable, str(HERE / "protected-lab-exec.py"), str(frozen),
                    args.card, MODE, str(args.cpu_miss_cap)])
    print("verified_cargo_executable_sha256=" + identity, flush=True)
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    lines = []
    for line in process.stdout:
        print(line, end="", flush=True)
        lines.append(line)
    if process.wait() != 0:
        raise SystemExit("Protected hybrid fixture failed; no acceptance recorded")
    if not target.is_file():
        raise SystemExit("Missing whole-model hybrid receipt")
    snapshot = target.read_bytes()
    receipt = json.loads(snapshot)
    validate_receipt(receipt, args.card, args.cpu_miss_cap, "".join(lines), args.cpu_workers, args.cpu_gpu_overlap, args.mapped_weights, args.kernel_profile, args.fast_gpu, args.long_fixture, args.cpu_q8, args.large_weight_cache, args.compact_info, args.batch_tasks, args.fast_gpu_fault,args.defer_router_io,args.gpu_router,args.retained_metadata,args.bank_profile,args.next_shared_kernels,args.next_shared_gdn,args.info_batch_upload,args.device_dispatch,args.cpu_fma,args.reclaim_gpu_sources,args.native_scratch_clear,args.cpu_q8_residual,args.cpu_mapped_prefetch,args.pageout_gpu_sources,args.cpu_threads,args.cpu_q8_two_pass,args.cpu_buffered_weights,compute_scratch_clear=args.compute_scratch_clear,grouped_status=args.grouped_status,readback_reuse=args.readback_reuse,gpu_upload_clear=args.gpu_upload_clear,inline_readback=args.inline_readback,cpu_work_stealing=args.cpu_work_stealing,request_io_reuse=args.request_io_reuse)
    record = {
        "scope": "fixed-public-count1-hybrid-output-and-lifecycle-only",
        "requested_cpu_fma_arithmetic": args.cpu_fma,
        "requested_cpu_buffered_weights":args.cpu_buffered_weights,
        "requested_gpu_source_cache_advice": args.reclaim_gpu_sources,
        "requested_native_scratch_clear":args.native_scratch_clear,
        "requested_cpu_q8_residual":args.cpu_q8_residual,
        "requested_cpu_mapped_prefetch":args.cpu_mapped_prefetch,
        "requested_pageout_gpu_sources":args.pageout_gpu_sources,
        "requested_cpu_threads":args.cpu_threads,
        "requested_cpu_q8_two_pass":args.cpu_q8_two_pass,
        "cargo_artifacts_sha256": artifacts_identity, "frozen_executable_sha256": identity,
        "card": args.card, "fixture": MODE, "cpu_miss_cap": args.cpu_miss_cap,
        "receipt_sha256": BASE.digest(snapshot), "protected_lifecycle_completion_checked": True,
        "matched_performance_measurement": False, "cpu_gpu_overlap": args.cpu_gpu_overlap, "cpu_workers": args.cpu_workers,
        "mapped_weights": args.mapped_weights, "kernel_profile": args.kernel_profile,
        "request_io_reuse": args.request_io_reuse,
        "fast_gpu": args.fast_gpu, "next_shared_kernels": args.next_shared_kernels, "next_shared_gdn": args.next_shared_gdn,
        "long_fixture": args.long_fixture,
        "cpu_q8_activations": args.cpu_q8,
        "large_weight_cache": args.large_weight_cache,
        "readback_reuse": args.readback_reuse,
        "gpu_upload_clear": args.gpu_upload_clear,
        "cpu_work_stealing": args.cpu_work_stealing,
        "inline_readback": args.inline_readback,
        "compact_info": args.compact_info, "info_batch_upload": args.info_batch_upload, "device_dispatch":args.device_dispatch,
        "batch_tasks": args.batch_tasks, "compute_scratch_clear":args.compute_scratch_clear,"grouped_status":args.grouped_status,
        "fast_gpu_fault": args.fast_gpu_fault,
        "gpu_router":args.gpu_router,"defer_router_io":args.defer_router_io,"retained_metadata":args.retained_metadata,"bank_profile":args.bank_profile,
    }
    with metadata.open("x") as output:
        json.dump(record, output, indent=2, sort_keys=True)
        output.write("\n")
    print("verified_native_count1_hybrid_receipt=true", flush=True)


if __name__ == "__main__":
    main()
