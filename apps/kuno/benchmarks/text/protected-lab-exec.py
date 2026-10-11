"""Verify actual no-swap/core and bounded host cgroup before lab execution."""
import os
import pathlib
import resource
import sys

def selected_target_policy(environment, mtp=False):
    target_only_scope(environment)
    intel_mtp_scope(environment)
    frozen_down_integer_scope(environment)
    iq4_register_scope(environment)
    target_submission_backlog_scope(environment)
    profile_bank_fill_scope(environment)
    grouped_status_scope(environment)
    grouped_ple_scope(environment)
    frozen_down_stride_scope(environment)
    selected = {key: value for key, value in environment.items() if key.startswith(("KUNO_LAB_", "KUNO_TEST_", "KUNO_PUBLIC_"))}
    if "KUNO_LAB_LANGUAGE_DECODE" in selected:
        if not mtp or selected.pop("KUNO_LAB_LANGUAGE_DECODE") != "target-only-public":
            raise ValueError("Target-only decoding requires the compiled public language scope")
    if "KUNO_LAB_MTP_DEVICE" in selected:
        if not mtp or selected.pop("KUNO_LAB_MTP_DEVICE") != "intel-public":
            raise ValueError("Intel draft placement requires the fixed public speculative scope")
        for key in ("KUNO_PUBLIC_MTP_INTEL_TOTAL_BYTES", "KUNO_PUBLIC_MTP_INTEL_USED_BYTES", "KUNO_PUBLIC_MTP_INTEL_DEVICE_ID"):
            selected.pop(key)
    if "KUNO_LAB_TARGET_STATUS_FOLDS" in selected:
        if not mtp or selected.pop("KUNO_LAB_TARGET_STATUS_FOLDS") != "grouped-public":
            raise ValueError("Grouped status folds require the canonical fixed public target")
    if "KUNO_LAB_FROZEN_BANK_FILL" in selected:
        if not mtp or selected.pop("KUNO_LAB_FROZEN_BANK_FILL") != "profile-ranked-public":
            raise ValueError("Profile filler requires the fixed Intel full-Q8 target scope")
    if "KUNO_LAB_TARGET_SUBMISSION_BACKLOG" in selected:
        if not mtp or selected.pop("KUNO_LAB_TARGET_SUBMISSION_BACKLOG") != "queued-2048-public":
            raise ValueError("Target backlog requires the fixed Intel full-Q8 target scope")
    if "KUNO_LAB_TARGET_IQ4_DECODER" in selected:
        if not mtp or selected.pop("KUNO_LAB_TARGET_IQ4_DECODER") != "packed-registers-public":
            raise ValueError("IQ4 registers require the fixed Intel full-Q8 target scope")
    if "KUNO_LAB_TARGET_PLE_READS" in selected:
        if not mtp or selected.pop("KUNO_LAB_TARGET_PLE_READS") != "grouped-public":
            raise ValueError("Grouped PLE reads require the canonical fixed public target")
    if "KUNO_LAB_FROZEN_DOWN_SCHEDULE" in selected:
        if not mtp or selected.pop("KUNO_LAB_FROZEN_DOWN_SCHEDULE") != "four-public":
            raise ValueError("DOWN scheduling requires the canonical original four-row target")
    if "KUNO_LAB_FROZEN_GU_SCHEDULE" in selected:
        frozen_gu_integer_scope(environment)
        if not mtp or selected.pop("KUNO_LAB_FROZEN_GU_SCHEDULE") != "four-public":
            raise ValueError("GU scheduling requires canonical fixed public integer scope")
    if "KUNO_LAB_FROZEN_DOWN_MATH" in selected:
        if not mtp or selected.pop("KUNO_LAB_FROZEN_DOWN_MATH") != "integer-public":
            raise ValueError("Integer DOWN requires fixed original speculative full-Q8 scope")
    if "KUNO_LAB_FROZEN_GU_MATH" in selected:
        frozen_gu_integer_scope(environment)
        if not mtp or selected.pop("KUNO_LAB_FROZEN_GU_MATH") != "integer-public":
            raise ValueError("Integer GU requires the canonical fixed public target")
    if "KUNO_LAB_TARGET_GDN" in selected:
        three_gdn_scope(environment)
        if not mtp or selected.pop("KUNO_LAB_TARGET_GDN") != "three-streaming-public":
            raise ValueError("Three-token GDN requires the canonical fixed public target")
    if "KUNO_LAB_FROZEN_TABLES" in selected:
        private_table_scope(environment)
        if not mtp or selected.pop("KUNO_LAB_FROZEN_TABLES") not in ("private-public", "workgroup-public"):
            raise ValueError("Public tables require the canonical frozen public target scope")
    if "KUNO_LAB_FROZEN_DOWN_SCALES" in selected:
        private_table_scope(environment)
        if not mtp or selected.pop("KUNO_LAB_FROZEN_DOWN_SCALES") != "shared-public":
            raise ValueError("Down scales require canonical frozen public target scope")
    if "KUNO_LAB_FROZEN_GU_FACTORS" in selected:
        private_table_scope(environment)
        if not mtp or selected.pop("KUNO_LAB_FROZEN_GU_FACTORS") != "shared-public":
            raise ValueError("Gate/up factors require canonical frozen public target scope")
    if "KUNO_LAB_FROZEN_DOWN_ROWS" in selected:
        private_table_scope(environment)
        if not mtp or selected.pop("KUNO_LAB_FROZEN_DOWN_ROWS") != "four-public":
            raise ValueError("Down rows require canonical frozen public target scope")
    if "KUNO_LAB_FROZEN_GU_ROWS" in selected:
        private_table_scope(environment)
        if not mtp or selected.pop("KUNO_LAB_FROZEN_GU_ROWS") != "two-public":
            raise ValueError("Gate/up rows require canonical frozen public target scope")
    if "KUNO_LAB_FROZEN_HOST_STAGE" in selected:
        private_table_scope(environment)
        if not mtp or selected.pop("KUNO_LAB_FROZEN_HOST_STAGE") != "packed-public":
            raise ValueError("Packed staging requires canonical frozen public GPU router scope")
    if "KUNO_LAB_FROZEN_MEMBERS" in selected:
        private_table_scope(environment)
        if not mtp or selected.pop("KUNO_LAB_FROZEN_MEMBERS") != "three-public":
            raise ValueError("Three-member arithmetic requires canonical frozen public GPU router scope")
    if "KUNO_LAB_FROZEN_PARTITION_EXIT" in selected:
        if (not mtp or selected.pop("KUNO_LAB_FROZEN_PARTITION_EXIT") != "1"
                or selected.get("KUNO_LAB_TARGET_SOURCES") != "frozen-host-public"):
            raise ValueError("Partition exit requires explicit frozen public target sources")
    if "KUNO_LAB_FROZEN_BANK_PLACEMENT" in selected:
        # The preserve-placement diagnostic keeps the warm replay's actual
        # routed VRAM residents; it never changes routing or arithmetic.
        if (not mtp or selected.pop("KUNO_LAB_FROZEN_BANK_PLACEMENT") != "preserve-public"
                or selected.get("KUNO_LAB_TARGET_SOURCES") != "frozen-host-public"
                or selected.get("KUNO_LAB_SPECULATIVE_PUBLIC_ONLY") != "1"
                or selected.get("KUNO_LAB_WARM_REPLAY") != "once"):
            raise ValueError("Bank placement requires the canonical warm frozen public scope")
    if "KUNO_LAB_TARGET_SOURCES" in selected:
        spec = selected.get("KUNO_LAB_SPECULATIVE_PUBLIC_ONLY") == "1"
        language = selected.get("KUNO_LAB_LANGUAGE_PUBLIC_ONLY") == "1"
        if (not mtp or selected.pop("KUNO_LAB_TARGET_SOURCES") != "frozen-host-public"
                or spec == language
                or spec and selected.get("KUNO_LAB_WARM_REPLAY") != "once"
                or selected.get("KUNO_LAB_TARGET_REUSE") != "request"
                or selected.get("KUNO_LAB_TARGET_FILL") != "direct-public"
                or selected.get("KUNO_LAB_TARGET_ROUTER") not in (None, "device-frozen-public")
                or any(key in selected for key in ("KUNO_LAB_TARGET_EXPERTS", "KUNO_LAB_TARGET_MOE"))):
            raise ValueError("Frozen host sources require the fixed original public target scope")
    if "KUNO_LAB_TARGET_QSA" in selected:
        if not mtp or selected.pop("KUNO_LAB_TARGET_QSA") != "chronological-public":
            raise ValueError("GPU QSA requires the explicit compiled public target selector")
    if "KUNO_LAB_MTP_VOCAB" in selected:
        english_vocab_scope(environment)
        if not mtp or selected.pop("KUNO_LAB_MTP_VOCAB") != "english-public":
            raise ValueError("English vocabulary requires fixed public Q8 draft scope")
    if "KUNO_LAB_MTP_Q8_HEAD" in selected:
        if (not mtp or selected.pop("KUNO_LAB_MTP_Q8_HEAD") not in ("broadcast-public", "shared-public")
                or "KUNO_LAB_MTP_PROFILE" in selected
                or (selected.get("KUNO_LAB_SPECULATIVE_PUBLIC_ONLY") == "1") == (selected.get("KUNO_LAB_LANGUAGE_PUBLIC_ONLY") == "1")):
            raise ValueError("Q8 head broadcast requires explicit public self-contained Q8 draft")
    if "KUNO_LAB_MTP_PROFILE" in selected:
        if not mtp or selected.pop("KUNO_LAB_MTP_PROFILE") != "strata-q2":
            raise ValueError("Alternate draft requires the explicit original Q2 profile")
    if "KUNO_LAB_TARGET_ROUTER" in selected:
        router = selected.pop("KUNO_LAB_TARGET_ROUTER")
        frozen = router == "device-frozen-public"
        spec = selected.get("KUNO_LAB_SPECULATIVE_PUBLIC_ONLY") == "1"
        language = selected.get("KUNO_LAB_LANGUAGE_PUBLIC_ONLY") == "1"
        if (not mtp or router not in ("device-warm-public", "device-frozen-public")
                or not frozen and selected.get("KUNO_LAB_WARM_REPLAY") != "once"
                or frozen and (environment.get("KUNO_LAB_TARGET_SOURCES") != "frozen-host-public"
                               or spec == language
                               or spec and selected.get("KUNO_LAB_WARM_REPLAY") != "once")
                or selected.get("KUNO_LAB_TARGET_REUSE") != "request"
                or "KUNO_LAB_TARGET_EXPERTS" in selected):
            raise ValueError("Device router requires the fixed original public request")
    if "KUNO_LAB_TARGET_REUSE" in selected:
        if (selected.pop("KUNO_LAB_TARGET_REUSE") != "request"
                or selected.get("KUNO_LAB_DEVICE_DISPATCH") != "inline"
                or selected.get("KUNO_LAB_TARGET_IO") != "chunk"):
            raise ValueError("Target request reuse requires direct dispatch and chunk I/O")
    if "KUNO_LAB_TARGET_MOE" in selected:
        if (selected.pop("KUNO_LAB_TARGET_MOE") != "live"
                or "KUNO_LAB_TARGET_EXPERTS" in selected):
            raise ValueError("Live reductions require original F32 target experts")
    if "KUNO_LAB_DEVICE_DISPATCH" in selected:
        if selected.pop("KUNO_LAB_DEVICE_DISPATCH") != "inline":
            raise ValueError("Target dispatch requires canonical inline")
    if "KUNO_LAB_TARGET_DENSE" in selected:
        if selected.pop("KUNO_LAB_TARGET_DENSE") not in ("verify3", "verify3-fused"):
            raise ValueError("Target dense policy requires canonical verify3")
    if "KUNO_LAB_TARGET_IO" in selected:
        if selected.pop("KUNO_LAB_TARGET_IO") != "chunk":
            raise ValueError("Target I/O policy requires canonical chunk")
    if "KUNO_LAB_TARGET_EXPERTS" in selected:
        if selected.pop("KUNO_LAB_TARGET_EXPERTS") != "prepared-verify3":
            raise ValueError("Target expert policy requires canonical prepared-verify3")
    if "KUNO_LAB_TARGET_FILL" in selected:
        if selected.pop("KUNO_LAB_TARGET_FILL") != "direct-public":
            raise ValueError("Target fill policy requires canonical direct-public")
    if mtp and "KUNO_LAB_MTP_SWEEPS" in selected:
        if selected.pop("KUNO_LAB_MTP_SWEEPS") != "fused":
            raise ValueError("MTP sweep policy requires canonical fused")
    if mtp and "KUNO_LAB_WARM_REPLAY" in selected:
        if selected.pop("KUNO_LAB_WARM_REPLAY") != "once":
            raise ValueError("Warm replay requires the one-shot public fixture")
    return selected

def original_frozen_target_scope(environment, selector, value, *, four_down=False, full_vocab=False):
    """Shared original target prerequisites; draft policies keep their own scope."""
    if selector not in environment:
        return
    spec = environment.get("KUNO_LAB_SPECULATIVE_PUBLIC_ONLY") == "1"
    language = environment.get("KUNO_LAB_LANGUAGE_PUBLIC_ONLY") == "1"
    required = {
        selector: value,
        "KUNO_LAB_TARGET_SOURCES": "frozen-host-public",
        "KUNO_LAB_FROZEN_PARTITION_EXIT": "1",
        "KUNO_LAB_TARGET_ROUTER": "device-frozen-public",
        "KUNO_LAB_TARGET_REUSE": "request",
        "KUNO_LAB_TARGET_FILL": "direct-public",
        "KUNO_LAB_DEVICE_DISPATCH": "inline",
        "KUNO_LAB_TARGET_IO": "chunk",
        "KUNO_LAB_TARGET_DENSE": "verify3-fused",
    }
    if four_down:
        required["KUNO_LAB_FROZEN_DOWN_ROWS"] = "four-public"
    conflicts = (
        "KUNO_LAB_FROZEN_TABLES", "KUNO_LAB_FROZEN_DOWN_SCALES",
        "KUNO_LAB_FROZEN_GU_FACTORS", "KUNO_LAB_FROZEN_GU_ROWS",
        "KUNO_LAB_FROZEN_HOST_STAGE", "KUNO_LAB_FROZEN_MEMBERS",
        "KUNO_LAB_FROZEN_BANK_PLACEMENT", "KUNO_LAB_TARGET_EXPERTS",
        "KUNO_LAB_TARGET_MOE",
    )
    if (any(environment.get(key) != expected for key, expected in required.items())
            or spec == language
            or spec and environment.get("KUNO_LAB_WARM_REPLAY") != "once"
            or language and "KUNO_LAB_WARM_REPLAY" in environment
            or any(key in environment for key in conflicts)
            or full_vocab and "KUNO_LAB_MTP_VOCAB" in environment):
        raise ValueError("Target read/schedule policy requires original bounded public frozen scope")


def grouped_status_scope(environment):
    original_frozen_target_scope(
        environment, "KUNO_LAB_TARGET_STATUS_FOLDS", "grouped-public", full_vocab=True,
    )


def grouped_ple_scope(environment):
    original_frozen_target_scope(
        environment, "KUNO_LAB_TARGET_PLE_READS", "grouped-public", full_vocab=True,
    )


def frozen_down_stride_scope(environment):
    original_frozen_target_scope(
        environment, "KUNO_LAB_FROZEN_DOWN_SCHEDULE", "four-public", four_down=True,
    )


def frozen_down_integer_scope(environment):
    selector = "KUNO_LAB_FROZEN_DOWN_MATH"
    if selector not in environment:
        return
    original_frozen_target_scope(environment, selector, "integer-public", full_vocab=True)
    if (environment.get("KUNO_LAB_MTP_SWEEPS") != "fused"
            or environment.get("KUNO_LAB_MTP_Q8_HEAD") not in (None, "shared-public")
            or any(key in environment for key in (
                "KUNO_LAB_MTP_PROFILE", "KUNO_LAB_LANGUAGE_DECODE",
                "KUNO_LAB_FROZEN_DOWN_ROWS", "KUNO_LAB_FROZEN_DOWN_SCHEDULE",
            ))):
        raise ValueError("Integer DOWN requires original speculative Q8 and original DOWN math")


def iq4_register_scope(environment):
    selector = "KUNO_LAB_TARGET_IQ4_DECODER"
    if selector not in environment:
        return
    original_frozen_target_scope(environment, selector, "packed-registers-public", full_vocab=True)
    if (environment.get("KUNO_LAB_MTP_DEVICE") != "intel-public"
            or environment.get("KUNO_LAB_MTP_SWEEPS") != "fused"
            or environment.get("KUNO_LAB_MTP_Q8_HEAD") not in (None, "shared-public")
            or any(key in environment for key in (
                "KUNO_LAB_MTP_PROFILE", "KUNO_LAB_LANGUAGE_DECODE",
                "KUNO_LAB_FROZEN_DOWN_SCHEDULE",
            ))):
        raise ValueError("IQ4 registers require original speculative Intel full-Q8 scope")



def target_submission_backlog_scope(environment):
    selector = "KUNO_LAB_TARGET_SUBMISSION_BACKLOG"
    if selector not in environment:
        return
    original_frozen_target_scope(environment, selector, "queued-2048-public", full_vocab=True)
    if (environment.get("KUNO_LAB_MTP_DEVICE") != "intel-public"
            or environment.get("KUNO_LAB_MTP_SWEEPS") != "fused"
            or environment.get("KUNO_LAB_MTP_Q8_HEAD") not in (None, "shared-public")
            or any(key in environment for key in (
                "KUNO_LAB_MTP_PROFILE", "KUNO_LAB_LANGUAGE_DECODE",
            ))):
        raise ValueError("Target backlog requires original speculative Intel full-Q8 scope")



def profile_bank_fill_scope(environment):
    selector = "KUNO_LAB_FROZEN_BANK_FILL"
    if selector not in environment:
        return
    original_frozen_target_scope(environment, selector, "profile-ranked-public", full_vocab=True)
    if (environment.get("KUNO_LAB_MTP_DEVICE") != "intel-public"
            or environment.get("KUNO_LAB_MTP_SWEEPS") != "fused"
            or environment.get("KUNO_LAB_MTP_Q8_HEAD") not in (None, "shared-public")
            or any(key in environment for key in (
                "KUNO_LAB_MTP_PROFILE", "KUNO_LAB_LANGUAGE_DECODE",
            ))):
        raise ValueError("Profile filler requires original speculative Intel full-Q8 scope")


def frozen_gu_integer_scope(environment):
    if not any(key in environment for key in ("KUNO_LAB_FROZEN_GU_MATH", "KUNO_LAB_FROZEN_GU_SCHEDULE")):
        return
    spec = environment.get("KUNO_LAB_SPECULATIVE_PUBLIC_ONLY") == "1"
    language = environment.get("KUNO_LAB_LANGUAGE_PUBLIC_ONLY") == "1"
    required = {
        "KUNO_LAB_FROZEN_GU_MATH": "integer-public",
        "KUNO_LAB_TARGET_SOURCES": "frozen-host-public",
        "KUNO_LAB_FROZEN_PARTITION_EXIT": "1",
        "KUNO_LAB_TARGET_ROUTER": "device-frozen-public",
        "KUNO_LAB_TARGET_REUSE": "request",
        "KUNO_LAB_TARGET_FILL": "direct-public",
        "KUNO_LAB_DEVICE_DISPATCH": "inline",
        "KUNO_LAB_TARGET_IO": "chunk",
        "KUNO_LAB_TARGET_DENSE": "verify3-fused",
    }
    target_only = environment.get("KUNO_LAB_LANGUAGE_DECODE") == "target-only-public"
    if target_only:
        target_only_scope(environment)
    else:
        required["KUNO_LAB_MTP_SWEEPS"] = "fused"
    conflicts = (
        "KUNO_LAB_FROZEN_TABLES", "KUNO_LAB_FROZEN_DOWN_SCALES",
        "KUNO_LAB_FROZEN_GU_FACTORS", "KUNO_LAB_FROZEN_GU_ROWS",
        "KUNO_LAB_FROZEN_HOST_STAGE", "KUNO_LAB_FROZEN_MEMBERS",
        "KUNO_LAB_FROZEN_BANK_PLACEMENT", "KUNO_LAB_TARGET_EXPERTS",
        "KUNO_LAB_TARGET_MOE", "KUNO_LAB_MTP_PROFILE", "KUNO_LAB_MTP_VOCAB",
    )
    if (any(environment.get(key) != value for key, value in required.items())
            or environment.get("KUNO_LAB_FROZEN_GU_SCHEDULE") not in (None, "four-public")
            or spec == language
            or spec and environment.get("KUNO_LAB_WARM_REPLAY") != "once"
            or environment.get("KUNO_LAB_MTP_Q8_HEAD") not in (None, "shared-public")
            or any(key in environment for key in conflicts)):
        raise ValueError("Integer GU requires the original bounded public GPU-router scope")

def target_only_scope(environment):
    if "KUNO_LAB_LANGUAGE_DECODE" not in environment:
        return
    required = {
        "KUNO_LAB_LANGUAGE_DECODE": "target-only-public",
        "KUNO_LAB_LANGUAGE_PUBLIC_ONLY": "1",
        "KUNO_LAB_TARGET_SOURCES": "frozen-host-public",
        "KUNO_LAB_FROZEN_PARTITION_EXIT": "1",
        "KUNO_LAB_TARGET_ROUTER": "device-frozen-public",
        "KUNO_LAB_TARGET_REUSE": "request",
        "KUNO_LAB_TARGET_FILL": "direct-public",
        "KUNO_LAB_DEVICE_DISPATCH": "inline",
        "KUNO_LAB_TARGET_IO": "chunk",
        "KUNO_LAB_TARGET_DENSE": "verify3-fused",
    }
    conflicts = (
        "KUNO_LAB_SPECULATIVE_PUBLIC_ONLY", "KUNO_LAB_WARM_REPLAY",
        "KUNO_LAB_MTP_SWEEPS", "KUNO_LAB_MTP_VOCAB", "KUNO_LAB_MTP_PROFILE",
        "KUNO_LAB_MTP_Q8_HEAD", "KUNO_LAB_MTP_DEVICE", "KUNO_LAB_TARGET_GDN",
        "KUNO_LAB_FROZEN_TABLES", "KUNO_LAB_FROZEN_DOWN_SCALES",
        "KUNO_LAB_FROZEN_GU_FACTORS", "KUNO_LAB_FROZEN_GU_ROWS",
        "KUNO_LAB_FROZEN_HOST_STAGE", "KUNO_LAB_FROZEN_MEMBERS",
        "KUNO_LAB_FROZEN_BANK_PLACEMENT", "KUNO_LAB_TARGET_EXPERTS",
        "KUNO_LAB_TARGET_MOE", "KUNO_LAB_FROZEN_DOWN_MATH", "KUNO_LAB_TARGET_IQ4_DECODER", "KUNO_LAB_TARGET_SUBMISSION_BACKLOG", "KUNO_LAB_FROZEN_BANK_FILL", "KUNO_PUBLIC_MTP_INTEL_TOTAL_BYTES",
        "KUNO_PUBLIC_MTP_INTEL_USED_BYTES", "KUNO_PUBLIC_MTP_INTEL_DEVICE_ID",
    )
    if (any(environment.get(key) != value for key, value in required.items())
            or environment.get("KUNO_LAB_LANGUAGE_FIXTURE") not in ("first-code", "warm-prose")
            or any(key in environment for key in conflicts)):
        raise ValueError("Target-only requires fixed original public target and no draft selectors")

def intel_mtp_scope(environment):
    metadata = ("KUNO_PUBLIC_MTP_INTEL_TOTAL_BYTES", "KUNO_PUBLIC_MTP_INTEL_USED_BYTES", "KUNO_PUBLIC_MTP_INTEL_DEVICE_ID")
    if not any(key in environment for key in ("KUNO_LAB_MTP_DEVICE", *metadata)):
        return
    required = {
        "KUNO_LAB_MTP_DEVICE": "intel-public",
        "KUNO_LAB_TARGET_SOURCES": "frozen-host-public",
        "KUNO_LAB_FROZEN_PARTITION_EXIT": "1",
        "KUNO_LAB_TARGET_ROUTER": "device-frozen-public",
        "KUNO_LAB_TARGET_REUSE": "request",
        "KUNO_LAB_TARGET_FILL": "direct-public",
        "KUNO_LAB_DEVICE_DISPATCH": "inline",
        "KUNO_LAB_TARGET_IO": "chunk",
        "KUNO_LAB_TARGET_DENSE": "verify3-fused",
        "KUNO_LAB_MTP_SWEEPS": "fused",
    }
    spec = environment.get("KUNO_LAB_SPECULATIVE_PUBLIC_ONLY") == "1"
    language = environment.get("KUNO_LAB_LANGUAGE_PUBLIC_ONLY") == "1"
    conflicts = (
        "KUNO_LAB_LANGUAGE_DECODE", "KUNO_LAB_MTP_VOCAB", "KUNO_LAB_FROZEN_TABLES",
        "KUNO_LAB_FROZEN_DOWN_SCALES", "KUNO_LAB_FROZEN_GU_FACTORS", "KUNO_LAB_FROZEN_GU_ROWS",
        "KUNO_LAB_FROZEN_HOST_STAGE", "KUNO_LAB_FROZEN_MEMBERS", "KUNO_LAB_TARGET_EXPERTS",
        "KUNO_LAB_TARGET_MOE", "KUNO_LAB_FROZEN_BANK_PLACEMENT",
    )
    if (any(environment.get(key) != value for key, value in required.items())
            or spec == language
            or spec and environment.get("KUNO_LAB_WARM_REPLAY") != "once"
            or any(key in environment for key in conflicts)):
        raise ValueError("Intel draft placement requires bounded original public frozen scope")
    values = []
    for key in metadata:
        value = environment.get(key, "")
        if not value or not value.isascii() or not value.isdecimal() or value.startswith("0"):
            raise ValueError("Intel draft metadata must be canonical positive decimal")
        parsed = int(value)
        if parsed > (2**32 - 1 if key.endswith("DEVICE_ID") else 2**64 - 1):
            raise ValueError("Intel draft metadata overflow")
        values.append(parsed)
    total, used, _ = values
    if used >= total:
        raise ValueError("Intel draft metadata must report actual nonzero free memory")

def english_vocab_scope(environment):
    if "KUNO_LAB_MTP_VOCAB" not in environment:
        return
    spec = environment.get("KUNO_LAB_SPECULATIVE_PUBLIC_ONLY") == "1"
    language = environment.get("KUNO_LAB_LANGUAGE_PUBLIC_ONLY") == "1"
    required = {
        "KUNO_LAB_MTP_VOCAB": "english-public",
        "KUNO_LAB_TARGET_SOURCES": "frozen-host-public",
        "KUNO_LAB_FROZEN_PARTITION_EXIT": "1",
        "KUNO_LAB_TARGET_ROUTER": "device-frozen-public",
        "KUNO_LAB_TARGET_REUSE": "request",
        "KUNO_LAB_TARGET_FILL": "direct-public",
        "KUNO_LAB_DEVICE_DISPATCH": "inline",
        "KUNO_LAB_TARGET_IO": "chunk",
        "KUNO_LAB_TARGET_DENSE": "verify3-fused",
        "KUNO_LAB_MTP_SWEEPS": "fused",
    }
    if (any(environment.get(key) != value for key, value in required.items())
            or spec == language
            or spec and environment.get("KUNO_LAB_WARM_REPLAY") != "once"
            or environment.get("KUNO_LAB_MTP_Q8_HEAD") not in (None, "shared-public")
            or any(key in environment for key in ("KUNO_LAB_MTP_PROFILE", "KUNO_LAB_TARGET_MOE", "KUNO_LAB_TARGET_EXPERTS"))):
        raise ValueError("English draft vocabulary requires original fixed public fused Q8 scope")

def three_gdn_scope(environment):
    if "KUNO_LAB_TARGET_GDN" not in environment:
        return
    spec = environment.get("KUNO_LAB_SPECULATIVE_PUBLIC_ONLY") == "1"
    language = environment.get("KUNO_LAB_LANGUAGE_PUBLIC_ONLY") == "1"
    required = {
        "KUNO_LAB_TARGET_GDN": "three-streaming-public",
        "KUNO_LAB_TARGET_SOURCES": "frozen-host-public",
        "KUNO_LAB_FROZEN_PARTITION_EXIT": "1",
        "KUNO_LAB_TARGET_ROUTER": "device-frozen-public",
        "KUNO_LAB_TARGET_REUSE": "request",
        "KUNO_LAB_TARGET_FILL": "direct-public",
        "KUNO_LAB_DEVICE_DISPATCH": "inline",
        "KUNO_LAB_TARGET_IO": "chunk",
        "KUNO_LAB_TARGET_DENSE": "verify3-fused",
    }
    if (any(environment.get(key) != value for key, value in required.items())
            or spec == language
            or spec and environment.get("KUNO_LAB_WARM_REPLAY") != "once"
            or any(key in environment for key in ("KUNO_LAB_TARGET_MOE", "KUNO_LAB_TARGET_EXPERTS"))):
        raise ValueError("Three-token GDN requires original fixed public frozen Verify3 scope")

def private_table_scope(environment):
    if any(key in environment for key in ("KUNO_LAB_FROZEN_TABLES", "KUNO_LAB_FROZEN_DOWN_SCALES", "KUNO_LAB_FROZEN_GU_FACTORS", "KUNO_LAB_FROZEN_DOWN_ROWS", "KUNO_LAB_FROZEN_GU_ROWS", "KUNO_LAB_FROZEN_HOST_STAGE", "KUNO_LAB_FROZEN_MEMBERS")):
        spec = environment.get("KUNO_LAB_SPECULATIVE_PUBLIC_ONLY") == "1"
        language = environment.get("KUNO_LAB_LANGUAGE_PUBLIC_ONLY") == "1"
        if (environment.get("KUNO_LAB_FROZEN_TABLES") not in (None, "private-public", "workgroup-public")
                or environment.get("KUNO_LAB_FROZEN_DOWN_SCALES") not in (None, "shared-public")
                or environment.get("KUNO_LAB_FROZEN_GU_FACTORS") not in (None, "shared-public")
                or environment.get("KUNO_LAB_FROZEN_DOWN_ROWS") not in (None, "four-public")
                or environment.get("KUNO_LAB_FROZEN_GU_ROWS") not in (None, "two-public")
                or environment.get("KUNO_LAB_FROZEN_HOST_STAGE") not in (None, "packed-public")
                or environment.get("KUNO_LAB_FROZEN_MEMBERS") not in (None, "three-public")
                or environment.get("KUNO_LAB_FROZEN_HOST_STAGE") is not None and (
                    environment.get("KUNO_LAB_TARGET_ROUTER") != "device-frozen-public"
                    or any(key in environment for key in ("KUNO_LAB_FROZEN_TABLES", "KUNO_LAB_FROZEN_GU_FACTORS", "KUNO_LAB_FROZEN_DOWN_SCALES")))
                or environment.get("KUNO_LAB_FROZEN_MEMBERS") is not None and (
                    environment.get("KUNO_LAB_TARGET_ROUTER") != "device-frozen-public"
                    or any(key in environment for key in ("KUNO_LAB_FROZEN_TABLES", "KUNO_LAB_FROZEN_GU_FACTORS", "KUNO_LAB_FROZEN_DOWN_SCALES", "KUNO_LAB_FROZEN_GU_ROWS")))
                or environment.get("KUNO_LAB_FROZEN_GU_ROWS") is not None and any(
                    key in environment for key in ("KUNO_LAB_FROZEN_TABLES", "KUNO_LAB_FROZEN_GU_FACTORS"))
                or environment.get("KUNO_LAB_TARGET_SOURCES") != "frozen-host-public"
                or environment.get("KUNO_LAB_FROZEN_PARTITION_EXIT") != "1"
                or spec == language
                or spec and environment.get("KUNO_LAB_WARM_REPLAY") != "once"):
            raise ValueError("Cooperative kernels require the canonical frozen public partition-exit scope")


def bank_placement_scope(environment):
    if "KUNO_LAB_FROZEN_BANK_PLACEMENT" in environment:
        if (environment.get("KUNO_LAB_FROZEN_BANK_PLACEMENT") != "preserve-public"
                or environment.get("KUNO_LAB_TARGET_SOURCES") != "frozen-host-public"
                or environment.get("KUNO_LAB_SPECULATIVE_PUBLIC_ONLY") != "1"
                or environment.get("KUNO_LAB_WARM_REPLAY") != "once"):
            raise ValueError("Bank placement requires the canonical warm frozen public scope")


def memory_limit(environment, arguments):
    if ("KUNO_LAB_HOST_CPU_BATCH_PUBLIC_ONLY" in environment
            or len(arguments) > 2 and arguments[2] == "public-host-cpu-batch-probe"):
        selected = {key: value for key, value in environment.items()
                    if key.startswith(("KUNO_LAB_", "KUNO_TEST_", "KUNO_PUBLIC_"))}
        required = {"KUNO_LAB_HOST_CPU_BATCH_PUBLIC_ONLY", "KUNO_LAB_DEVICE_DISPATCH",
                    "KUNO_LAB_STATIC_CACHE_BYTES",
                    "KUNO_TEST_Q3_DIR", "KUNO_PUBLIC_TARGET_CARRY13_REFERENCE",
                    "KUNO_PUBLIC_RECEIPT"}
        if (set(selected) != required or any(not value for value in selected.values())
                or selected["KUNO_LAB_HOST_CPU_BATCH_PUBLIC_ONLY"] != "1"
                or selected["KUNO_LAB_DEVICE_DISPATCH"] != "inline"
                or selected["KUNO_LAB_STATIC_CACHE_BYTES"] != "8589934592"
                or len(arguments) != 3
                or arguments[1:] != ["intel", "public-host-cpu-batch-probe"]):
            raise ValueError("HOST CPU batches require the fixed Intel original public operator")
        # The original model/cache/mirror envelope is retained. This is not
        # extra private capacity: CPU1MiB/GPU operator owners remain independent.
        return 48 * 1024**3
    target_only_scope(environment)
    intel_mtp_scope(environment)
    frozen_down_integer_scope(environment)
    iq4_register_scope(environment)
    target_submission_backlog_scope(environment)
    profile_bank_fill_scope(environment)
    grouped_status_scope(environment)
    grouped_ple_scope(environment)
    frozen_down_stride_scope(environment)
    frozen_gu_integer_scope(environment)
    english_vocab_scope(environment)
    three_gdn_scope(environment)
    private_table_scope(environment)
    bank_placement_scope(environment)
    if len(arguments) > 2 and arguments[2] == "iq-quant-probe":
        selected = {key: value for key, value in environment.items() if key.startswith(("KUNO_LAB_", "KUNO_TEST_", "KUNO_PUBLIC_"))}
        if (set(selected) != {"KUNO_PUBLIC_RECEIPT"}
                or not selected["KUNO_PUBLIC_RECEIPT"]
                or len(arguments) != 3 or arguments[1] != "intel"):
            raise ValueError("IQ4 value acceptance requires the fixed Intel compiled public fixtures")
        return 8 * 1024**3
    if "KUNO_LAB_PUBLIC_HOST_CUBECL_ONLY" in environment or (len(arguments) > 2 and arguments[2] == "public-host-weight-cubecl-probe"):
        selected = {key: value for key, value in environment.items() if key.startswith(("KUNO_LAB_", "KUNO_TEST_", "KUNO_PUBLIC_"))}
        if (set(selected) != {"KUNO_LAB_PUBLIC_HOST_CUBECL_ONLY", "KUNO_PUBLIC_RECEIPT"}
                or any(not value for value in selected.values())
                or selected["KUNO_LAB_PUBLIC_HOST_CUBECL_ONLY"] != "1"
                or len(arguments) != 3 or arguments[1] not in ("intel", "nvidia") or arguments[2] != "public-host-weight-cubecl-probe"):
            raise ValueError("CubeCL host weights require the compiled original public fixture")
        # The compiled probe rejects unsupported build features before device
        # creation. This launcher admits no model, private input or policy flag.
        return 8 * 1024**3
    if "KUNO_LAB_QSA_CHRONOLOGICAL_PUBLIC_ONLY" in environment or (len(arguments) > 2 and arguments[2] == "qsa-chronological-probe"):
        selected = {key: value for key, value in environment.items() if key.startswith(("KUNO_LAB_", "KUNO_TEST_", "KUNO_PUBLIC_"))}
        if (set(selected) != {"KUNO_LAB_QSA_CHRONOLOGICAL_PUBLIC_ONLY", "KUNO_PUBLIC_RECEIPT"}
                or any(not value for value in selected.values())
                or selected["KUNO_LAB_QSA_CHRONOLOGICAL_PUBLIC_ONLY"] != "1"
                or len(arguments) != 3 or arguments[1] not in ("intel", "nvidia") or arguments[2] != "qsa-chronological-probe"):
            raise ValueError("QSA selection requires the compiled original public fixture")
        return 8 * 1024**3
    if "KUNO_LAB_PUBLIC_HOST_WEIGHT_ONLY" in environment or (len(arguments) > 2 and arguments[2] == "public-host-weight-probe"):
        selected = {key: value for key, value in environment.items() if key.startswith(("KUNO_LAB_", "KUNO_TEST_", "KUNO_PUBLIC_"))}
        if (set(selected) != {"KUNO_LAB_PUBLIC_HOST_WEIGHT_ONLY", "KUNO_PUBLIC_RECEIPT"}
                or any(not value for value in selected.values())
                or selected["KUNO_LAB_PUBLIC_HOST_WEIGHT_ONLY"] != "1"
                or len(arguments) != 3 or arguments[1] not in ("intel", "nvidia") or arguments[2] != "public-host-weight-probe"):
            raise ValueError("Host weights require the compiled original public fixture")
        return 8 * 1024**3
    if "KUNO_LAB_STRATA_MTP_PUBLIC_ONLY" in environment or (len(arguments) > 2 and arguments[2] == "strata-mtp-probe"):
        selected = {key: value for key, value in environment.items() if key.startswith(("KUNO_LAB_", "KUNO_TEST_", "KUNO_PUBLIC_"))}
        if "KUNO_LAB_MTP_SWEEPS" in selected and selected.pop("KUNO_LAB_MTP_SWEEPS") != "fused":
            raise ValueError("Q2 draft requires canonical fused sweeps")
        permitted = {"KUNO_LAB_STRATA_MTP_PUBLIC_ONLY", "KUNO_TEST_STRATA_MTP", "KUNO_TEST_DASLAB_DIR", "KUNO_PUBLIC_DASLAB_CARRY", "KUNO_PUBLIC_STRATA_MTP_REFERENCE", "KUNO_PUBLIC_RECEIPT"}
        if (set(selected) != permitted or any(not value for value in selected.values())
                or selected["KUNO_LAB_STRATA_MTP_PUBLIC_ONLY"] != "1"
                or len(arguments) != 3 or arguments[1] not in ("intel", "nvidia") or arguments[2] != "strata-mtp-probe"):
            raise ValueError("Q2 draft requires the fixed independent public reference")
        return 8 * 1024**3

    if "KUNO_LAB_LANGUAGE_PUBLIC_ONLY" in environment:
        selected = selected_target_policy(environment, mtp=True)
        required = {"KUNO_LAB_LANGUAGE_PUBLIC_ONLY","KUNO_LAB_LANGUAGE_FIXTURE","KUNO_TEST_Q3_DIR","KUNO_TEST_MTP","KUNO_PUBLIC_RECEIPT","KUNO_LAB_STATIC_CACHE_BYTES"}
        if (set(selected) != required or any(not value for value in selected.values())
                or selected["KUNO_LAB_LANGUAGE_PUBLIC_ONLY"] != "1"
                or selected["KUNO_LAB_LANGUAGE_FIXTURE"] not in ("first-code","warm-prose")
                or selected["KUNO_LAB_STATIC_CACHE_BYTES"] != "8589934592"
                or len(arguments)!=3 or arguments[1:]!=["intel","daslab-language-probe"]
                or "KUNO_LAB_WARM_REPLAY" in environment
                or environment.get("KUNO_LAB_TARGET_ROUTER") not in (None, "device-frozen-public")):
            raise ValueError("Language benchmark requires the frozen public fixture")
        # Includes separately declared192MiB tokenizer and4MiB fixture owners;
        # these are never borrowed from GPU/staging/runtime reservations.
        return 48*1024**3 + 196*1024**2

    if "KUNO_LAB_BATCH_ROUTER_PUBLIC_ONLY" in environment:
        selected = {key:value for key,value in environment.items() if key.startswith(("KUNO_LAB_","KUNO_TEST_","KUNO_PUBLIC_"))}
        if (set(selected) != {"KUNO_LAB_BATCH_ROUTER_PUBLIC_ONLY","KUNO_PUBLIC_RECEIPT"}
                or any(not value for value in selected.values())
                or selected["KUNO_LAB_BATCH_ROUTER_PUBLIC_ONLY"] != "1"
                or len(arguments)!=3 or arguments[1] not in ("intel","nvidia") or arguments[2]!="batch-router-gpu-probe"):
            raise ValueError("Batch router requires the fixed public fixture")
        return 8*1024**3

    if "KUNO_LAB_IQ_LIVE_MEMBERS_PUBLIC_ONLY" in environment:
        selected = {key:value for key,value in environment.items() if key.startswith(("KUNO_LAB_","KUNO_TEST_","KUNO_PUBLIC_"))}
        if (set(selected) != {"KUNO_LAB_IQ_LIVE_MEMBERS_PUBLIC_ONLY","KUNO_PUBLIC_RECEIPT"}
                or any(not value for value in selected.values())
                or selected["KUNO_LAB_IQ_LIVE_MEMBERS_PUBLIC_ONLY"] != "1"
                or len(arguments)!=3 or arguments[1] not in ("intel","nvidia") or arguments[2]!="iq-live-members-moe-probe"):
            raise ValueError("Live-member IQ operator requires the fixed public fixture")
        return 8*1024**3
    if "KUNO_LAB_VERIFY3_FINITE_PUBLIC_ONLY" in environment:
        selected = {key: value for key, value in environment.items() if key.startswith(("KUNO_LAB_", "KUNO_TEST_", "KUNO_PUBLIC_"))}
        if (set(selected) != {"KUNO_LAB_VERIFY3_FINITE_PUBLIC_ONLY", "KUNO_PUBLIC_RECEIPT"}
                or any(not value for value in selected.values())
                or selected["KUNO_LAB_VERIFY3_FINITE_PUBLIC_ONLY"] != "1"
                or len(arguments) != 3 or arguments[1] not in ("intel", "nvidia") or arguments[2] != "verify3-finite-probe"):
            raise ValueError("Verify3 finite operator requires the fixed public fixture")
        return 8 * 1024**3
    if "KUNO_LAB_VERIFY3_DENSE_PUBLIC_ONLY" in environment:
        selected = {key: value for key, value in environment.items() if key.startswith(("KUNO_LAB_", "KUNO_TEST_", "KUNO_PUBLIC_"))}
        if (set(selected) != {"KUNO_LAB_VERIFY3_DENSE_PUBLIC_ONLY", "KUNO_PUBLIC_RECEIPT"}
                or any(not value for value in selected.values())
                or selected["KUNO_LAB_VERIFY3_DENSE_PUBLIC_ONLY"] != "1"
                or len(arguments) != 3 or arguments[1] not in ("intel", "nvidia") or arguments[2] != "verify3-dense-probe"):
            raise ValueError("Verify3 operator requires the fixed public fixture")
        return 8 * 1024**3
    if "KUNO_LAB_SPECULATIVE_PUBLIC_ONLY" in environment:
        permitted = {"KUNO_LAB_SPECULATIVE_PUBLIC_ONLY", "KUNO_LAB_STATIC_CACHE_BYTES", "KUNO_TEST_Q3_DIR", "KUNO_TEST_MTP", "KUNO_PUBLIC_TARGET_CARRY13_REFERENCE", "KUNO_PUBLIC_RECEIPT"}
        selected = selected_target_policy(environment, mtp=True)
        if (set(selected) != permitted or any(not value for value in selected.values())
                or selected["KUNO_LAB_SPECULATIVE_PUBLIC_ONLY"] != "1"
                or selected["KUNO_LAB_STATIC_CACHE_BYTES"] != "8589934592"
                or len(arguments) != 3 or arguments[1:] != ["intel", "daslab-speculative-probe"]):
            raise ValueError("Speculative cache requires the fixed public dual-GPU fixture")
        return 48 * 1024**3
    if "KUNO_LAB_RESIDENT_MTP_PUBLIC_ONLY" in environment or (len(arguments) > 2 and arguments[2] == "resident-mtp-probe"):
        selected = {key: value for key, value in environment.items() if key.startswith(("KUNO_LAB_", "KUNO_TEST_", "KUNO_PUBLIC_"))}
        if "KUNO_LAB_MTP_SWEEPS" in selected and selected.pop("KUNO_LAB_MTP_SWEEPS") != "fused":
            raise ValueError("MTP sweep policy requires canonical fused")
        permitted={"KUNO_LAB_RESIDENT_MTP_PUBLIC_ONLY", "KUNO_TEST_MTP", "KUNO_PUBLIC_DASLAB_CARRY", "KUNO_PUBLIC_MTP_REFERENCE", "KUNO_PUBLIC_RECEIPT"}
        if (set(selected) != permitted or any(not value for value in selected.values())
                or selected["KUNO_LAB_RESIDENT_MTP_PUBLIC_ONLY"] != "1"
                or len(arguments) != 3 or arguments[1] not in ("intel", "nvidia") or arguments[2] != "resident-mtp-probe"):
            raise ValueError("Resident MTP requires fixed public artifacts")
        return 8 * 1024**3
    target_modes = {"KUNO_LAB_TARGET_ROWS_PUBLIC_ONLY": "daslab-target-rows-probe", "KUNO_LAB_TARGET_REPAIR_PUBLIC_ONLY": "daslab-target-repair-probe"}
    target_gate = next((key for key in target_modes if key in environment), None)
    if target_gate is not None:
        permitted = {target_gate, "KUNO_LAB_STATIC_CACHE_BYTES", "KUNO_TEST_Q3_DIR", "KUNO_PUBLIC_TARGET_ROWS_REFERENCE", "KUNO_PUBLIC_RECEIPT"}
        selected = selected_target_policy(environment)
        if (set(selected) != permitted or any(not value for value in selected.values())
                or selected[target_gate] != "1"
                or selected["KUNO_LAB_STATIC_CACHE_BYTES"] != "8589934592"
                or len(arguments) != 3 or arguments[1:] != ["intel", target_modes[target_gate]]):
            raise ValueError("Larger target cache requires the fixed public DASLab target fixture")
        # Original mapped model pages and the separately bounded 8GiB cache;
        # this does not enlarge any request-derived owning allocation.
        return 48 * 1024**3
    value = environment.get("KUNO_LAB_LARGE_WEIGHT_CACHE", "0")
    if value == "0":
        return 8 * 1024**3
    if value != "1" or environment.get("KUNO_LAB_MAPPED_WEIGHTS") != "1" or len(arguments) != 4 or arguments[1] not in ("intel", "nvidia") or arguments[2] != "resident-count1-hybrid-chat" or arguments[3] not in tuple(str(cap) for cap in range(11)):
        raise ValueError("Larger public cache requires the fixed mapped hybrid fixture")
    # Public model page cache, not a private-buffer allowance. The native MAIN
    # and CPU owning ledgers are unchanged. Bound the whole process to48GiB.
    return 48 * 1024**3

def main():
    try:
        expected = memory_limit(os.environ, sys.argv[1:])
    except ValueError as error:
        raise SystemExit(str(error))
    cgroup = next(x.split(":", 2)[2].strip() for x in pathlib.Path("/proc/self/cgroup").read_text().splitlines() if x.startswith("0::"))
    path = pathlib.Path("/sys/fs/cgroup") / cgroup.lstrip("/")
    if (path / "memory.swap.max").read_text().strip() != "0" or resource.getrlimit(resource.RLIMIT_CORE) != (0, 0):
        raise SystemExit("Protected laboratory cgroup limits missing")
    if (path / "memory.max").read_text().strip() != str(expected):
        raise SystemExit("Protected laboratory memory limit mismatch")
    print(f"verified_lab_limits swap_max=0 core_soft=0 core_hard=0 memory_max={expected}", flush=True)
    os.execv(sys.argv[1], sys.argv[1:])

if __name__ == "__main__":
    main()
