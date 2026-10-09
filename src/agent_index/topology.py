# MoE dispatch expansion derived from NVIDIA AISimulate; see THIRD_PARTY_NOTICES.md.
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Pinned TPU8i hop overlay from the original cross-hardware study.

This is an uncalibrated sensitivity model, not a physical Boardfly simulator.
Native AISim supplies communication byte-transfer time. We add only the
assumed 250 ns per serialized hop; both overlap branches are charged.
"""

def hops(span):
    return 0 if span <= 1 else 1 if span <= 4 else 3 if span <= 32 else 7


def phase_hops(n, kind):
    if n <= 1:
        return 0
    if n & (n - 1):
        raise ValueError("The pinned overlay supports power-of-two groups only")
    if kind not in ("all_reduce", "all_gather", "reduce_scatter", "alltoall"):
        raise ValueError(f"Unaudited collective: {kind}")
    if kind == "alltoall":
        count = sum(hops(1 << peer.bit_length()) for peer in range(1, n))
    else:
        count = sum(hops(1 << level) for level in range(1, n.bit_length()))
    return count * (2 if kind == "all_reduce" else 1)


def events(spec, parent=""):
    """Expand pinned native operation descriptors; fail on unknown collectives."""
    typ, value = next(iter(spec.items()))
    name = value.get("name", typ)
    multiplier = value.get("scale_factor", 1)
    result = []

    def add(n, kind):
        if n > 1 and multiplier:
            result.append(dict(operation=name, parent=parent, participants=n,
                               kind=kind, occurrences=multiplier))

    if typ == "CustomAllReduce":
        add(value["tp_size"], "all_reduce")
    elif typ == "Nccl":
        add(value["num_gpus"], value["operation"])
    elif typ == "Overlap":
        for branch in ("group_a", "group_b"):
            for child in value[branch]:
                child_events = events(child, name + "/" + branch)
                for event in child_events:
                    event["occurrences"] *= multiplier
                result.extend(child_events)
    elif typ == "MoeDispatch":
        if value["flavor"] != "CustomAllReduce" or value["attn_cp_size"] != 1:
            raise ValueError("Unaudited MoE dispatch recipe")
        n = value["moe_tp_size"] * value["moe_ep_size"]
        dp = value["attention_dp_size"]
        tp = n // dp
        pre = value["pre_dispatch"]
        if value["backend"] == "vllm":
            if tp > 1 and not (pre and value["attn_ar_modeled"]):
                add(n, "all_reduce")
            if dp > 1:
                add(n, "all_gather" if pre else "reduce_scatter")
        elif value["backend"] == "sglang":
            if tp > 1 and dp > 1:
                add(tp, "reduce_scatter" if pre else "all_gather")
                add(n, "all_gather" if pre else "reduce_scatter")
            elif tp > 1:
                add(n, "all_reduce")
            elif dp > 1:
                add(n, "all_gather" if pre else "reduce_scatter")
        else:
            raise ValueError(value["backend"])
    elif typ == "P2P":
        if value["pp_size"] != 1 or multiplier != 0:
            raise ValueError("Hop overlay does not support pipeline parallelism")
    elif any(x in typ.lower() for x in ("comm", "dispatch", "alltoall", "reduce", "nccl")):
        raise ValueError("Unaudited communication family: " + typ)
    return result


def penalty_ms(specs):
    return sum(event["occurrences"] * phase_hops(event["participants"], event["kind"])
               for spec in specs for event in events(spec)) * .25 / 1000
