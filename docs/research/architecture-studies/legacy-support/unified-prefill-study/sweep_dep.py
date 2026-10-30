#!/usr/bin/env python3
"""DEP8/16/32/64 sensitivity for the unified 256K+1K prefill workload."""

from __future__ import annotations

import contextlib
import csv
import hashlib
import io
import json
import os
import sys
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BASE_DIR = ROOT / "experiments/5t-architecture-design-study"
SWA_DIR = ROOT / "experiments/5t-swa-architecture-study"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(BASE_DIR))
sys.path.insert(0, str(SWA_DIR))
os.environ.setdefault("MPLCONFIGDIR", "/tmp/aic-unified-prefill-dep-mpl")

import run_unified_prefill as u  # noqa: E402
import run_study as base  # noqa: E402
import run_swa_study as swa  # noqa: E402
from aiconfigurator.cli.api import cli_estimate  # noqa: E402


OUTPUT = HERE / "output"
CACHE = OUTPUT / "dep_estimate_cache.jsonl"
DEPS = (8, 16, 32, 64)
CPRS = (1, 2, 4, 8)
SYSTEMS = u.SYSTEMS
PREFIX = u.PREFIX
CHUNK = 1024
GIB = 2**30
CAPACITY = u.CAPACITY_GIB


def signature(model: str) -> str:
    path = Path(model) / "config.json"
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16] if path.exists() else "builtin"


def key(model: str, system: str, dep: int, cpr: int) -> str:
    return json.dumps([model, signature(model), system, PREFIX, CHUNK, dep, cpr], separators=(",", ":"))


def load_cache() -> dict[str, dict[str, Any]]:
    if not CACHE.exists():
        return {}
    return {
        row["cache_key"]: row
        for row in (json.loads(line) for line in CACHE.read_text(encoding="utf-8").splitlines() if line.strip())
    }


def estimate(model: str, system: str, dep: int, cpr: int) -> dict[str, Any]:
    stdout, stderr = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
        result = cli_estimate(
            model, base.SYSTEMS[system]["id"], mode="static_ctx",
            backend_name="sglang", backend_version="estimate", database_mode="SOL",
            isl=PREFIX + CHUNK, osl=1, prefix=PREFIX, batch_size=cpr,
            tp_size=1, pp_size=1, attention_dp_size=dep,
            moe_tp_size=1, moe_ep_size=dep,
            gemm_quant_mode="fp8_block", moe_quant_mode="w4a8_mxfp4_mxfp8_trtllm",
            kvcache_quant_mode="fp8", fmha_quant_mode="bfloat16",
            stride=1, engine_step_backend="python", systems_paths=base.SYSTEMS_PATHS,
        )
    latency = {str(k): float(v) for k, v in result.summary.get_context_latency_dict().items()}
    sources = {str(k): str(v or "unknown").lower() for k, v in result.summary.get_context_source_dict().items()}
    memory = result.summary.get_memory()
    return {
        "cache_key": key(model, system, dep, cpr), "model": model,
        "system": system, "dep": dep, "cpr": cpr,
        "latency_ms": sum(latency.values()), "latency": latency, "sources": sources,
        "non_sol_ms": sum(v for k, v in latency.items() if v > 0 and sources.get(k) != "sol"),
        "weights_gib": float(memory["weights"]), "kvcache_gib": float(memory["kvcache"]),
        "activations_gib": float(memory["activations"]), "nccl_gib": float(memory["nccl"]),
        "others_gib": float(memory["others"]), "memory_gib": float(memory["total"]),
        "oom": bool(result.summary.check_oom()), **u.classify_latency(latency),
    }


def run(models: list[str]) -> dict[str, dict[str, Any]]:
    models = list(dict.fromkeys(models))
    cache = load_cache()
    tasks = [(m, s, d, c) for m in models for s in SYSTEMS for d in DEPS for c in CPRS]
    pending = [task for task in tasks if key(*task) not in cache]
    print(f"DEP references: {len(tasks)-len(pending)} cached, {len(pending)} pending", flush=True)
    if pending:
        with CACHE.open("a", encoding="utf-8") as stream:
            for index, task in enumerate(pending, 1):
                row = estimate(*task)
                cache[row["cache_key"]] = row
                stream.write(json.dumps(row, sort_keys=True) + "\n")
                stream.flush()
                if index % 250 == 0 or index == len(pending):
                    print(f"  completed {index}/{len(pending)}", flush=True)
    return cache


def get(cache: dict[str, dict[str, Any]], model: str, system: str, dep: int, cpr: int):
    return cache[key(model, system, dep, cpr)]


def metrics(latency: float, dep: int, cpr: int) -> dict[str, Any]:
    return {
        "latency_ms": latency, "input_tps_user": CHUNK * 1000 / latency,
        "group_input_tps": dep * cpr * CHUNK * 1000 / latency,
        "input_tps_per_gpu": cpr * CHUNK * 1000 / latency,
        # Normalize to 64 GPUs by replicating independent DEP-sized groups.
        "normalized_64gpu_input_tps": 64 * cpr * CHUNK * 1000 / latency,
        "global_concurrency_per_group": dep * cpr,
    }


def direct_rows(exact: list[dict[str, Any]], cache: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for model in exact:
        for system in SYSTEMS:
            for dep in DEPS:
                for cpr in CPRS:
                    run_ = get(cache, model["model"], system, dep, cpr)
                    row = {
                        "architecture_id": "exact__" + model["id"], "label": model["label"],
                        "family": model["family"], "study_kind": "direct AIC config",
                        "attention": model["attention"], "total_t": model.get("total_t"),
                        "active_b": model.get("active_b"), "depth": model.get("depth"),
                        "experts": model.get("experts"), "topk": model.get("topk"),
                        "system": system, "dep": dep, "cpr": cpr,
                        **metrics(run_["latency_ms"], dep, cpr),
                        "memory_gib": run_["memory_gib"], "oom": run_["oom"],
                        "attention_ms": run_["attention_ms"], "dispatch_ms": run_["dispatch_ms"],
                        "moe_compute_ms": run_["moe_compute_ms"], "body_other_ms": run_["other_ms"],
                        "non_sol_ms": run_["non_sol_ms"],
                    }
                    out.append(row)
    return out


def broad_rows(specs: list[Any], refs: dict[int, str], dsa_models: dict, cache: dict[str, dict[str, Any]]):
    direct_balanced = {
        depth: min(
            (s for s in specs if s.kind == "direct" and s.active_tier == "balanced" and s.depth == depth),
            key=lambda s: abs(s.expert_intermediate - 3072),
        ) for depth in base.DEPTHS
    }
    out = []
    for spec in specs:
        for system in SYSTEMS:
            for dep in DEPS:
                for cpr in CPRS:
                    body = get(cache, spec.model_path, system, dep, cpr)
                    kr = get(cache, refs[spec.depth], system, dep, cpr)
                    dr = get(cache, direct_balanced[spec.depth].model_path, system, dep, cpr)
                    dsa_runs = {k: get(cache, model, system, dep, cpr) for k, model in dsa_models.items()}
                    native_attn = u.context_v4_attention(body["latency"]) if spec.kind == "direct" else u.context_kimi_attention(body["latency"])
                    body_ms = body["latency_ms"] - native_attn
                    kda, mla, attn_res = u.context_kimi_primitives(kr["latency"], spec.depth)
                    shared, full = dsa_runs[(2048, True)], dsa_runs[(2048, False)]
                    frac = 21 / 78
                    norm = full["latency"].get("context_add_norm_1", 0.0) / 78
                    full_2048 = full["latency"].get("context_attention", 0.0) / 78
                    shared_2048 = shared["latency"].get("context_attention", 0.0) / 78
                    skip_2048 = (shared_2048 - frac * full_2048) / (1-frac)
                    dsa = {}
                    for topk in (512, 1024, 2048, 4096):
                        full_ratio = u.dsa_context_formula(system, CHUNK, cpr, topk, False) / u.dsa_context_formula(system, CHUNK, cpr, 2048, False)
                        skip_ratio = u.dsa_context_formula(system, CHUNK, cpr, topk, True) / u.dsa_context_formula(system, CHUNK, cpr, 2048, True)
                        fv, sv = full_2048 * full_ratio, skip_2048 * skip_ratio
                        dsa[(topk, True)] = norm + frac*fv + (1-frac)*sv
                        if topk == 2048:
                            dsa[(topk, False)] = norm + fv
                    native_w, _ = base.body_native_weight_cache(spec, PREFIX+CHUNK)
                    body_weight = body["weights_gib"] * GIB - native_w
                    overhead = body["activations_gib"] + body["nccl_gib"] + body["others_gib"]
                    dispatch, moe = body["dispatch_ms"], body["moe_compute_ms"]
                    other = max(0.0, body_ms-dispatch-moe)
                    for pid, profile in base.ATTENTION_PROFILES.items():
                        counts = base.allocate_counts(profile["parts"], spec.depth)
                        attention = counts.get("kda",0)*(kda+attn_res) + counts.get("mla",0)*(mla+attn_res)
                        attention += counts.get("v4",0)*u.context_v4_attention(dr["latency"])/spec.depth
                        if counts.get("dsa"):
                            attention += counts["dsa"] * dsa[(int(profile.get("dsa_topk",2048)), bool(profile.get("dsa_shared_index",True)))]
                        attn_w, cache_req = u.broad_weight_cache(profile, counts, spec.depth, PREFIX+CHUNK)
                        memory = (body_weight+attn_w)/GIB + cpr*cache_req/GIB + overhead
                        latency = body_ms+attention
                        total, active = base.estimated_params(spec, attn_w)
                        out.append({
                            "architecture_id": f"factorial__{spec.id}__{pid}",
                            "label": f"{spec.id} · {profile['label']}",
                            "family": "Controlled 5T factorial", "study_kind": "AIC component composite",
                            "attention": profile["label"], "total_t": total/1e12, "active_b": active/1e9,
                            "depth": spec.depth, "experts": spec.experts, "topk": spec.topk,
                            "system": system, "dep": dep, "cpr": cpr, **metrics(latency, dep, cpr),
                            "memory_gib": memory, "oom": memory > CAPACITY,
                            "attention_ms": attention, "dispatch_ms": dispatch,
                            "moe_compute_ms": moe, "body_other_ms": other,
                            "non_sol_ms": body["non_sol_ms"] + kr["non_sol_ms"] + dr["non_sol_ms"] + sum(x["non_sol_ms"] for x in dsa_runs.values()),
                        })
    return out


def gqa_attention(latency: dict[str, float]) -> float:
    return sum(v for k,v in latency.items() if k == "context_attention" or k.endswith("_attn_norm") or k.endswith("_qkv_gemm") or k.endswith("_proj_gemm"))


def swa_rows(paths: dict[str,str], v4_model: str, kimi_model: str, dsa_model: str, cache: dict[str,dict[str,Any]]):
    out=[]
    for system in SYSTEMS:
        for dep in DEPS:
            for cpr in CPRS:
                refs={rid:get(cache,path,system,dep,cpr) for rid,path in paths.items()}
                common=refs["common_global"]
                body=sum(common["latency"].values())-gqa_attention(common["latency"])
                gqa={rid:gqa_attention(x["latency"])/swa.LAYERS for rid,x in refs.items()}
                vr=get(cache,v4_model,system,dep,cpr); kr=get(cache,kimi_model,system,dep,cpr); dr=get(cache,dsa_model,system,dep,cpr)
                kda,mla,res=u.context_kimi_primitives(kr["latency"],swa.LAYERS)
                prim={"v4":u.context_v4_attention(vr["latency"])/swa.LAYERS,"kda":kda+res,"mla":mla+res,"dsa":(dr["latency"].get("context_attention",0)+dr["latency"].get("context_add_norm_1",0))/78}
                dispatch=common["latency"].get("context_global_moe_pre_dispatch",0)+common["latency"].get("context_global_moe_post_dispatch",0)
                moe=common["latency"].get("context_global_moe",0); other=max(0.0,body-dispatch-moe)
                body_weight=common["weights_gib"]-swa.LAYERS*swa.gqa_weight_bytes(swa.REFS["common_global"])/GIB
                overhead=common["activations_gib"]+common["nccl_gib"]+common["others_gib"]
                for profile in swa.PROFILES:
                    attention=profile["local"]*gqa[profile["local_ref"]] if profile.get("local_ref") else 0
                    mech=profile["global_mechanism"]
                    attention += profile["global"]*(gqa[profile["global_ref"]] if mech=="gqa" else prim[mech])
                    aw,cache_req=swa.profile_weight_cache(profile,PREFIX+CHUNK)
                    memory=body_weight+aw/GIB+cpr*cache_req/GIB+overhead
                    latency=body+attention
                    out.append({
                        "architecture_id":"swa__"+profile["id"],"label":profile["label"],
                        "family":"SWA and hybrid extensions","study_kind":"AIC component composite",
                        "attention":profile["label"],"total_t":5.0,"active_b":125.9,"depth":61,
                        "experts":1216,"topk":25,"system":system,"dep":dep,"cpr":cpr,
                        **metrics(latency,dep,cpr),"memory_gib":memory,"oom":memory>CAPACITY,
                        "attention_ms":attention,"dispatch_ms":dispatch,"moe_compute_ms":moe,
                        "body_other_ms":other,"non_sol_ms":sum(x["non_sol_ms"] for x in list(refs.values())+[vr,kr,dr]),
                    })
    return out


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields=[]
    for row in rows:
        for k in row:
            if k not in fields: fields.append(k)
    with path.open("w",encoding="utf-8",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(rows)


def summarize(rows: list[dict[str,Any]]):
    out=[]
    for aid in sorted({r["architecture_id"] for r in rows}):
        sample=next(r for r in rows if r["architecture_id"]==aid)
        s={k:sample.get(k) for k in ("architecture_id","label","family","study_kind","attention","total_t","active_b","depth","experts","topk")}
        for system,prefix in (("GB300","gb"),("VR200 HBM+NVLink","rubin")):
            sr=[r for r in rows if r["architecture_id"]==aid and r["system"]==system]
            c1=[r for r in sr if r["cpr"]==1 and not r["oom"]]
            s[f"{prefix}_min_fitting_dep"]=min((r["dep"] for r in c1),default=0)
            best=min(c1,key=lambda r:r["latency_ms"],default=None)
            s[f"{prefix}_best_dep_cpr1"]=best["dep"] if best else 0
            s[f"{prefix}_best_latency_cpr1_ms"]=best["latency_ms"] if best else None
            fixed=[r for r in sr if r["dep"]*r["cpr"]==64 and not r["oom"]]
            best64=max(fixed,key=lambda r:r["group_input_tps"],default=None)
            s[f"{prefix}_best_dep_global64"]=best64["dep"] if best64 else 0
            s[f"{prefix}_best_cpr_global64"]=best64["cpr"] if best64 else 0
            s[f"{prefix}_best_latency_global64_ms"]=best64["latency_ms"] if best64 else None
            for dep in DEPS:
                r=next(x for x in sr if x["dep"]==dep and x["cpr"]==1)
                s[f"{prefix}_dep{dep}_c1_latency_ms"]=r["latency_ms"]
                s[f"{prefix}_dep{dep}_c1_memory_gib"]=r["memory_gib"]
                s[f"{prefix}_dep{dep}_c1_oom"]=r["oom"]
                s[f"{prefix}_dep{dep}_c1_dispatch_ms"]=r["dispatch_ms"]
                s[f"{prefix}_dep{dep}_c1_moe_ms"]=r["moe_compute_ms"]
        out.append(s)
    return out


def main():
    specs=base.body_specs();refs=base.make_models(specs);anchors=base.make_native_anchors();dsa_model=base.make_dsa_validation_model();dsa_models=u.make_dsa_models(dsa_model)
    exact=u.exact_models(anchors["glm52_ratio"]["model_path"])
    paths,v4_model,kimi_model=swa.make_models()
    models=[m["model"] for m in exact]+[s.model_path for s in specs]+list(refs.values())+list(dsa_models.values())+list(paths.values())+[v4_model,kimi_model,dsa_model]
    cache=run(models)
    rows=direct_rows(exact,cache)+broad_rows(specs,refs,dsa_models,cache)+swa_rows(paths,v4_model,kimi_model,dsa_model,cache)
    summary=summarize(rows)
    write_csv(OUTPUT/"dep_sweep_results.csv",rows);write_csv(OUTPUT/"dep_sweep_summary.csv",summary)
    meta={"workload":{"prefix":PREFIX,"new_tokens":CHUNK},"deps":DEPS,"cprs":CPRS,"systems":SYSTEMS,"architectures":len(summary),"rows":len(rows),"comparison_views":["c/r=1 with 64-GPU normalized replication","single-group fixed global concurrency=64"]}
    (OUTPUT/"dep_sweep_metadata.json").write_text(json.dumps(meta,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(meta,indent=2))


if __name__=="__main__": main()
