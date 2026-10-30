"""Separate Qwen full-GQA scaling study; never mutates the frozen index or AISim.

Run with the repository's pinned .venv. Output partitions are resumable only
when their configuration signature matches. No model weights are downloaded.
"""
import argparse
import csv
import hashlib
import html
import importlib.metadata
import json
import math
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from agent_index import index, simulation
from agent_index.topology import penalty_ms

HERE = Path(__file__).resolve().parent
OUT = ROOT / "results/qwen-gqa"
CONTEXT = 262144
BASE = "Qwen/Qwen3-Coder-480B-A35B-Instruct"
# id, layers, hidden, expert intermediate, experts, top-k, Q heads, KV heads
DESIGNS = [
    ("Qwen-480B", 62, 6144, 2560, 160, 8, 96, 8),
    ("Qwen-like-1.5T", 80, 6144, 3072, 320, 16, 96, 8),
    ("Qwen-like-3T", 80, 9216, 4096, 320, 16, 144, 12),
    ("Qwen-like-5T-joint", 96, 12288, 4608, 320, 16, 192, 16),
    ("Qwen-like-5T-deeper", 144, 9216, 4096, 320, 16, 144, 12),
    ("Qwen-like-5T-wide-experts", 72, 12288, 6144, 320, 16, 192, 16),
    ("Qwen-like-5T-many-experts", 62, 6144, 2560, 1760, 88, 96, 8),
    ("Qwen-like-5T-deep-aligned", 144, 8192, 4608, 320, 16, 128, 16),
    ("Qwen-like-5T-wide-trunk", 72, 16384, 4608, 320, 16, 256, 16),
]
PRIMARY = ["GB300 NVL72", "VR200", "B300 · 800G", "MI455X", "MI355X · 400G"]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def csvsave(path, rows):
    if not rows:
        return
    with path.open("w", newline="") as stream:
        w = csv.DictWriter(stream, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        w.writeheader()
        w.writerows(rows)


def prepare():
    simulation._sdk()  # Pin version and force offline model config loading.
    import aisimulate_core
    basefile = Path(aisimulate_core.__file__).parent / "model_configs" / (BASE.replace("/", "--") + "_config.json")
    baseline = json.loads(basefile.read_text())
    models = []
    for name, layers, hidden, intermediate, experts, topk, heads, kv in DESIGNS:
        c = dict(baseline, num_hidden_layers=layers, max_window_layers=layers,
                 hidden_size=hidden, moe_intermediate_size=intermediate,
                 num_experts=experts, num_experts_per_tok=topk,
                 num_attention_heads=heads, num_key_value_heads=kv)
        assert topk / experts == .05
        path = OUT / "models" / name / "config.json"
        if not path.exists() or json.loads(path.read_text()) != c:
            dump(path, c)
        routed = layers * 3 * hidden * intermediate * experts
        attention = layers * (2 * hidden * heads * 128 + 2 * hidden * kv * 128)
        other = 2 * c["vocab_size"] * hidden + layers * (hidden * experts + 2 * hidden + 256)
        models.append(dict(model=name, layers=layers, hidden=hidden, expert_width=intermediate,
            experts=experts, topk=topk, q_heads=heads, kv_heads=kv, head_dim=128,
            total_params_T=(routed + attention + other) / 1e12,
            active_params_B=(routed * .05 + attention + other) / 1e9,
            routed_params_T=routed / 1e12, selection_fraction=.05,
            fp8_kv_GiB_per_request=2 * layers * kv * 128 * CONTEXT / 2**30,
            config_sha256=digest(path), config_path=str(path.parent)))
    assert json.loads((OUT / "models/Qwen-480B/config.json").read_text()) == baseline
    csvsave(OUT / "architectures.csv", [{k:v for k,v in m.items() if k!='config_path'} for m in models])
    return models, basefile


def layouts(hardware, model):
    # Wider TP/DCP than the original pilot. DCP only partitions replicated KV
    # heads; do not grant DCP arbitrary cache sharding at TP <= KV heads.
    for stage in (1, 2, 4, 8, 16, 32, 64, 72, 128):
        if stage > hardware["chip_budget"] or (hardware["restricted"] and stage > 8):
            continue
        for tp in (1, 2, 4, 8, 16, 32, 64):
            if stage % tp or model["q_heads"] % tp or model["hidden"] % tp:
                continue
            kv = model["kv_heads"]
            if (tp <= kv and kv % tp) or (tp > kv and tp % kv):
                continue
            for dcp in (1, 2, 4, 8):
                if dcp > max(1, tp // kv) or tp % dcp:
                    continue
                for etp in (1, 2, 4, 8):
                    if stage % etp or model["experts"] % (stage // etp) or model["expert_width"] % etp:
                        continue
                    pps = (1, 2, 3, 4, 6, 8) if hardware["restricted"] and stage == 8 else (1,)
                    for pp in pps:
                        if stage * pp <= hardware["chip_budget"]:
                            yield dict(tp=tp, dp=stage // tp, dcp=dcp, etp=etp, ep=stage // etp, pp=pp)


def native(hardware, model, layout, nn):
    perf, _, build, apply, get_model = simulation._sdk()
    p = layout
    c = dict(model=model["config_path"], system=hardware["profile"], backend="vllm",
        backend_version="0.24.0", worker_type="decode", tp=p["tp"], pp=p["pp"],
        attention_dp=p["dp"], dcp=p["dcp"], cp_size=1, moe_tp_size=p["etp"],
        moe_ep_size=p["ep"], nextn=nn, enable_shared_layer=False,
        gemm_quant_mode="bfloat16", moe_quant_mode=hardware["quant"],
        fmha_quant_mode="bfloat16", kvcache_quant_mode="fp8", comm_quant_mode="half",
        estimation_mode="op_level", database_mode="SOL", fallback_policy="deny",
        systems_paths=(str(ROOT / "configs/hardware"),),
        estimator_config={"correction": {"enabled": False}})
    mc = build(tp_size=p["tp"], pp_size=p["pp"], attention_dp_size=p["dp"],
        moe_tp_size=p["etp"], moe_ep_size=p["ep"], cp_size=1, dcp_size=p["dcp"],
        gemm_quant_mode="bfloat16", moe_quant_mode=hardware["quant"],
        fmha_quant_mode="bfloat16", kvcache_quant_mode="fp8", comm_quant_mode="half")
    apply(mc, nn)
    g = get_model(c["model"], mc, "vllm")
    assert (g._num_layers, g._num_experts, g._topk) == (model["layers"], model["experts"], model["topk"])
    expected = model["fp8_kv_GiB_per_request"] * 2**30 / min(p["tp"], model["kv_heads"]) / p["dcp"]
    assert math.isclose(g.get_kvcache_rank_bytes_per_sequence(CONTEXT), expected, rel_tol=1e-12)
    return c, g, perf.best_available(c)


def memory(c, g, batch, nn):
    m = simulation._sdk()[1](c["model"], c["system"], c["backend"], c["backend_version"],
        max_num_tokens=batch * (nn + 1), max_batch_size=batch,
        memory_fraction_kind="of_total", memory_fraction_value=.9,
        tp_size=c["tp"], pp_size=c["pp"], attention_dp_size=c["attention_dp"],
        moe_tp_size=c["moe_tp_size"], moe_ep_size=c["moe_ep_size"], cp_size=1,
        dcp_size=c["dcp"], nextn=nn, gemm_quant_mode=c["gemm_quant_mode"],
        moe_quant_mode=c["moe_quant_mode"], kvcache_quant_mode="fp8",
        fmha_quant_mode="bfloat16", comm_quant_mode="half",
        systems_path=c["systems_paths"][0], allow_naive_fallback=False, worker_type="decode")
    breakdown = m["memory_breakdown"]
    weights = breakdown["weights_bytes"]
    bound = simulation._weight_bound(g, c["pp"]) if c["pp"] > 1 else weights
    cache = batch * g.get_kvcache_rank_bytes_per_sequence(CONTEXT + nn + 1)
    stage_guard = max(0., bound - weights)
    boundary = 2 * batch * (nn + 1) * g._hidden_size * 2 if c["pp"] > 1 else 0.
    metadata = (bound + cache) / 32
    extra = cache + stage_guard + boundary + metadata
    return dict(fits=extra <= m["total_kv_size_bytes"],
        memory_gib=(sum(breakdown.values()) + extra) / 2**30,
        weights_gib=weights / 2**30, cache_gib=cache / 2**30,
        stage_weight_guard_gib=stage_guard / 2**30, metadata_guard_gib=metadata / 2**30,
        boundary_buffer_gib=boundary / 2**30)


def cap(c, g, nn):
    lo, hi = 0, 1
    while hi < 65536 and memory(c, g, hi, nn)["fits"]:
        lo, hi = hi, hi * 2
    assert hi < 65536, "Search guard reached"
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        if memory(c, g, mid, nn)["fits"]:
            lo = mid
        else:
            hi = mid
    return lo


def collect(args, models):
    catalog = index.load_catalog(ROOT)
    for hardware_key, hardware in catalog.items():
        if args.hardware and hardware_key not in args.hardware:
            continue
        for model in models:
            if args.model and model["model"] not in args.model:
                continue
            for nn in args.nextn:
                filename = f'{hardware["profile"]}__{model["model"]}__nn{nn}'
                target = OUT / "parts" / filename
                json_path = Path(str(target) + ".json")
                csv_path = Path(str(target) + ".csv")
                signature = dict(hardware=hardware, model=model, nextn=nn,
                    engine=simulation.PINNED_AISIM, context=CONTEXT,
                    hardware_sha=digest(ROOT / "configs/hardware" / (hardware["profile"] + ".yaml")),
                    adapter_version=1)
                if json_path.exists():
                    old = json.loads(json_path.read_text())
                    assert old["signature"] == signature
                    print("resume", filename, flush=True)
                    continue
                rows, exclusions, checks = [], [], []
                for p in layouts(hardware, model):
                    c, g, m = native(hardware, model, p, nn)
                    try:
                        limit = cap(c, g, nn)
                    except ValueError as exc:
                        if "no KV budget:" not in str(exc):
                            raise
                        exclusions.append(dict(**p, reason=str(exc)))
                        continue
                    if not limit:
                        exclusions.append(dict(**p, reason="No admitted request"))
                        continue
                    specs = [json.loads(op._spec_json()) for op in g.generation_ops]
                    hop = penalty_ms(specs) if hardware.get("hop_overlay") else 0.
                    prev = 0.
                    for b in range(1, limit + 1):
                        ops = m.static_phase_diagnostics(batch_size=b, context_length=CONTEXT, prefix=0, prefill=False)
                        native_ms = sum(o["latency_ms"] for o in ops)
                        ms = native_ms + hop
                        assert ms > 0 and ms >= prev - 1e-9
                        prev = ms
                        mem = memory(c, g, b, nn)
                        assert mem["fits"]
                        # Exclusive top-level buckets; do not double-count nested overlap branches.
                        buckets = dict(attention_ms=0., moe_ms=0., communication_ms=hop, other_ms=0.)
                        for o in ops:
                            name = o["name"].lower()
                            bucket = "attention_ms" if "generation_attention" in name else "communication_ms" if any(x in name for x in ("dispatch", "allreduce", "all_reduce", "p2p", "dcp", "comm")) else "moe_ms" if "moe" in name else "other_ms"
                            buckets[bucket] += o["latency_ms"]
                        assert math.isclose(sum(buckets.values()), ms, rel_tol=1e-10)
                        al = 3 if nn else 1
                        chips = p["tp"] * p["dp"] * p["pp"]
                        requests = b * p["dp"]
                        row = dict(hardware=hardware_key, model=model["model"], nextn=nn, al=al,
                            **p, batch_per_replica=b, chips=chips, requests=requests,
                            capacity=requests / chips, speed=1000 * al / ms,
                            tpot_ms=ms / al, raw_ms=ms, native_ms=native_ms, hop_ms=hop,
                            tokens_s_chip=requests / chips * 1000 * al / ms,
                            max_batch=limit, quant=hardware["quant"], **mem, **buckets)
                        rows.append(row)
                    assert not memory(c, g, limit + 1, nn)["fits"]
                    checks.append(dict(**p, max_batch=limit, monotonic=True, next_batch_rejected=True))
                target.parent.mkdir(parents=True, exist_ok=True)
                csvsave(csv_path, rows)
                dump(json_path, dict(signature=signature, rows=len(rows),
                    checks=checks, exclusions=exclusions))
                print("complete", filename, "rows", len(rows), "layouts", len(checks), flush=True)


def loadrows():
    rows = []
    for path in sorted((OUT / "parts").glob("*.csv")):
        if not path.with_suffix('.json').exists():
            continue  # Ignore an in-progress partition.
        for r in index.read_csv(path):
            for k, v in list(r.items()):
                if k not in ("hardware", "model", "quant", "fits"):
                    r[k] = float(v)
            rows.append(r)
    return rows


def table(rows):
    if not rows:
        return "<p>No feasible configuration.</p>"
    keys = list(rows[0])
    def cell(x):
        return html.escape(f"{x:.3f}" if isinstance(x, float) else str(x))
    return '<div class="scroll"><table><tr>' + ''.join('<th>'+cell(k)+'</th>' for k in keys) + '</tr>' + ''.join('<tr>'+''.join('<td>'+cell(r.get(k, ''))+'</td>' for k in keys)+'</tr>' for r in rows) + '</table></div>'


def analyze(models):
    os.environ["MPLCONFIGDIR"] = str(OUT / "mpl")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    rows = loadrows()
    model_map = {m['model']:m for m in models}
    for r in rows:
        design = model_map[r['model']]
        assignments = int(r['batch_per_replica'] * r['dp'] * (r['nextn'] + 1) * design['topk'])
        r['routed_assignments_per_ep_rank'] = assignments / r['ep']
        # Native moe_sol uses min(E/EP, floor(tokens*topk/EP)). Screen only
        # unsaturated fractional expert counts; do not alter native timing.
        r['unsaturated_fractional_expert_count'] = assignments < design['experts'] and assignments % int(r['ep']) != 0
    groups = {}
    partitions = [json.loads(p.read_text()) for p in (OUT / "parts").glob("*.json")]
    catalog = index.load_catalog(ROOT)
    for part in partitions:
        signature = part["signature"]
        hw = next(k for k,v in catalog.items() if v == signature["hardware"])
        groups[(hw, signature["model"]["model"], signature["nextn"])] = []
    for row in rows:
        groups.setdefault((row["hardware"], row["model"], int(row["nextn"])), []).append(row)
    curves = {k: index.envelope(v) for k, v in groups.items()}
    screened = {k:index.envelope([r for r in v if not r['unsaturated_fractional_expert_count']]) for k,v in groups.items()}
    summary, winners, frontier = [], [], []
    for (hw, model, nn), curve in curves.items():
        ref = curves.get(("GB300 NVL72", model, nn), [])
        denom = index.area(ref)
        summary.append(dict(hardware=hw, model=model, nextn=nn,
            memory_max_agents_chip=max((r["capacity"] for r in groups[hw, model, nn]), default=0.),
            fastest_TPS=max((r["speed"] for r in curve), default=0.),
            **{f"agents_chip_at_{s}":index.capacity(curve, s) for s in (20, 50, 100, 200, 400)},
            log50_400_area=index.area(curve), model_GB300e=index.area(curve)/denom if denom else None,
            **{f'screened_agents_chip_at_{s}':index.capacity(screened[hw,model,nn],s) for s in (50,100,200,400)},
            screened_area_fraction=index.area(screened[hw,model,nn])/index.area(curve) if index.area(curve) else None))
        frontier.extend(curve)
        for s in (0, 20, 50, 100, 200, 400):
            eligible = [r for r in curve if r["speed"] >= s]
            if eligible:
                r = max(eligible, key=lambda r:(r["capacity"], r["speed"]))
                winners.append(dict(target_TPS=s, **r))
            else:
                winners.append(dict(target_TPS=s, hardware=hw, model=model, nextn=nn, capacity=0., feasible=False))
    csvsave(OUT / "summary.csv", summary)
    csvsave(OUT / "winning_configs.csv", winners)
    csvsave(OUT / "frontier_nodes.csv", frontier)
    csvsave(OUT / 'screened_frontier_nodes.csv', [r for c in screened.values() for r in c])
    csvsave(OUT / "all_candidates.csv", rows)
    routed = [m['routed_params_T'] for m in models if m['model'].startswith('Qwen-like-5T-') and 'many-experts' not in m['model']]
    assert max(routed) == min(routed)
    assert all(math.isclose(r['capacity'],r['requests']/r['chips']) and
               math.isclose(r['speed'],1000*r['al']/r['raw_ms']) and
               math.isclose(r['raw_ms'],sum(r[k] for k in ('attention_ms','moe_ms','communication_ms','other_ms')))
               for r in rows)
    dump(OUT / 'checks.json', dict(partitions_completed=len(partitions),
        expected_full_partitions=len(catalog)*len(models)*2,
        completed_admitted_layouts=sum(len(p['checks']) for p in partitions),
        excluded_layouts=sum(len(p['exclusions']) for p in partitions),
        admitted_rows=len(rows), chip_accounting=True, timing_bucket_sum=True,
        matched_5T_routed_parameter_counts=True, native_cache_formula_checked=True,
        each_admitted_layout_next_batch_rejected=True,
        quality_or_empirical_validation=False))
    snapshots = []
    for row in winners:
        if row['hardware'] not in ('GB300 NVL72','VR200') or row['model'] not in ('Qwen-480B','Qwen-like-5T-joint') or row['target_TPS'] not in (0,100,200) or not row.get('feasible',True):
            continue
        p = {k:int(row[k]) for k in ('tp','dp','dcp','etp','ep','pp')}
        c,g,m = native(catalog[row['hardware']],model_map[row['model']],p,int(row['nextn']))
        ops = m.static_phase_diagnostics(batch_size=int(row['batch_per_replica']),context_length=CONTEXT,prefix=0,prefill=False)
        assert math.isclose(sum(o['latency_ms'] for o in ops),row['native_ms'],rel_tol=1e-12)
        snapshots.append(dict(winner=row,operations=ops,
            operation_specs=[json.loads(o._spec_json()) for o in g.generation_ops]))
    dump(OUT/'operation_audits.json',snapshots)
    # Frozen GLM/Kimi are contextual baselines, not a fair attention-only ablation.
    frozen = index.saved_candidates(ROOT)
    def plotcurve(ax, curve, label, **kwargs):
        xs = np.unique([20., 400., *[p["speed"] for p in curve if 20 < p["speed"] < 400]])
        ax.step(xs, [index.capacity(curve, x) for x in xs], where="pre", label=label, **kwargs)
    def format_ax(ax):
        ax.set_xscale("log"); ax.set_xlim(20, 400); ax.set_ylim(bottom=0)
        ax.set_xticks([20,50,100,200,400], labels=[20,50,100,200,400])
        ax.minorticks_off()
        ax.set_xlabel("Minimum SOL TPS/user"); ax.set_ylabel("Concurrent requests / allocated chip")
        ax.grid(alpha=.2); ax.legend(fontsize=8)
    for nn in (0, 3):
        fig, axes = plt.subplots(1, 2, figsize=(14,5))
        for ax, hw in zip(axes, ("GB300 NVL72", "VR200")):
            for m in models:
                c = curves.get((hw, m["model"], nn), [])
                if c: plotcurve(ax, c, m["model"])
            format_ax(ax); ax.set_title(hw)
        fig.suptitle("256Ki full-GQA scaling · " + ("no speculation" if not nn else "NN3/AL3 pessimistic proxy"))
        fig.tight_layout(); fig.savefig(OUT / f"scaling_nn{nn}.svg")
        fig.savefig(OUT / f"scaling_nn{nn}.png", dpi=130); plt.close(fig)
        fig, axes = plt.subplots(1, 2, figsize=(14,5))
        for ax, model in zip(axes, ("Qwen-480B", "Qwen-like-5T-joint")):
            for hw in PRIMARY:
                c = curves.get((hw, model, nn), [])
                if c: plotcurve(ax, c, hw)
            format_ax(ax); ax.set_title(model)
        fig.suptitle("Hardware capacity frontiers · " + ("no speculation" if not nn else "NN3/AL3 proxy"))
        fig.tight_layout(); fig.savefig(OUT / f"hardware_nn{nn}.svg")
        fig.savefig(OUT / f"hardware_nn{nn}.png", dpi=130); plt.close(fig)
    for filename, selection in (('context',('Qwen-480B','Qwen-like-5T-joint')),
                               ('matched_depth_width',('Qwen-like-5T-deep-aligned','Qwen-like-5T-joint','Qwen-like-5T-wide-trunk'))):
        fig, axes = plt.subplots(1, 2, figsize=(14,5))
        for ax, hw in zip(axes,('GB300 NVL72','VR200')):
            for model in selection:
                c = curves.get((hw,model,3),[])
                if c: plotcurve(ax,c,model)
            if filename == 'context':
                for model in index.MODELS:
                    c = index.envelope([p for p in frozen if p['hardware']==hw and p['model']==model])
                    plotcurve(ax,c,model+' (frozen)',linestyle='--',linewidth=2)
            format_ax(ax); ax.set_title(hw)
        fig.suptitle('NN3/AL3 proxy · '+('real-model context, not an attention-only ablation' if filename=='context' else 'equal routed parameters;16 KV heads; depth versus trunk width'))
        fig.tight_layout(); fig.savefig(OUT/f'{filename}.svg')
        fig.savefig(OUT/f'{filename}.png',dpi=130); plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(14,9))
    for ax, (hw, model) in zip(axes.flat, [(h,m) for h in ('GB300 NVL72','VR200') for m in ('Qwen-480B','Qwen-like-5T-joint')]):
        for nn, color in ((0,'#1976a3'),(3,'#ad5c24')):
            key = hw, model, nn
            if key in curves:
                label = 'No spec' if nn == 0 else 'NN3/AL3 proxy'
                plotcurve(ax, curves[key], label+' native', color=color)
                plotcurve(ax, screened[key], label+' screened', color=color, linestyle='--')
        format_ax(ax); ax.set_title(hw+' · '+model)
    fig.suptitle('Expert-rank integer-rounding sensitivity; no timing correction')
    fig.tight_layout(); fig.savefig(OUT/'expert_rounding_sensitivity.svg'); plt.close(fig)
    # Per-operation curves at a fixed layout: no changes of parallelism hidden inside a line.
    fig, axes = plt.subplots(2, 2, figsize=(14,9))
    for ax, (hw, model) in zip(axes.flat, [(h,m) for h in ("GB300 NVL72","VR200") for m in ("Qwen-480B","Qwen-like-5T-joint")]):
        sub = [r for r in rows if r["hardware"]==hw and r["model"]==model and r["nextn"]==0 and r["chips"]==64 and r["tp"]==8 and r["dcp"]==1 and r["etp"]==2 and r["pp"]==1]
        sub.sort(key=lambda r:r["capacity"])
        for col in ("raw_ms", "attention_ms", "moe_ms", "communication_ms", "other_ms"):
            ax.plot([r["capacity"] for r in sub], [r[col] for r in sub], label=col.replace("_ms",""))
        ax.set_title(hw+" · "+model); ax.set_xlabel("Concurrency / allocated chip")
        ax.set_ylabel("No-spec TPOT (ms)"); ax.grid(alpha=.2); ax.legend(fontsize=8)
    fig.suptitle("Fixed TP8 DP8 ETP2 EP32 PP1 — ends at admitted memory capacity")
    fig.tight_layout(); fig.savefig(OUT / "components.svg"); plt.close(fig)
    arch = [{k:r[k] for k in ("model","total_params_T","active_params_B","layers","hidden","expert_width","experts","topk","q_heads","kv_heads","fp8_kv_GiB_per_request")} for r in models]
    body = '<!doctype html><meta charset="utf-8"><title>Qwen full-GQA scaling</title><style>body{font:16px system-ui;color:#192c40;max-width:1400px;margin:35px auto;padding:20px}img{width:100%}table{border-collapse:collapse;font-size:13px}th,td{padding:8px;border:1px solid #ccd}.scroll{overflow:auto}.warn{padding:18px;background:#fff1d5}h2{margin-top:40px}</style>'
    body += '<h1>Qwen3-Coder: full-GQA decode and hypothetical scaling</h1><p>Separate exploratory study; the published GLM/Kimi index is unchanged. Native pinned AISim SOL; 262,144 cached tokens; FP8 KV; BF16 attention and non-expert GEMMs; hardware-adapted expert precision. No achieved-performance discount.</p>'
    body += '<div class="warn"><b>Speculation limitation:</b> NN3/AL3 retains native generic MTP overhead and pessimistic repeated-prefix reads. It is not a trained Qwen draft model or a reuse-aware verification SOL. No-spec results are the cleaner architecture comparison. Neither mode predicts model quality, routing balance in practice, or vendor software efficiency.<br><b>Expert rounding:</b> native SOL can charge zero expert-weight reads when fewer than one routed assignment reaches the average rank. The separate screened frontier excludes unsaturated fractional-count configurations; it does not edit AISim or fix real routing imbalance.</div>'
    at100 = []
    for m in ('Qwen-480B','Qwen-like-1.5T','Qwen-like-3T','Qwen-like-5T-joint','Qwen-like-5T-deep-aligned','Qwen-like-5T-wide-trunk','Qwen-like-5T-many-experts'):
        z = dict(Model=m)
        for hw, short in (('GB300 NVL72','GB300'),('VR200','Rubin')):
            for nn in (0,3):
                s = next((r for r in summary if (r['hardware'],r['model'],r['nextn']) == (hw,m,nn)),None)
                if s:
                    z[short+(' no-spec' if nn==0 else ' NN3/AL3')] = s['agents_chip_at_100']
        at100.append(z)
    body += '<h2>At a glance: concurrency / chip at100 SOL TPS/user</h2>'+table(at100)
    body += '<p><b>Interpretation:</b> full-GQA cache reads grow with concurrency×context×layers×KV heads. The baseline carries31GiB of FP8 KV per request at256Ki; joint5T carries96GiB. Unlike sparse/linear attention, these layers must read the whole prefix on every ordinary decode step. More HBM bandwidth therefore continues to matter well before the memory-capacity ceiling is reached.</p><p>At equal routed parameter budgets, deeper-aligned5T has144GiB KV/request, joint5T96GiB and wide-trunk5T72GiB. Their concurrency differences show why a parameter count alone is not a performance specification. The many-expert-only~5T control keeps31GiB KV/request and is deliberately not our central scaling assumption: keeping depth and attention fixed makes it much easier to serve at long context. Conversely,8-chip systems face the large-model weight-capacity limit and our conservative multi-node PP schedule. Their low scores do not establish an impossibility result for optimized interleaved PP.</p><p>All figures are conditional projections. The native speculation proxy may be slower than no-speculation when full-cache reads dominate; do not interpret that as evidence that a real reuse-aware Qwen speculative implementation would slow down.</p>'
    body += '<h2>Architecture definitions</h2>'+table(arch)+'<p>All layers retain full GQA, RoPE and routed top-5% MoE; no shared experts, SWA or MLA. All ~5T variants except many-experts have exactly equal routed parameter budgets; total counts differ slightly in attention. Many-expert control is only approximately parameter-matched. Counts are analytical matrix inventories including two embedding/logit matrices, router and small normalization terms; active counts include embeddings by convention. No quality equivalence or training feasibility claim.</p><p><b>Sharding confound:</b> the3T and original deeper5T variants have12 KV heads, permitting only TP1/2/4 in this integral, power-of-two TP search. Their losses cannot be attributed to depth alone. The deep-aligned/joint/wide-trunk trio fixes16 KV heads,320/top16 experts and expert width4608 while trading144×8192,96×12288,72×16384 layers×hidden. This keeps routed parameters exactly fixed and admits wider TP/DCP. Query-to-KV ratios vary8/12/16. The separate wide-expert control trades depth for expert intermediate width.</p>'
    body += '<h2>Scaling within the Qwen family</h2><img src="scaling_nn0.svg"><img src="scaling_nn3.svg"><h2>Controlled depth versus trunk width</h2><img src="matched_depth_width.svg"><h2>Existing-model context</h2><img src="context.svg"><p>Dashed GLM/Kimi curves reuse the frozen headline search and NN3/AL3 assumptions. Different sizes, attention, native speculative modeling and candidate families make these contextual comparisons, not isolated architecture effects. In particular, Qwen verification omits shared-prefix read reuse; these gaps must not be attributed solely to attention architecture.</p>'
    body += '<h2>Hardware comparison</h2><img src="hardware_nn0.svg"><img src="hardware_nn3.svg">'
    bottlenecks = []
    for a in snapshots:
        r = a['winner']
        if r['target_TPS'] == 100:
            attn = next(o for o in a['operations'] if o['name']=='generation_attention')
            moe = next(o for o in a['operations'] if o['name']=='generation_moe')
            bottlenecks.append(dict(hardware=r['hardware'],model=r['model'],nextn=r['nextn'],
                pure_attention_pct=100*attn['latency_ms']/r['raw_ms'],moe_pct=100*moe['latency_ms']/r['raw_ms'],
                attention_math_ms=attn['details']['sol']['math_ms'],attention_HBM_ms=attn['details']['sol']['memory_ms'],
                moe_math_ms=moe['details']['sol']['math_ms'],moe_HBM_ms=moe['details']['sol']['memory_ms']))
    csvsave(OUT/'bottlenecks.csv',bottlenecks)
    body += '<h2>What consumes time?</h2><img src="components.svg"><p>Exclusive native top-level timing buckets sum to the full iteration. Attention includes its DCP merge operations where present; communication contains MoE dispatch/P2P and explicitly named collectives; other includes projections, routing, logits, elementwise work and the small embedding all-reduce. Thus these buckets are not an exact all-communication decomposition. The fixed-layout graph uses DCP1. No-spec single-query attention avoids the verification-reuse confound.</p>'+table(bottlenecks)+'<p>The table isolates the pure attention and MoE operators, including their native math/HBM rooflines, at each100-TPS winner. Milliseconds are per native iteration before division by AL. In these cases memory time dominates arithmetic time; the5T no-spec case additionally spends much more of its budget reading expert weights.</p>'
    body += '<h2>Expert-count rounding sensitivity</h2><img src="expert_rounding_sensitivity.svg"><p>Native Rust moe_sol charges min(E/EP, floor(N×topk/EP)) expert matrices per rank, where N=batch×DP×(nextn+1). This is unique-until-saturation, not a probability-of-overlap calculation; the workload-distribution label does not enter this SOL formula. Before saturation, fractional rank counts can undercharge critical-rank weight reads. For example, joint5T,TP64/DP1/EP64,batch1,no-spec charges zero MoE memory time. The screened envelope excludes candidates with N×topk&lt;E and nonintegral N×topk/EP, keeping unchanged native timings otherwise. It is a restricted feasible-layout sensitivity, not a calibrated correction or real-runtime lower bound. The CSV reports its area retention and threshold capacities. Solid=native; dashed=screened. Overlap of lines means the screen did not change that section.</p>'
    body += '<h2>GB300 / Rubin thresholds</h2>'+table([r for r in summary if r['hardware'] in ('GB300 NVL72','VR200')])
    compact_winners = []
    for r in winners:
        if r['hardware'] in ('GB300 NVL72','VR200','B300 · 800G') and r['model'] in ('Qwen-480B','Qwen-like-5T-joint') and r['target_TPS'] in (50,100,200):
            z = {k:r[k] for k in ('hardware','model','nextn','target_TPS','capacity')}
            if r.get('feasible', True):
                z.update(layout=' '.join(f'{k.upper()}{int(r[k])}' for k in ('tp','dp','dcp','etp','ep','pp')),
                    requests=int(r['requests']), allocated_chips=int(r['chips']),
                    actual_TPS=r['speed'], memory_GiB_rank=r['memory_gib'])
            compact_winners.append(z)
    body += '<h2>Representative winning configurations</h2>'+table(compact_winners)+'<p>Concurrency / chip = requests divided by all allocated chips. Batch per attention replica is different: requests = batch × DP. DCP is contained within TP, not another multiplicative chip dimension; total chips = TP × DP × PP. These tables select maximum density first and fastest configuration on density ties.</p>'
    body += '<h2>All hardware: baseline and joint ~5T</h2>'+table([r for r in summary if r['model'] in ('Qwen-480B','Qwen-like-5T-joint')])
    body += '<h2>Search and limitations</h2><p>Every admitted integer batch in the finite layout family is timed; capacity at each speed is independently maximized over saved candidates. Denominator counts only allocated chips including every PP stage. Search: 1/2/4/8/16/32/64/72/128 chips, bounded by the frozen hardware budget; TP through64; ETP through8; DCP divides only actual replicated KV heads. Query heads, KV heads, experts and dimensions must partition integrally. Restricted 8-chip systems use local TP/EP or 2/3/4/6/8-stage serial PP, not cross-node EP. Unrestricted systems use PP1. No arbitrary expert placement or fractional experts. Integer divisibility therefore changes usable layouts between designs.</p><p>Admission retains the published 90% HBM,4GiB runtime reservation, native workspace and 1/32 weight+KV guard. PP adds unequal-layer weight and boundary-buffer guards, retaining the conservative whole-model KV reservation on each stage. Serial PP has no interleaving or throughput multiplier. This can strongly penalize PP-only deployments and is not a proof of their best attainable performance. All hardware resources/topology/precision are frozen from the packaged release, including uncertain TPU/Jalapeño/Huawei assumptions; no new vendor specifications are asserted. No SRAM residency model, measured scheduling, prefill, queueing, duty cycles or calibrated congestion. Native ideal balance and unique-until-saturation expert reads are retained, with the separate rounding screen above.</p><p>50–400 index values integrate the exact staircase against d log(TPS), normalized to GB300 for the same model and speculation case. They are not combined into the published index. Zero beyond the fastest saved feasible configuration, no extrapolation; zero does not mean the model cannot run below the speed band. Display sampling does not affect scores. Hypothetical models reuse the256Ki boundary as a workload assumption, not a trained context guarantee.</p>'
    body += '<h2>Data and reproducibility</h2><p>'+ ' · '.join(f'<a href="{f}">{f}</a>' for f in ('architectures.csv','summary.csv','winning_configs.csv','frontier_nodes.csv','screened_frontier_nodes.csv','all_candidates.csv','provenance.json','checks.json','operation_audits.json','bottlenecks.csv')) +'</p>'
    body += '<p>Implementation: studies/qwen_gqa/run_study.py; shared src/agent_index/index.py (exact envelope/integration), simulation.py (pinned SDK and stage-weight guard), topology.py (frozen TPU hop overlay). AISim inspected source commit5c466b567d0ef62d983958ddfcbccbad0e201a29: sdk/models/moe.py (GQA/MoE graph), sdk/models/base.py (KV/DCP rules), crates/core/src/perfmodel/operators/moe.rs:940 (SOL expert integer arithmetic). The inspected source commit is not asserted to be the wheel build commit. Some communication operators retain a native source tag of silicon despite analytical resource-profile execution; tags alone do not establish measured calibration. No empirical kernel database or correction factor is requested by this study.</p>'
    body += f'<p>{len(rows):,} admitted model/hardware/layout/batch/mode candidates; {len(groups)} completed model/hardware/mode partitions. Empty partitions and excluded layouts are recorded in parts/*.json. Representative winners have full native operation/specification snapshots in operation_audits.json; all saved candidates retain mutually exclusive timing buckets.</p>'
    (OUT / "REPORT.html").write_text(body)
    print("REPORT", OUT / "REPORT.html", "candidates", len(rows), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--hardware", action="append")
    p.add_argument("--model", action="append")
    p.add_argument("--nextn", type=int, nargs="+", default=[0,3], choices=[0,3])
    p.add_argument("--analyze-only", action="store_true")
    p.add_argument("--collect-only", action="store_true")
    args = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    models, basefile = prepare()
    if not args.analyze_only:
        collect(args, models)
    dump(OUT / "provenance.json", dict(engine=simulation.PINNED_AISIM,
        source_inspected_commit="5c466b567d0ef62d983958ddfcbccbad0e201a29",
        wheel_build_commit_unknown=True, script_sha256=digest(__file__),
        baseline_config_sha256=digest(basefile), context=CONTEXT,
        model_config_hashes={m['model']:m['config_sha256'] for m in models},
        native_wheel_provenance="../../environment/runtime.json",
        runtime_provenance_sha256=digest(ROOT / "environment/runtime.json"),
        shared_wrapper_hashes={name:digest(ROOT/'src/agent_index'/name) for name in ('simulation.py','index.py','topology.py')},
        hardware_catalog_sha256=digest(ROOT / "configs/catalog.json"),
        hardware_hashes={p.name:digest(p) for p in (ROOT / 'configs/hardware').glob('*.yaml')},
        argv=sys.argv, frozen_index_unchanged=True))
    if not args.collect_only:
        analyze(models)
