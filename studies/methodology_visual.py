"""Render the current methodology from saved results only; no simulator calls."""
import csv
import hashlib
import html
import json
import math
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from agent_index.index import capacity
from log400_release import combined_area, segments

OUT = ROOT / 'results/methodology-current'
REFERENCE = ROOT / 'reference/2026-10-15'
STUDY = ROOT / 'results/workload-capacity-oct15-log400'
INPUT = STUDY / 'curves_and_inputs.json'
COLORS = {'GB300 NVL72': '#2563a6', 'VR200': '#b94b20'}
LOW, HIGH = 50, 400
LABELS = {'GLM-5.2':'GLM-5.2', 'Kimi-K3':'Kimi-K3',
          'MAI2-informed-h128-w512-sh4352':'MAI-2 2.8T', 'Qwen-480B':'Qwen3-Coder 480B'}


def add_sidenotes(page):
    # Attach explanations to existing passages; these never alter score inputs.
    notes = [
        ('We study long-context, warm-cache inference:', 'Why 256Ki context?', 'Trace-informed study choice',
         'Jason’s earlier workload-distribution analysis motivated roughly 256K as a representative long-context agent workload, conditional on the harness and compaction policy. It is not a midpoint between product context limits. We fix 262,144 tokens to keep the comparison controlled; performance at this context need not equal average performance over a distribution.',
         'Background: <a href="https://epoch.ai/publications/estimating-the-agent-population">Epoch AI agent-capacity research</a> and its <a href="https://github.com/epoch-research/compute-to-agents">associated code and data</a>. The exact 256Ki choice is the author’s modeling judgment informed by that work, not a pooled mean recomputed here. The later warm-cache cohort is selected around this context; it does not independently prove the choice.'),
        ('GLM-5.2, Kimi-K3, Qwen3-Coder 480B, and <strong>MAI-2 2.8T</strong>', 'Model selection and MAI-2 assumptions', 'Architecture coverage; disclosed and assumed dimensions',
         'GLM supplies sparse attention; Kimi supplies a large MLA-based model; Qwen supplies full GQA; MAI-2 2.8T supplies sliding-window/global attention and latent MoE. Equal weights broaden architectural coverage; they do not imply known traffic shares, equal quality or complete frontier coverage.<br><br>The MAI-2 reconstruction follows Microsoft’s disclosed 102 layers, hidden width 8,704, 85 local and 17 global attention layers, 16 local and 2 global KV heads, 96 query heads, top-7 of 512 routed experts plus one shared expert, routed latent dimension 4,096 and expert intermediate width 4,352.<br><br>It is an informed proxy, not an exact production implementation. Head size 128, window 512 and vocabulary 200,019 are inherited from the predecessor; shared intermediate width 4,352 is assumed. This gives approximately 2.84T total and 86B active parameters, versus the reported 2.8T/80B. Serving quantization, AL3, draft implementation and quality at 256Ki are assumptions. Latent projections are conservatively replicated under TP.',
         'Sources: <a href="https://microsoft.ai/pdf/Frontier-Pre-training-at-Microsoft-AI.pdf?v=2">MAI-Base-2 disclosure</a> and <a href="https://microsoft.ai/wp-content/uploads/2026/06/main_20260602_2.pdf">MAI-Base-1 report</a>.'),
        ('<a href="https://github.com/ai-dynamo/aisimulate">NVIDIA AISimulate</a> is an analytical', 'Why analytical SOL?', 'Comparison framework',
         'Hardware-specification-based operation costs let us compare platforms without requiring equally optimized serving code on every device. This is more structured than a single aggregate FLOP or bandwidth ratio, but not a cycle-accurate chip simulator. An omitted optimization or incorrect traffic assumption can change the modeled ceiling.',
         'We apply no achieved-performance multiplier. Earlier benchmark grounding motivates efficiency scenarios, not a universal guarantee that every chip achieves at least 50% of SOL.'),
        ('The decode study uses <strong>nextn3 / AL3</strong>', 'Why AL3 with nextn3?', 'Measured origin; shared assumption',
         'InferenceX’s Kimi-K3 SPEED-Bench coding reference reports AL = 3.00 for three speculative tokens, measured with vLLM DSpark. That motivates this operating point. Applying the same accepted length to GLM, Qwen and MAI-2 is a controlled assumption, not a measurement of those models or every workload.',
         '<a href="https://github.com/SemiAnalysisAI/InferenceX/blob/main/golden_al_distribution/kimik3_dspark_probabilistic_sample_method_block_rejection_sample_method.yaml">Acceptance reference</a>. Four target verification positions do not imply a 75% independent acceptance probability per drafted token. Draft cost still follows each native AISim graph, not one shared measured DSpark implementation.'),
        ('AL3 is a shared assumption, not measured acceptance', 'Why these precisions?', 'Hardware-adapted modeling policy',
         'Low-bit expert weights reflect the intended co-optimized inference scenario. FP8 is retained where our hardware profiles do not support FP4. FP4-capable profiles use the native NVFP4 analytical representation even when the hardware advertises another format; this is not bit-exact format equivalence.',
         'BF16 non-expert GEMMs and FP8 KV are fixed study settings, with the native Kimi FP8 decode-MLA exception. We do not establish equal model accuracy, format-specific overhead or equivalent kernel efficiency across vendors.'),
        ('For each hardware/model pair, we evaluate supported parallelism', 'Why this search and memory budget?', 'Finite search and admission assumptions',
         'We search supported layouts and admitted integer batches rather than prescribing one rack-wide configuration. Only allocated chips count. Integral expert placement and platform scale-up limits constrain the candidates; node-restricted systems use serial pipeline stages across nodes.',
         'The configured memory fraction is 90%, with native workspace, a 4GiB fixed reservation and additional guards. For Kimi’s SGLang path, this fraction governs the static pool after pre-load overhead; displayed totals also include transient activations and can exceed 90%. The other plotted models use a total-memory budget. These are study conventions, not vendor-calibrated usable-memory limits.'),
        ('Independently optimize prefill for <strong>maximum', 'Why 1Ki new input and throughput?', 'Representative warm-input proxy',
         '1,024 new tokens over the existing context represents a short warm-cache input phase and reuses our aligned prefill sweeps. The selected trace cohort averages about 1,423 new tokens; 1,024 is a convenient study point, not that exact mean.',
         'We optimize throughput because this term budgets the accelerator resources needed to supply input work. It does not impose a time-to-first-token SLA. Using the 1Ki per-token rate for the whole cohort approximates its input-length distribution; larger inputs can have different efficiency.'),
        ('The selected coding-trace cohort supplies the workload ratio:', 'Why 1.513 input/output?', 'Saved trace measurement',
         'The warm-cache cohort has 110,565 requests, mean new input 1,422.854 and mean output 940.206 tokens. Dividing aggregate new input by aggregate output gives 1.5133425. The saved query selects Claude-provider requests with positive output, ≥90% cached input, 128–384Ki cached tokens and 1–10Ki new tokens.',
         'This conditions the ratio on the study’s warm long-context workload. It excludes cold misses and is not a universal agent traffic ratio. We apply it to every model without adjusting tokenization or verbosity. Exact query and artifact hashes are in the companion data and the trace-grounding expander.'),
        ('A stronger prefill chip needs a smaller', 'Why add chip budgets?', 'Steady-state resource accounting',
         'One continuously decoding request at target speed s needs r × s new input tokens processed per second on average. Dividing that demand by the best prefill rate gives prefill chips per request; adding decode chips per request yields total resource demand.',
         'This automatically changes the phase allocation with each chip’s strengths. It assumes same-chip-type, fully utilized pools, fleet-scale fractional allocation and resident caches. Transfers, scheduling and extra cache-placement overhead are not included.'),
        ('Plot minimum output speed on the horizontal axis', 'Why log 50–400 AUC?', 'Equal weight per speed doubling',
         'A proportional speed improvement has the same horizontal span anywhere in the band: 50–100, 100–200 and 200–400 each receive one third of the integration weight. This balances slower throughput-oriented work with faster interactive work. A range avoids hinging the index on one target near a configuration switch. Equal log-speed weight is a benchmark convention, not a measured demand distribution; actual area also depends on supported capacity.',
         'For illustration only, 50% of SOL speed at fixed concurrency would turn this into 25–200 achieved TPS/user. Neither that efficiency nor the endpoint choices are universal. If a deployment has a known speed SLA, its pointwise capacity is more directly relevant.'),
        ('Each model contributes one GB300-relative', 'Why GB300 and equal model weights?', 'Normalization and basket convention',
         'GB300 is the study’s reference platform. Dividing each model’s area by its own GB300 area removes raw request-capacity units before combining models; a small model cannot dominate merely because it fits more requests. Each architecture then receives one quarter of the arithmetic average.',
         'Equal weights are an explicit architectural-coverage choice, not estimated traffic shares. The score remains sensitive to the basket and is neither quality-adjusted nor a cost/power-normalized hardware exchange rate. Frozen versions and visible component scores make that dependence auditable.'),
    ]
    for n, (anchor, title, basis, rationale, source) in enumerate(notes, 1):
        assert page.count(anchor) == 1, anchor
        pos = page.index(anchor)
        start = page.rfind('<p', 0, pos)
        end = page.index('</p>', pos)
        paragraph = page[start:end]
        ref = f'<sup><a class="note-ref" href="#note-{n}" aria-label="Note {n}: {html.escape(title)}" onclick="document.getElementById(\'note-{n}\').open=true">[{n}]</a></sup>'
        note = f'<aside class="sidenote"><details id="note-{n}"><summary>{n} · {title}</summary><p class="note-basis">{basis}</p><p>{rationale}</p>' + (f'<p>{source}</p>' if source else '') + '</details></aside>'
        page = page[:start] + note + paragraph + ref + page[end:]
    return page


def adjusted(c, speed, prefill, ratio):
    return c/(1+ratio*speed*c/prefill) if c and prefill > 0 else 0.


def chart(curves, model, num, prefills, ratio):
    w, h, left, right, top, bottom = 960, 440, 76, 28, 24, 70
    maximum = math.ceil(max(capacity(curves[hw][model], LOW) for hw in COLORS)/5)*5
    x = lambda speed: left + math.log(speed/LOW)/math.log(HIGH/LOW)*(w-left-right)
    y = lambda cap: h-bottom-cap/maximum*(h-top-bottom)
    out = [f'<svg class="frontier" id="chart-{num}" viewBox="0 0 {w} {h}" role="img" aria-labelledby="title-{num} desc-{num}">',
           f'<title id="title-{num}">{html.escape(LABELS[model])}: GB300 and VR200 capacity frontiers</title>',
           f'<desc id="desc-{num}">Minimum SOL tokens per second per user on a logarithmic horizontal axis from 50 to 400. Equal spacing for 50, 100, 200 and 400. Vertical axis is concurrent decoding requests per total prefill and decode chip. Shading illustrates log-speed capacity area; numeric areas are integrated analytically. Modeled, not measured.</desc>']
    for tick in range(0, maximum+1, 5):
        out.append(f'<path d="M{left},{y(tick):.3f} H{w-right}" class="grid"/><text x="{left-12}" y="{y(tick)+5:.3f}" text-anchor="end">{tick}</text>')
    for tick in (50, 100, 200, 400):
        out.append(f'<path d="M{x(tick):.3f},{top} V{h-bottom}" class="grid"/><text x="{x(tick):.3f}" y="{h-bottom+28}" text-anchor="middle">{tick}</text>')
    for hw, color in COLORS.items():
        segs = segments(curves[hw][model])
        # Sample only for drawing; duplicate boundaries preserve layout jumps.
        coords = [(x(s), y(adjusted(c,s,prefills[hw,model],ratio)))
                  for a,b,c in segs
                  for s in [a+(b-a)*i/max(1,math.ceil(b-a)) for i in range(max(1,math.ceil(b-a))+1)]]
        path = 'M' + ' L'.join(f'{a:.6f},{b:.6f}' for a, b in coords)
        fill = path + f' L{x(HIGH):.6f},{y(0):.6f} L{x(LOW):.6f},{y(0):.6f} Z'
        out.append(f'<path d="{fill}" fill="{color}" fill-opacity=".12"/>')
        out.append(f'<path d="{path}" stroke="{color}" stroke-width="3" fill="none"/>')
    out += [f'<path id="cursor-{num}" d="M0,24 V370" stroke="#647e91" stroke-dasharray="5 5"/>',
            *[f'<circle id="dot-{num}-{i}" r="5" fill="{color}" stroke="white" stroke-width="2"/>' for i, color in enumerate(COLORS.values())],
            f'<text x="{(left+w-right)/2}" y="{h-12}" text-anchor="middle">Minimum SOL TPS / user (log scale) →</text>',
            f'<text transform="translate(19,{(top+h-bottom)/2}) rotate(-90)" text-anchor="middle">Decode concurrency / total serving chip</text>', '</svg>']
    return ''.join(out), maximum


def main():
    saved = json.loads(INPUT.read_text())
    curves, ratio = saved['curves'], saved['ratio']
    visual = json.loads((REFERENCE/'visual_inputs.json').read_text())
    memory_profiles = visual['memory_profiles']
    memory_sources = visual['memory_component_sources']
    models = saved['models']
    for hw, by_model in visual['enriched_curves'].items():
        for model, points in by_model.items():
            lookup = {tuple(p[k] for k in ('tp','dp','dcp','etp','ep','pp','batch')):p for p in points}
            for point in curves[hw][model]:
                key = tuple(point[k] for k in ('tp','dp','dcp','etp','ep','pp','batch'))
                if key in lookup:
                    for k in ('weights_gib','cache_gib','other_memory_gib','memory'):
                        point[k] = lookup[key][k]
    prefills = {(r['hardware'],r['model']):r['tokens_chip_second'] for r in saved['prefill_rates']}
    workload = json.loads((STUDY/'workload.json').read_text())
    scores = list(csv.DictReader((STUDY/'index.csv').open()))
    per_model = {(r['hardware'],r['model']):r for r in csv.DictReader((STUDY/'per_model_scores.csv').open()) if r['scenario']=='observed_ratio'}
    prefill_winners = visual['prefill_winners']
    prefill_rows = ''.join(f"<tr><th>{r['model']} / {r['hardware'].replace(' NVL72','')}</th>"
                          f"<td>{float(r['input_tokens_per_chip_second']):,.1f}</td>"
                          f"<td>{r['requests']} / {r['chips']}</td>"
                          f"<td>{float(r['forward_ms']):,.2f}</td>"
                          f"<td><code>{r['layout']}</code></td></tr>" for r in prefill_winners)
    examples, panels = [], []
    for n, model in enumerate(models):
        a, b = (combined_area(curves[hw][model],prefills[hw,model],ratio) for hw in COLORS)
        for hw, computed in zip(COLORS,(a,b)):
            assert math.isclose(computed,float(per_model[hw,model]['combined_auc']),rel_tol=1e-12)
        svg, ymax = chart(curves, model, n, prefills, ratio)
        example = dict(model=model,label=LABELS[model],GB300_area=a,VR200_area=b,combined_ratio=b/a,ymax=ymax,
                       prefill_rates={hw:prefills[hw,model] for hw in COLORS},
                       curves={hw:curves[hw][model] for hw in COLORS})
        examples.append(example)
        panels.append(f'''<article class="model" id="model-{n}" role="tabpanel" aria-labelledby="tab-{n}" tabindex="0"{'' if n == 1 else ' hidden'}>
<h3>{LABELS[model]}: VR200 has {b/a:.2f}× the combined capacity-area</h3>
<div class="legend"><span class="gb">━ GB300</span><span class="vr">━ VR200</span><span>Shading = area under each curve</span></div>{svg}
<p class="areas"><span class="vr">VR200 area <strong>{b:,.3f}</strong></span> ÷ <span class="gb">GB300 area <strong>{a:,.3f}</strong></span> = <strong>{b/a:.3f}×</strong></p></article>''')
    vr = next(r for r in scores if r['hardware']=='VR200')
    combined = sum(e['combined_ratio'] for e in examples)/4
    assert math.isclose(combined,float(vr['agent_index_GB300e']),rel_tol=1e-12)
    data = dict(updated='2026-10-15',low=LOW,high=HIGH,weighting='log',context=262144,
                nextn=3,accepted_length=3,ratio=ratio,model_weights=[.25]*4,
                models=examples,memory_profiles=memory_profiles,memory_component_sources=memory_sources,decode_score=float(vr['decode_only_GB300e']),combined_score=combined,
                workload=workload,scores=scores,
                prefill_winners=prefill_winners,
                source_commit='2f0f4cd8606d580567b253d02f3d699fe36db865',
                source_sha256={str(f.relative_to(ROOT)) if f.is_relative_to(ROOT) else f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in (INPUT,STUDY/'workload.json',STUDY/'index.csv',STUDY/'per_model_scores.csv',REFERENCE/'visual_inputs.json')},
                simulation_rerun=False)
    page = TEMPLATE.replace('%%PANELS%%',''.join(panels)).replace('%%DATA%%',json.dumps(data).replace('</','<\\/'))
    page = page.replace('%%PREFILL_ROWS%%', prefill_rows)
    page = page.replace('%%TABS%%',''.join(f'<button id="tab-{n}" role="tab" aria-selected="{str(n==1).lower()}" aria-controls="model-{n}" tabindex="{0 if n==1 else -1}" onclick="showModel({n})">{LABELS[m]}</button>' for n,m in enumerate(models)))
    page = page.replace('%%AVERAGE%%','('+' + '.join(f"{e['combined_ratio']:.3f}" for e in examples)+f') ÷ 4 = <strong>{combined:.3f} GB300e</strong>')
    page = page.replace('%%SCORE_ROWS%%',''.join(f"<tr><th>{html.escape(r['hardware'])}</th><td>{float(r['agent_index_GB300e']):.3f}</td>"+''.join(f"<td>{float(r[k]):.3f}</td>" for k in ('GLM_GB300e','Kimi_GB300e','Qwen3_GB300e','MAI_proxy_GB300e'))+'</tr>' for r in scores))
    page = page.replace('%%VR_GLM%%',f"{examples[0]['combined_ratio']:.2f}")
    page = page.replace('%%VR_KIMI%%',f"{examples[1]['combined_ratio']:.2f}")
    page = add_sidenotes(page)
    assert '%%' not in page and '/home/' not in page
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/'REPORT.html').write_text(page)
    (OUT/'methodology_current_data.json').write_text(json.dumps(data,indent=2)+'\n')
    print(OUT/'REPORT.html')



TEMPLATE = r'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>How the agent capacity index works</title>
<style>
*{box-sizing:border-box}body{margin:0;background:#fff;color:#20313d;font:16px/1.5 system-ui,sans-serif}main{max-width:1000px;margin:auto;padding:32px 24px}h1{font-size:34px;letter-spacing:-.035em;line-height:1.15;margin:0 0 12px}h2{font-size:21px;margin:0 0 8px}h3{font-size:19px;margin:0}.lead{max-width:840px;font-size:18px;margin:0 0 18px}.context{font-size:13px;color:#5e6d78;margin:0 0 24px}.explanation{display:grid;grid-template-columns:1fr 1fr;gap:30px;margin:22px 0}.explanation p{margin:0;font-size:15px}.chart{border:1px solid #dce3e8;border-radius:10px;padding:20px}.tabs{display:flex;gap:8px;margin-bottom:18px}button{font:inherit;border:1px solid #cad5dd;background:white;border-radius:6px;padding:7px 16px;color:#384f60;cursor:pointer}button[aria-selected="true"]{background:#eaf0f5;border-color:#647e91;font-weight:650}button:focus-visible{outline:3px solid #2563a6;outline-offset:2px}.legend{display:flex;flex-wrap:wrap;gap:12px 24px;font-size:14px;margin:10px 0}.gb{color:#2563a6}.vr{color:#b94b20}.frontier{width:100%;height:auto;display:block}.frontier text{font:15px system-ui;fill:#526675}.grid{stroke:#dbe3e8;stroke-width:1}.areas{text-align:center;font-size:18px;margin:8px 0 0;font-variant-numeric:tabular-nums}.caption{font-size:13px;color:#5e6d78;margin:12px 0 0}.scores{display:grid;grid-template-columns:1fr 1fr;gap:24px;margin:24px 0}.score{background:#f3f6f8;border-radius:8px;padding:18px}.score p{margin:8px 0 0}.formula{font-size:21px;font-variant-numeric:tabular-nums}.small{font-size:13px;color:#526675}footer{border-top:1px solid #dce3e8;padding-top:14px;font-size:12px;color:#5e6d78}a{color:#2563a6}.model[hidden]{display:none}@media(max-width:650px){main{padding:24px 16px}h1{font-size:29px}.explanation,.scores{grid-template-columns:1fr;gap:16px}.chart{padding:12px}.formula{font-size:18px}.areas{font-size:16px}.legend{gap:8px 16px}}@media print{main{padding:0;max-width:none}.tabs{display:none}.model[hidden]{display:block}.model{break-inside:avoid;margin-bottom:20px}.chart{border:0;padding:0}.scores{break-inside:avoid}}
.controls{display:flex;align-items:center;flex-wrap:wrap;gap:10px;margin:18px 0 12px}.controls input{flex:1;min-width:140px;accent-color:#2563a6}.controls button{padding:4px 10px;font-size:13px}.configs{display:grid;grid-template-columns:1fr 1fr;gap:14px}.config{border-left:3px solid #2563a6;background:#f3f6f8;padding:12px;font-size:14px}.config:last-child{border-color:#b94b20}.config p{margin:5px 0}.config strong{font-size:19px}.config code,td code{font-size:12px;overflow-wrap:anywhere}details{margin:16px 0}summary{cursor:pointer;color:#2563a6;font-weight:600}details p{font-size:14px}table{border-collapse:collapse;width:100%;font-size:13px;margin:12px 0}th,td{text-align:left;border-bottom:1px solid #dce3e8;padding:8px;vertical-align:top}th{font-weight:600}.scroll{overflow-x:auto}.math{padding:10px 12px;background:#f3f6f8;font-variant-numeric:tabular-nums}.prefill-detail{border-top:1px solid #dce3e8;padding-top:16px}.number{font-variant-numeric:tabular-nums}@media(max-width:650px){.configs{grid-template-columns:1fr}.controls label{width:100%}}@media print{.controls{display:none}}
.layer-figure{margin:20px 0;padding:18px;border:1px solid #dce3e8;border-radius:10px;background:#fafcfd}.layer-figure h3{font-size:17px}.layer-flow{display:flex;align-items:stretch;gap:8px;margin:16px 0}.layer-node{flex:1;min-width:0;border-top:3px solid #2563a6;background:#eaf1f8;padding:12px;font-size:13px}.layer-node b{display:block;font-size:14px;margin-bottom:6px}.layer-node.link{border-color:#8a6a3c;background:#f6f0e6}.layer-arrow{align-self:center;color:#617889;font-size:22px}.roofline-grid{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-top:12px}.resource{padding:12px;background:#eaf1f8;border-radius:6px;font-size:14px}.resource.memory{background:#f9eee6}.resource b{display:block}.roofline-merge{text-align:center;padding:8px;font-size:14px}.layer-total{padding:12px;background:#e7f1eb;border-left:3px solid #438261;font-size:14px}.layer-figure figcaption{font-size:12px;color:#526675;margin-top:12px}@media(max-width:650px){.layer-flow{flex-direction:column}.layer-arrow{transform:rotate(90deg);line-height:1}.roofline-grid{grid-template-columns:1fr}.layer-figure{padding:12px}}@media print{.layer-figure{break-inside:avoid}}
.tabs{flex-wrap:wrap}
.steps-nav{display:flex;flex-wrap:wrap;gap:8px 18px;margin:24px 0 32px;font-size:14px}.steps-nav a{text-decoration:none}.step{border-top:1px solid #dce3e8;padding-top:24px;margin-top:30px;scroll-margin-top:20px}.eyebrow{font-size:12px;letter-spacing:.1em;font-weight:650;color:#526675;margin:0 0 8px}.step>h2{font-size:25px;line-height:1.25}.step>p{max-width:920px}.step>details{background:#f8fafb;border:1px solid #e3e9ed;border-radius:7px;padding:12px 16px}.step>details p:last-child{margin-bottom:0}.step .chart{margin-top:20px}.step .score{margin:18px 0}.step .explanation h3{margin-bottom:7px}@media(max-width:650px){.step>h2{font-size:22px}.steps-nav{gap:8px 14px}.step>details{padding:10px 12px}}@media print{.steps-nav{display:none}.step{break-inside:auto}}
.companion-link{display:block;border:2px solid #2563a6;border-radius:10px;background:#eef5fc;padding:18px 22px;margin:20px 0 24px;text-decoration:none}.companion-link strong{display:block;font-size:20px}.companion-link span{display:block;margin-top:5px;color:#384f60;font-size:15px}.companion-link:hover{background:#e1eefb}.companion-link:focus-visible{outline:3px solid #b94b20;outline-offset:3px}
main{max-width:1300px;padding-right:300px}.sidenote{float:right;clear:right;width:238px;margin:0 -274px 16px 0;font-size:13px}.sidenote details{margin:0;border-left:2px solid #c5d5e3;padding:8px 12px;background:#f8fafc}.sidenote summary{font-size:13px;line-height:1.45}.sidenote details p{font-size:13px;line-height:1.5;overflow-wrap:anywhere}.sidenote .note-basis{font-size:11px;text-transform:uppercase;letter-spacing:.04em;color:#657888;margin:12px 0 8px}.note-ref{font-size:11px;text-decoration:none;padding:3px}.step{display:flow-root}.sidenote details:target{border-color:#2563a6;background:#eef5fc}.note-ref:focus-visible{outline:2px solid #2563a6}.tabs button{max-width:100%}@media(max-width:1100px){main{max-width:1000px;padding-right:24px}.sidenote{float:none;clear:none;width:auto;margin:12px 0}.sidenote details{padding:9px 13px}.sidenote details p{font-size:14px}}@media(max-width:650px){main{padding-right:16px}}@media print{main{padding:0}.sidenote{float:none;width:auto;margin:8px 0}.sidenote details{border-left:2px solid #aaa}}
</style></head><body><main>
<h1>How the agent capacity index works</h1>
<p class="lead">Start with an agentic workload. Estimate the hardware resources needed to serve it. Then compare how many concurrent decoding requests each chip can support across a useful range of speeds.</p>
<p class="context">15 October 2026 · log 50–400 SOL TPS/user · modeled <a href="#sol-definition">speed of light (SOL)</a> · GB300 = 1</p>
<p><a href="https://docs.google.com/spreadsheets/d/1WMmHnxZ18077l4YzXxlhiwX2dRS3B4dZTqjiX1CMz84/edit" target="_blank" rel="noopener noreferrer"><strong>Full results spreadsheet ↗</strong></a> · Accelerator scores, hardware specifications and methodology notes.</p>
<p class="small">Numbered notes explain the evidence and judgment behind each choice. Expand them in the side margin, or inline on smaller screens.</p>
<a class="companion-link" href="PERFORMANCE_KNEES.html"><strong>Explore the knee points: why performance curves bend →</strong><span>A separate interactive guide to concurrency, token speed, expert-weight reuse, bandwidth and memory capacity.</span></a>
<a class="companion-link" href="ACCELERATOR_COMPARISON.html"><strong>Compare all accelerators: ranked score charts →</strong><span>The current combined index, plus a rated-power comparison for selected vendors.</span></a>
<nav class="steps-nav" aria-label="Explanation steps"><a href="#workload">1 · Workload</a><a href="#simulate">2 · Simulate</a><a href="#decode">3 · Decode</a><a href="#prefill">4 · Prefill</a><a href="#compare">5 · Compare curves</a><a href="#combine">6 · Combine models</a></nav>

<section class="step" id="workload" aria-labelledby="workload-title">
<p class="eyebrow">STEP 1 · DEFINE THE COMPARISON</p>
<h2 id="workload-title">Choose a workload and an architecture basket</h2>
<p>We study long-context, warm-cache inference: each request has an existing <strong>262,144-token context (256Ki tokens)</strong>, with the cache required by its architecture already resident. Sliding-window layers retain only their window, while other layers follow their own cache rules. Decode generates output from this state; prefill processes new input using the existing prefix. Keeping this workload fixed lets us compare hardware under the same conditions.</p>
<div class="explanation"><div><h3>One active workload</h3><p>At each target output speed, count <strong>concurrent decoding requests per total serving chip</strong>. The denominator will include the prefill chips needed to supply those requests’ input—not just the chips executing decode.</p></div><div><h3>Four equally weighted architectures</h3><p>GLM-5.2, Kimi-K3, Qwen3-Coder 480B, and <strong>MAI-2 2.8T</strong> provide sparse attention, MLA, full GQA and sliding-window/global-hybrid cases.</p></div></div>
<details><summary>What is counted as an active agent</summary>
<p><strong>The numerator is concurrent decoding requests.</strong> The denominator includes both the decode chips and the prefill chips needed to supply their new input. The plotted label is therefore “decode concurrency per total serving chip.” It does not count extra agents currently prefilling, waiting or using tools. The worked example in step 5 shows how to count both pools.</p>
</details>
</section>

<section class="step" id="simulate" aria-labelledby="simulate-title">
<p class="eyebrow">STEP 2 · ESTIMATE A FORWARD PASS</p>
<h2 id="simulate-title">Use AISim to translate the workload into execution time</h2>
<p><a href="https://github.com/ai-dynamo/aisimulate">NVIDIA AISimulate</a> is an analytical inference-performance model. We give it the architecture, hardware specifications, batch and parallelism. It estimates operation-level compute, memory movement and communication costs without running the model on the target accelerator.</p>
<div class="math" id="sol-definition"><strong>What does “speed of light” (SOL) mean here?</strong><p>It is a hardware-specification-based estimate of how fast the <em>modeled work</em> could run using peak compute and data-transfer rates. For a roofline-modeled operation, we calculate both the time needed to do the arithmetic and the time needed to move its data; the larger time sets its limit. We then account for required communication and the sequence of operations.</p><p><strong>100% of modeled SOL means matching that predicted speed—not keeping every resource at 100% utilization.</strong> An operation can use all its memory bandwidth while leaving much of its compute idle. The index reports this idealized performance, not measured production speed, and applies no blanket efficiency discount.</p></div>
<p class="math">Model + hardware + configuration → operation timings → full iteration time<br><strong>TPS/user = accepted output tokens per iteration ÷ iteration time</strong></p>
<p>The decode study uses <strong>nextn3 / AL3</strong>: four target verification positions and an assumed average of three accepted output tokens per iteration. Native verification and draft work are included.</p>
<p class="small">TPS/user is average output tokens per second for one decoding request. TPOT is its reciprocal, in seconds per output token. With speculative decoding these are averages: tokens can arrive in bursts, not at evenly spaced intervals.</p>
<details><summary>Inside AISim: a layer diagram, roofline equations and implementation details</summary>
<p><strong>What is AISim?</strong> <a href="https://github.com/ai-dynamo/aisimulate">AISimulate (AISim)</a> is NVIDIA’s open-source performance-modeling toolkit for LLM inference, and the successor to AIConfigurator. It predicts execution time from a model’s architecture, its deployment configuration and the hardware’s resources. We use its <strong>operation-level speed-of-light (SOL) mode</strong>: an analytical calculation that runs on a CPU, without needing to load the model’s weights onto the target accelerator.</p>
<p class="math">Model architecture + hardware specs + batch and parallelism<br>→ operation costs → decode-step time → TPS/user → capacity curve</p>
<figure class="layer-figure" aria-labelledby="layer-title">
<h3 id="layer-title">Zoom in: one routed-expert transformer layer</h3>
<p class="small">Follow one batch through the layer. Each box contains operations whose work depends on the model, batch and chosen sharding.</p>
<div class="layer-flow">
<div class="layer-node"><b>1 · Attention</b>Project the token states; access the KV cache according to the attention pattern; form the output.<br><br>Include attention-side collectives when required.</div>
<span class="layer-arrow" aria-hidden="true">→</span>
<div class="layer-node link"><b>2 · Route + dispatch</b>Select experts for each token and send token states to the chips holding those experts.</div>
<span class="layer-arrow" aria-hidden="true">→</span>
<div class="layer-node"><b>3 · Expert computation</b>Read the required expert weights and run the routed matrix multiplications.</div>
<span class="layer-arrow" aria-hidden="true">→</span>
<div class="layer-node link"><b>4 · Combine</b>Return and combine expert outputs; pass the updated token states to the next layer.</div>
</div>
<h3>Inside a roofline-modeled compute operation</h3>
<div class="roofline-grid">
<div class="resource"><b>Compute constraint</b>Required FLOPs ÷ peak FLOPs/s<br>At the operation’s modeled precision.</div>
<div class="resource memory"><b>HBM constraint</b>Modeled bytes moved ÷ HBM bytes/s<br>Weights, cache and intermediate data as applicable.</div>
</div>
<div class="roofline-merge">↘ &nbsp; <strong>Take the larger time</strong> &nbsp; ↙<br>These two constraints are not added together.</div>
<div class="layer-total"><strong>Assemble the layer time.</strong> Sum sequential operation costs and required communication costs. For explicitly modeled overlapping branches, use the longer branch instead of their sum.<br><br><strong>Then assemble the complete iteration:</strong> layer timings + other model operations + native draft work → iteration time ÷ AL3 → time per output token.</div>
<figcaption>Schematic, not a measured execution timeline; box sizes do not represent duration. The four architectures have different attention and draft graphs. Normalization, residual paths and model-specific branches are omitted from this drawing for readability, not intentionally removed from the simulator’s model graph. Communication has its own collective-specific cost rules; the two-resource maximum above is for compute operations, not a universal formula for every operator.</figcaption>
</figure>
<p><strong>1. Describe the work.</strong> AISim’s model definitions represent the four architectures as graphs of operations: attention projections, cache access and attention, routed expert computation, and communication between chips. Layer counts, matrix dimensions, expert counts and experts selected per token determine how much work is required. Each candidate specifies its batch and supported TP/DP/EP/ETP/DCP/PP layout, with <strong>262,144 cached tokens per request</strong>. MAI-2 uses the native STEP3P7 graph; its search does not support DCP.</p>
<p><strong>2. Price each operation against hardware resources.</strong> Hardware profiles supply compute rates at the relevant precision, HBM bandwidth and capacity, and interconnect bandwidth and scale-up sizes. For a roofline-modeled compute operation, the core calculation is:</p>
<p class="math">Compute time = required arithmetic operations ÷ peak operations/second<br>Memory time = modeled HBM bytes moved ÷ HBM bytes/second<br>Operation time ≈ <strong>max(compute time, memory time)</strong></p>
<p><strong>A simple example.</strong> Moving 80 decimal GB through 8 TB/s of HBM bandwidth takes 10 ms at peak rate. If that operation’s arithmetic takes 4 ms at peak compute, its roofline time is 10 ms, not 14 ms: memory is the limiting resource. Required non-overlapped communication and subsequent operations still add time. These are illustrative numbers, not a measured model run.</p>
<p>This asks which resource limits that operation. More bandwidth helps a weight- or cache-reading bottleneck; more FLOPs help a compute bottleneck. The byte counts depend on the architecture, precision and sharding. For MoE, the model distributes routed work across expert ranks and caps expert-weight reads at the resident expert inventory. Attention uses the model’s own attention pattern rather than treating every architecture as full attention.</p>
<p><strong>3. Include communication and combine timings.</strong> AISim charges the collectives required by the chosen layout, using their transfer volumes, participant counts and the hardware profile’s link rates. Our hardware policies determine which scale-up and scale-out paths a layout can use; TPU v8i also receives the study’s explicit hop-latency adjustment. The study sums the emitted operation timings across the model. Where AISim explicitly represents overlapping branches, it takes the longer branch duration instead of adding both. Consequently, the result is an operation-by-operation estimate, not simply total model FLOPs divided by chip FLOPs.</p>
<p><strong>4. Convert a speculative step into output speed.</strong> The current score uses <code>nextn=3</code>: four target verification positions, with an assumed <strong>average accepted length of 3 output tokens per iteration</strong>. AISim’s native model graph includes verification and its model-specific draft-work representation. We divide that complete modeled iteration time by the accepted length:</p>
<p class="math">TPOT = iteration time ÷ 3<br>TPS/user = 3 ÷ iteration time in seconds<br>Example: 23.097 ms per iteration → 7.699 ms/output token → <strong>129.885 TPS/user</strong></p>
<p class="small">AL3 is a shared assumption, not measured acceptance for every architecture. Native full-GQA and SWA verification can pessimistically repeat prefix reads. Precision is hardware-adapted FP4 experts via an NVFP4 analytical proxy, except H100, H200 and TPU v7 using FP8 experts. Non-expert GEMMs use BF16; the nominal attention setting is BF16, but the native Kimi decode-MLA path uses FP8. KV storage is FP8. These modeling choices do not establish equal accuracy across formats or models.</p>
<p><strong>5. Search for capacity at each speed.</strong> Our wrapper evaluates supported layouts and every admitted integer batch in the search. Admission accounts for model weights, KV cache, workspace and fixed reservations, using the study’s 90% configured fraction, backend-specific pool semantics and additional guards. Each candidate produces a speed and a density: concurrent requests divided by allocated chips. At every speed threshold, we select the highest density among candidates that meet it. Those selections form the decode step frontier; adding the prefill resource term gives the combined curve scored above. This is an optimum over saved candidates, not proof of the global best serving implementation.</p>
<p class="small"><strong>How to interpret SOL.</strong> The score uses peak-resource estimates with performance corrections disabled. It describes the performance implied by the modeled operation graph and hardware specifications; implementation efficiency, memory reuse and communication assumptions determine how closely a deployed system can approach it. The headline applies no utilization discount.</p>
<p class="small"><strong>Implementation reference.</strong> The current runs use AISim commit <code>2f0f4cd8606d580567b253d02f3d699fe36db865</code>. The engine is unchanged; the MAI-2 study adds a process-local architecture adapter composed of native operators, not an upstream-supported MAI implementation. <code>studies/mai2_swa.py</code> and <code>studies/mai2_prefill.py</code> supply its phase sweeps and heterogeneous-layer memory guards. <code>src/agent_index/index.py</code> constructs decode envelopes; <code>studies/log400_release.py</code> integrates combined capacity over log speed and assembles the four-model index. Finite searches, conservative serial pipeline scheduling, topology assumptions and ideal expert balancing limit the result; this is not an SGLang scheduling trace.</p>
</details>
<details><summary>What SOL establishes—and what needs benchmarking</summary>
<p>AISim makes a consistent operation-level comparison possible without access to every accelerator or equally optimized serving implementations. It estimates the constraints implied by the model graph and hardware specifications. That is useful for cross-hardware roofline analysis, while actual benchmarks are needed to establish achievable efficiency.</p>
<p><strong>Reading an efficiency percentage.</strong> If SOL predicts 200 TPS/user for a given configuration and concurrency, achieving 50% of that speed means 100 TPS/user and twice the modeled time per output token. It does not mean half the latency, or automatically half the maximum concurrency at a speed target. Real deployments can lose performance to kernel efficiency, scheduling, load imbalance and communication overhead; the achieved fraction can differ by hardware and workload.</p>
<p><strong>A ceiling conditional on the model.</strong> This is not a proven upper bound over every possible implementation. Fusion, better data reuse or a different communication algorithm can change the modeled work itself. Beating an imperfect SOL estimate would expose a modeling assumption, not exceed the chip’s physical limits. Likewise, overly optimistic traffic or topology assumptions can overstate the achievable speed.</p>
<p>It is not a cycle-accurate reproduction of each chip. Memory reuse, expert balance, communication, speculative verification and vendor-specific execution remain modeling assumptions. Benchmark grounding on some configurations does not guarantee the same achieved fraction on every platform. The score measures modeled hardware capacity for this workload—not model quality, API pricing or measured concurrent-user capacity.</p>
</details>
</section>

<section class="step" id="decode" aria-labelledby="decode-title">
<p class="eyebrow">STEP 3 · SEARCH SERVING CONFIGURATIONS</p>
<h2 id="decode-title">Find the most decode concurrency that meets each speed target</h2>
<p>For each hardware/model pair, we evaluate supported parallelism layouts and integer batches. We reject configurations that fail the memory-admission rules. At a target such as <strong>100 TPS/user</strong>, retain the candidates fast enough to meet it and choose the highest concurrent-request count per allocated chip.</p>
<p class="math">C<sub>decode</sub>(s) = max [concurrent requests ÷ allocated decode chips]<br>over saved memory-feasible candidates with modeled TPS/user ≥ s</p>
<p>Repeat that selection at every speed threshold. A higher requirement usually forces a smaller batch or a different layout. The deployment chip limit is a ceiling, not a requirement to occupy a whole rack: only allocated chips count.</p>
<details><summary>Memory admission, parallelism and the limits of this search</summary><p>Admission accounts for weights, KV cache, workspace, fixed reservations and guards using a 90% configured memory fraction, with Kimi’s static-pool exception described in step 5. Layouts use supported TP/DP/EP/ETP/DCP/PP combinations with integral expert placement. For node-restricted systems, cross-node deployment uses the study’s conservative serial pipeline schedule; MAI-2’s decode search uses DCP1.</p><p>This is the best result among saved candidates—not proof of the best possible implementation. Integer batches and configuration switches produce steps. A candidate exactly at the speed threshold remains eligible; if none qualify, capacity is zero. The explorer in step 5 shows the winning layouts and a worked explanation of their labels.</p></details>
<p class="small">To understand the shapes rather than the score calculation, see the separate <a href="PERFORMANCE_KNEES.html">visual guide to performance knees</a>.</p>
</section>

<section class="step" id="prefill" aria-labelledby="prefill-title">
<p class="eyebrow">STEP 4 · PROVIDE THE INPUT WORK TOO</p>
<h2 id="prefill-title">Add the prefill chips required by the same workload</h2>
<p>Independently optimize prefill for <strong>maximum new-input tokens per allocated chip-second</strong>: 262,144 cached tokens plus <strong>1,024 new tokens per request</strong>, without a latency target or speculative decoding. Call that rate <strong>P</strong>.</p>
<p>The selected coding-trace cohort supplies the workload ratio: <strong>1.513 new input tokens per output token</strong>. Faster output therefore requires proportionally more input-processing throughput. Each hardware/model pair supplies its own prefill rate; there is no fixed 15% weighting.</p>

<p>A stronger prefill chip needs a smaller input-processing pool; a stronger decode chip can change that balance in the other direction. We therefore add the <strong>resources needed for the same workload</strong>, rather than applying the same percentage blend to every chip.</p>
<p>At target output speed <strong>s</strong>, let <strong>C(s)</strong> be decode agents per decode chip, <strong>P</strong> be new input tokens per prefill chip-second, and <strong>r = 1.5133425</strong> be new input tokens per output token:</p>
<p class="math">Decode chips per agent = 1 / C(s)<br>Prefill chips per agent = r × s / P<br><strong>Combined agents per total chip = 1 / [1/C(s) + r × s/P]</strong><br>Prefill chip share = (r × s/P) / [1/C(s) + r × s/P]</p>
<p>The phase share varies with hardware, architecture and target speed. Combine the capacities <strong>before</strong> calculating log-speed area and normalizing to GB300.</p>
<p class="small">This is ideal fleet-scale allocation between separately provisioned, fully utilized pools of the same chip type. Output is paced at the requested target, rather than assuming all of a selected configuration’s extra speed is used. “Agents” means concurrent decoding requests after provisioning their input work—not wall-clock concurrent users including tools, idle time or waiting. We do not simulate mixed-batch contention or prefill/decode transfers here.</p>

<details><summary>Trace grounding: cohort, token ratio and representativeness</summary>
<p>It is informed by the <strong>public Syfi coding trace v0.0.2</strong> numeric request data analyzed in TraceLab. The saved warm-cache cohort contains <strong>110,565 requests</strong>, averaging <strong>1,422.854 new input tokens</strong> and <strong>940.206 output tokens</strong>. Their ratio is <strong>1.513342523752522</strong>.</p>
<p class="math">r = sum of new input tokens / sum of output tokens<br>= mean new input / mean output for the same request cohort</p>
<p>This is a ratio of aggregate token totals, not the mean of individual request ratios. The cohort selects Claude-provider requests with positive output, at least <strong>90% of total input already cached</strong>, <strong>131,072–393,216 cached tokens</strong>, and <strong>1–10,240 new input tokens</strong>. Cached tokens are not included in the numerator.</p>
<p class="small">We reuse the saved cohort statistics, query and database hash—not a fresh database extraction in this explainer update. This is a warm-cache coding-workload estimate, not a universal traffic distribution. The simulator still uses a fixed 262,144-token prefix and a 1,024-new-token prefill rate; it does not replay the cohort’s input-length distribution. The same token ratio is applied across architectures, without modeling tokenizer, verbosity or quality differences. Cold-cache misses are excluded.</p>
<p class="small">Reproducibility: <code>results/workload-capacity-oct15-mai2/workload.json</code> records the unchanged cohort query and source artifact. Database SHA-256: <code style="overflow-wrap:anywhere">a7bab286bc640844560850965ccf47975cf66407154132abaab90f27ec9be744</code>.</p>
</details>
<details><summary>Prefill optimization: winning configurations and a numerical example</summary>
<p><strong>Workload.</strong> Every request has the architecture-specific cache for its own 262,144-token prefix resident on the worker. It then receives <strong>1,024 new input tokens</strong>. The forward pass processes those new tokens using the existing cache, subject to the model’s attention pattern. We count only the new tokens as throughput—not the cached prefix. Prefill has <strong>no speculative decoding</strong> (nextn=0), unlike the decode curves.</p>
<p><strong>Optimization.</strong> Independently of the decode winners, choose the memory-admitted layout and batch with the highest new-input tokens per allocated chip-second. Prefill throughput P has <strong>no latency floor or separate AUC</strong>; it supplies the input-work term in the combined curve. If B is requests per attention-DP group, D is the group count, N is allocated chips and T is the modeled forward duration in seconds:</p>
<p class="math">Prefill throughput P = (B × D × 1,024) ÷ (N × T)<br>Use each hardware/model’s own P in the combined-capacity equation above.</p>
<div class="scroll"><table><thead><tr><th>Model / hardware</th><th>New tokens/chip/s</th><th>Requests / chips</th><th>Forward ms</th><th>Winning layout</th></tr></thead><tbody>%%PREFILL_ROWS%%</tbody></table></div>
<p><strong>Example: GLM on GB300.</strong> TP1 DP4 means 4 attention groups on 4 chips. Each handles 5 requests, so 20 requests add 20 × 1,024 = 20,480 tokens in 0.210475 seconds. Dividing by 4 chips gives <strong>24,325.9 new tokens/chip/s</strong>. ETP2 EP2 distributes the expert matrices over those same 4 chips. This differs from the 64-chip decode example because prefill optimizes throughput, not concurrent requests at a required output speed.</p>
<p>The table gives representative GLM and Kimi prefill winners; the four-model calculation also uses MAI-2 and Qwen’s independently optimized rates, shown in the explorer. Prefill and decode need not use the same layouts.</p>
<p class="small"><strong>Scope limits.</strong> This is cached-prefix prefill, not cold processing of the full 256Ki prefix. Prefix construction, KV transfers, queueing and contention with simultaneous decode are excluded. The finite search uses DCP1/PCP1, unchunked prefill and balanced attention-DP batches; it does not optimize chunking or prefill context parallelism. Node-restricted systems use the study’s conservative serial PP schedule, not an optimized overlapping pipeline. The 4Ki and 10Ki sensitivity runs do not enter this score.</p>
</details>
<details><summary>Deployment assumptions: separate pools and cache placement</summary>
<p>The current scores use the <strong>same accelerator type in both pools</strong>, with each pool independently optimized. Fractional prefill-chip counts represent a large fleet sharing that capacity across many decode instances, not fractions of individual machines.</p>
<p>Each phase is modeled with its required cache already resident. A real deployment must arrange that placement: a prefill worker needs access to the existing prefix before processing new input, and the decode worker needs the resulting KV afterward. The phase simulations include their own resident-cache memory costs, but the combination does not track cache lifetimes or charge additional fleet-level duplication and transfer costs. It is an ideal resource budget, not a complete disaggregated-serving simulation.</p>
</details>
</section>

<section class="step" id="compare" aria-labelledby="compare-title">
<p class="eyebrow">STEP 5 · COMPARE A RANGE OF INTERACTIVITIES</p>
<h2 id="compare-title">Measure the area under each combined-capacity curve</h2>
<p>Plot minimum output speed on the horizontal axis and <strong>decode concurrency per total prefill + decode chip</strong> on the vertical axis. Integrate over <strong>50–400 SOL TPS/user on a logarithmic speed axis</strong>. More area means more supported capacity across this range of speed requirements.</p>
<p><strong>Equal weight per doubling:</strong> 50–100, 100–200 and 200–400 each occupy the same width. This gives slower throughput-oriented agents and faster interactive agents a balanced place in the comparison. Equal width means equal integration weight—not equal area, which also depends on capacity.</p>
<p>For each model, divide the hardware’s area by GB300’s area. The tabs below show this calculation; move the slider to inspect the selected decode configuration and its added prefill allocation.</p>
<details><summary>Choosing an AUC instead of a single speed target</summary>
<p>A single target is appropriate for a deployment with one known speed requirement. The index instead summarizes a range of acceptable interactivities, avoiding dependence on one threshold that may sit just before or after a configuration switch. The explorer still shows individual targets, so the summary need not replace a workload-specific comparison.</p>
<p>Log weighting treats proportional speed changes consistently: doubling from 50 to 100 has the same integration weight as doubling from 200 to 400. The band and weighting are declared benchmark choices, not a statistically estimated user preference. A known application’s speed requirements can instead be evaluated directly in the explorer.</p>
<p class="small">We divide the two capacity areas, rather than averaging pointwise hardware/GB300 ratios. Where GB300 capacity is positive, this amounts to weighting the relative advantage by GB300’s capacity as well as the speed-axis weight. Large percentage advantages where baseline capacity is tiny therefore do not automatically dominate. A hardware’s feasible region still contributes area where GB300 has no feasible candidate.</p>
</details>
<details><summary>Why 50–400, and how is log area calculated?</summary>
<p>The band includes slower throughput-oriented workloads and faster interactive ones without making ultra-fast specialist serving the headline. Each doubling receives one third of the speed-axis weight. This is a transparent way to cover several operating regimes without assuming a specific distribution of demand.</p>
<p class="formula">Area = ∫<sub>50</sub><sup>400</sup> C<sub>combined</sub>(s) d ln(s) = ∫<sub>50</sub><sup>400</sup> C<sub>combined</sub>(s) ds / s</p>
<p>Between saved decode-speed breakpoints, decode capacity c is constant and combined capacity is c/(1 + rsc/P). Its exact log-speed area from a to b is <code>c × ln[b(1 + rca/P) / (a(1 + rcb/P))]</code>. We sum every segment; unsupported speeds contribute zero. The display grid does not determine the score. Dividing by ln(400/50) would express an average capacity, but cancels in the hardware/GB300 ratio. The log base also cancels.</p>
<p>At an illustrative 50% of modeled SOL speed <em>at fixed concurrency</em>, 50–400 SOL TPS/user corresponds to 25–200 achieved TPS/user. That is a scenario, not a universal calibration: implementation, configuration and hardware can change the achieved fraction. We apply no such discount to the index itself.</p>
</details>
<div class="chart">
<div class="tabs" role="tablist" aria-label="Choose model">
%%TABS%%
</div>
%%PANELS%%
<div class="controls"><label for="speed">Minimum speed: <strong><output id="speed-value" for="speed">100</output> TPS/user</strong></label><input id="speed" type="range" min="0" max="3000" value="1000" step="1" oninput="updateExplorer()" aria-label="Minimum SOL TPS per user, logarithmic slider"><button onclick="setSpeed(50)">50</button><button onclick="setSpeed(100)">100</button><button onclick="setSpeed(200)">200</button><button onclick="setSpeed(400)">400</button></div>
<div id="configs" class="configs" aria-live="polite"></div>
<p class="caption">The graph and bold density include prefill chips. The layout, batch, step time and memory describe the selected <em>decode</em> instance. Its maximum modeled speed can exceed the target; the workload calculation paces output at the target. Memory is the modeled total per rank (one chip), including weights, KV cache, workspace, fixed reservations and guards. Percentages use physical HBM as the denominator; the applicable memory limit is noted on each card. Additional prefill capacity is a fractional fleet-scale allocation, not necessarily a realizable fraction of one machine.</p>
<details><summary>What do the parallelism labels mean? A worked example</summary>
<p><button onclick="showModel(0);setSpeed(100)">Show GLM at 100 TPS/user</button> On GB300, the selected layout is <code>TP4 DP16 DCP4 ETP2 EP32 PP1</code>. These are different ways of organizing <strong>the same 64 chips</strong>, not six factors to multiply together.</p>
<div class="scroll"><table><thead><tr><th>Label</th><th>Meaning in this example</th></tr></thead><tbody>
<tr><th>TP4 — tensor parallelism</th><td>Four chips cooperate on sharded attention projections and other non-expert tensor work for each request group. TP is still relevant when DCP equals TP.</td></tr>
<tr><th>DP16 — attention data parallelism</th><td>16 groups handle different requests. Each group has 4 chips and a batch of 57 requests. These are not 16 independent full-model copies: they share the distributed expert layer.</td></tr>
<tr><th>DCP4 — decode context parallelism</th><td>The cached sequence and context-attention work are partitioned across the 4 chips within each TP group, with partial results merged. It uses the existing chips; it does not multiply the chip count again.</td></tr>
<tr><th>EP32 — expert parallelism</th><td>The routed experts are distributed into 32 partitions across that same chip set. This is a partition count, not the number of experts selected by each token.</td></tr>
<tr><th>ETP2 — expert tensor parallelism</th><td>Each expert partition’s matrices are sharded over 2 chips. Thus the expert view is 32 × 2 = 64 chips, matching the attention view of 16 × 4 = 64.</td></tr>
<tr><th>PP1 — pipeline parallelism</th><td>One stage, so layers are not split across additional pipeline stages. In general allocated chips = TP × DP × PP = ETP × EP × PP.</td></tr>
</tbody></table></div>
<p class="math">57 requests/group × 16 groups = <strong>912 concurrent requests</strong><br>912 ÷ 64 allocated chips = <strong>14.25 requests/chip</strong><br>23.097 ms/speculative step ÷ AL3 = 7.699 ms/output token → <strong>129.885 TPS/user</strong>, exceeding the 100 target.</p>
<p>The saved candidate passes memory admission at <strong>241.17 GiB/rank</strong>, or <strong>89.9% of the modeled 268.22 GiB physical HBM per chip</strong> (288 decimal GB). The admission ceiling is <strong>90% = 241.40 GiB</strong>; this usage already includes the study’s reservations and guards. A 72-chip deployment ceiling does not force use of 72: only the 64 allocated chips enter this density. At a different speed target, another layout or batch can win.</p>
<p>At a paced 100 TPS/user, the input supply requires 1.5133425 × 100 × 912 ≈ <strong>138,017 new tokens/s</strong>. With GB300 prefill throughput of 24,325.9 new tokens/chip/s, allocate the equivalent of <strong>5.674 prefill chips</strong> in addition to the 64 decode chips. Combined density is 912 ÷ 69.674 ≈ <strong>13.090 agents/total chip</strong>. This is a fleet-average resource budget, not a claim that a six-chip prefill layout was simulated.</p>
</details>
<p class="caption">Each shaded area is counted independently, including the overlap. Decode capacity is a step frontier from a finite search. Combined capacity slopes downward within each step as faster output needs more prefill chips. Integration is analytical between all saved speed breakpoints; visual sampling does not affect the scores. Candidates at exactly the threshold remain eligible; regions with no eligible candidate contribute zero. Areas shown rounded.</p>
</div>

</section>

<section class="step" id="combine" aria-labelledby="combine-title">
<p class="eyebrow">STEP 6 · SUMMARIZE THE ARCHITECTURE BASKET</p>
<h2 id="combine-title">Normalize each model first, then average the four scores</h2>
<p>Each model contributes one GB300-relative combined-area ratio. Their equal-weight arithmetic mean is the headline index. This prevents models with higher raw request capacity from dominating simply because they support more requests.</p>
<div class="score"><p class="small">VR200 worked example: GLM + Kimi + Qwen + MAI-2 2.8T</p><p class="formula">%%AVERAGE%%</p><p class="small">Calculated from unrounded values. GB300 is 1 for every model and therefore 1 for the average.</p></div>
<details><summary>Interpreting model-mixture sensitivity</summary>
<p>The variation is real within the model: different attention/cache patterns, expert work, memory fit and communication expose different hardware advantages. An average is useful as a <strong>defined basket score</strong>, not as a universal multiplier. We normalize each model to GB300 first, then give the four architectures equal weight so raw capacity on one model cannot overwhelm the others.</p>
<p>For example, the current combined calculation gives VR200 <strong>%%VR_GLM%%× on GLM</strong> and <strong>%%VR_KIMI%%× on Kimi</strong>. Qwen and MAI-2 broaden architectural coverage; they do not remove model-mixture sensitivity. Use the per-model table and curves whenever the intended architecture is known. Equal model weights express a benchmark choice, not known traffic shares or a statistical confidence interval.</p>
</details>
<details><summary>Versioning the index and testing stability over time</summary>
<p><strong>Stability across successive model generations has not yet been established.</strong> Additional architecture studies and older saved runs are not a same-engine, same-policy backtest across actual frontier checkpoints. Hypothetical larger models cannot substitute for that test. Stable ordering in a limited sensitivity study also does not establish stable GB300-equivalent magnitudes.</p>
<p>Treat a score as belonging to a fixed version of the model basket, workload, hardware profiles and simulator. A future model update can legitimately change it. For longitudinal comparisons, retain the frozen version; when publishing a new basket, rerun the compared hardware consistently and show old/new results side by side. The current index should not be treated as an immutable physical conversion factor.</p>
</details>
<details><summary>All 23 accelerator profiles and their model components</summary>
<p>GB300 is first; the remaining profiles are ordered by combined score. Components below are already normalized to GB300 per model. Roadmap profiles, capacity scenarios and precision/topology assumptions remain modeled inputs, not verified serving benchmarks.</p>
<div class="scroll"><table><thead><tr><th>Accelerator</th><th>Combined</th><th>GLM</th><th>Kimi</th><th>Qwen</th><th>MAI-2 2.8T</th></tr></thead><tbody>%%SCORE_ROWS%%</tbody></table></div>
</details>
<p>The result is a <strong>versioned, workload-specific hardware comparison</strong>, not a universal exchange rate between accelerators. Keep the component scores visible when using the index for a model-specific deployment decision.</p>
</section>
<footer>Analytical hardware comparison, not measured serving performance or an exact cost ratio. Excludes queueing, TTFT, tool pauses, temporal agent duty cycles and cold-cache misses; vendor implementation efficiency can differ. Counts allocated chips, not idle rack capacity. Combined areas are normalized per model, then equally averaged. Calculations use unrounded saved results; this explainer update launches no simulations.<br>
<a href="PERFORMANCE_KNEES.html">Next: explore knee points and performance bottlenecks →</a><br>Data and full score table are embedded in this standalone HTML. <a href="methodology_current_data.json">Download companion example data</a> (when opened alongside the companion file).
</footer></main>
<script id="example-data" type="application/json">%%DATA%%</script>
<script>
const exampleData=JSON.parse(document.getElementById('example-data').textContent);
let selectedModel=1;
function winner(points,speed){return points.filter(p=>p.speed>=speed).reduce((best,p)=>!best||p.capacity>best.capacity?p:best,null);}
function setSpeed(speed){document.getElementById('speed').value=1000*Math.log2(speed/50);updateExplorer();}
function speedX(speed){return 76+Math.log(speed/50)/Math.log(8)*856;}
function memoryParts(p,physical){
 const parts=[['Weights',p.weights_gib,'#2563a6'],['KV cache',p.cache_gib,'#438261'],['Workspace + reservations/guards',p.other_memory_gib,'#8795a1']];
 // The final 10% is a budget overlay, not another allocation added to usage.
 // Kimi's existing static-pool note covers transient overlap with this region.
 return `<div style="position:relative;display:flex;height:22px;background:white;border:1px solid #dce3e8;border-radius:3px;overflow:hidden;margin:8px 0" aria-hidden="true">${parts.map(([name,gib,color])=>`<span style="flex:0 0 ${100*gib/physical}%;background:${color}"></span>`).join('')}<span title="Nominal 10% safety buffer" style="position:absolute;left:90%;right:0;top:0;bottom:0;border-left:1px dashed #526675;background:repeating-linear-gradient(135deg,rgba(82,102,117,.18) 0 3px,rgba(255,255,255,.65) 3px 7px);text-align:center;font-size:10px;line-height:20px;color:#20313d">10%</span></div><div class="small number">${parts.map(([name,gib,color])=>`<span style="color:${color}">●</span> ${name}: <b>${gib.toFixed(1)} GiB</b> (${(100*gib/physical).toFixed(1)}%)`).join('<br>')}<br>▧ <b>Safety buffer: ${(physical*.1).toFixed(1)} GiB (last 10%)</b><br>Percent of physical HBM; hatching marks the nominal buffer.</div>`;
}
function updateExplorer(){
 const slider=document.getElementById('speed'),speed=Math.round(50*2**(Number(slider.value)/1000)), model=exampleData.models[selectedModel];
 slider.setAttribute('aria-valuetext',speed+' SOL TPS per user');
 document.getElementById('speed-value').textContent=speed;
 document.getElementById('configs').innerHTML=Object.entries(model.curves).map(([hw,points],i)=>{
  const p=winner(points,speed),dot=document.getElementById(`dot-${selectedModel}-${i}`);
  const rate=model.prefill_rates[hw],extra=p&&rate>0?exampleData.ratio*speed*p.capacity/rate:0;
  const combined=p&&rate>0?p.capacity/(1+extra):0;
  dot.setAttribute('cx',speedX(speed));dot.setAttribute('cy',370-combined/model.ymax*346);
  if(!p)return `<div class="config"><b>${hw}</b><p>No feasible saved configuration.</p></div>`;
  const mem=exampleData.memory_profiles[hw],usedPercent=100*p.memory/mem.physical_gib;
  const memoryLimit=model.model==='Kimi-K3'?'Configured fraction: 90% of the static pool after runtime overhead; displayed total also includes transient activations.':'Admission ceiling: 90% of physical HBM ('+(mem.physical_gib*.9).toFixed(2)+' GiB).';
  const layout=p.layout||['tp','dp','dcp','etp','ep','pp'].map(k=>k.toUpperCase()+p[k]).join(' ');
  return `<div class="config"><b>${hw}</b><p><strong>${combined.toFixed(3)}</strong> agents / total chip at <b>${speed} TPS/user</b></p><p>Decode density ${p.capacity.toFixed(3)} requests/chip; native maximum ${p.speed.toFixed(1)} TPS/user.</p><p class="number">${p.requests.toLocaleString()} requests ÷ ${p.chips} decode chips · ${p.batch} requests/DP group<br>${p.raw_ms.toFixed(3)} ms/speculative step</p><p class="number"><b>Modeled memory:</b> ${p.memory.toFixed(2)} GiB/rank — <b>${usedPercent.toFixed(1)}% of physical HBM</b><br>Physical: ${mem.physical_gib.toFixed(2)} GiB/chip.<br>${memoryLimit}</p>${memoryParts(p,mem.physical_gib)}<code>${layout}</code><p>Prefill: ${rate.toLocaleString(undefined,{maximumFractionDigits:1})} new tokens/chip/s.<br>Add ${(extra*p.chips).toFixed(3)} equivalent prefill chips to this decode allocation; ${(100*extra/(1+extra)).toFixed(1)}% of total chips.</p></div>`;
 }).join('');
 document.getElementById('cursor-'+selectedModel).setAttribute('d',`M${speedX(speed)},24 V370`);
}
function showModel(n){selectedModel=n;for(let i=0;i<exampleData.models.length;i++){document.getElementById('model-'+i).hidden=i!==n;const tab=document.getElementById('tab-'+i);tab.setAttribute('aria-selected',String(i===n));tab.tabIndex=i===n?0:-1;}updateExplorer();}
document.querySelectorAll('[role="tab"]').forEach(tab=>tab.addEventListener('keydown',event=>{if(['ArrowLeft','ArrowRight','Home','End'].includes(event.key)){event.preventDefault();const count=exampleData.models.length,n=event.key==='Home'?0:event.key==='End'?count-1:(selectedModel+(event.key==='ArrowRight'?1:count-1))%count;showModel(n);document.getElementById('tab-'+n).focus();}}));
updateExplorer();
</script></body></html>'''


if __name__ == '__main__':
    main()
