"""Standalone teaching companion: saved native curves + labeled toy equations.

No AISim execution, no new hardware projections, no score changes.
"""
import csv
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/methodology-current'
REFERENCE = ROOT / 'reference/2026-10-15'
MODELS = [
    ('GLM-5.2', 'GLM-5.2', 256, 8),
    ('Kimi-K3', 'Kimi-K3', 896, 16),
    ('Qwen-480B', 'Qwen3-Coder 480B', 160, 8),
    ('MAI2-informed-h128-w512-sh4352', 'MAI-2 2.8T', 512, 7),
]


def main():
    data = json.loads((REFERENCE/'performance_knees_data.json').read_text())
    models = data['models']
    page = PAGE.replace('%%DATA%%',json.dumps(data).replace('</','<\\/'))
    assert '%%' not in page and '/home/' not in page
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/'PERFORMANCE_KNEES.html').write_text(page)
    (OUT/'performance_knees_data.json').write_text(json.dumps(data,indent=2)+'\n')
    print(OUT/'PERFORMANCE_KNEES.html')
    print('Saved curves:',sum(len(c) for m in models for c in m['curves'].values()),'; no simulations run')


PAGE = r'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Reading the knees in inference performance</title><style>
*{box-sizing:border-box}body{margin:0;color:#20313d;background:#fff;font:16px/1.55 system-ui,sans-serif}main{max-width:1080px;margin:auto;padding:32px 24px}h1{font-size:35px;line-height:1.15;letter-spacing:-.03em}h2{font-size:23px;margin:0 0 12px}h3{font-size:17px}a{color:#2563a6}.lead{font-size:19px;max-width:860px}.small,figcaption{font-size:13px;color:#526675}.grid{display:grid;grid-template-columns:repeat(3,1fr);gap:14px}.card{background:#f3f6f8;padding:16px;border-radius:8px}.card p{font-size:14px;margin-bottom:0}section{border-top:1px solid #dce3e8;margin-top:28px;padding-top:24px}.controls{display:flex;gap:12px;align-items:center;flex-wrap:wrap;margin:16px 0}select,button{font:inherit;border:1px solid #bccbd5;border-radius:6px;padding:7px;background:#fff}input{accent-color:#2563a6}.plots{display:grid;grid-template-columns:1fr 1fr;gap:18px}svg{width:100%;height:auto;display:block}svg text{font:13px system-ui;fill:#526675}.legend{display:flex;gap:18px;flex-wrap:wrap;font-size:14px}.blue{color:#2563a6}.orange{color:#b94b20}.green{color:#368269}.purple{color:#8756a1}.math{background:#f3f6f8;padding:12px 16px;font-variant-numeric:tabular-nums}table{border-collapse:collapse;width:100%;font-size:13px}td,th{padding:8px;border-bottom:1px solid #dce3e8;text-align:left}.scroll{overflow:auto}summary{cursor:pointer;color:#2563a6;font-weight:600}details{margin:16px 0}code{overflow-wrap:anywhere;font-size:12px}.notice{border-left:3px solid #b94b20;padding:8px 14px;background:#fbf5ef}footer{margin-top:32px;padding-top:16px;border-top:1px solid #dce3e8;font-size:12px;color:#526675}button:focus-visible,select:focus-visible,input:focus-visible{outline:3px solid #2563a6;outline-offset:3px}@media(max-width:700px){main{padding:24px 14px}.grid,.plots{grid-template-columns:1fr}h1{font-size:29px}}@media print{details{display:block}.controls{display:none}section{break-inside:avoid}}
.page-link{display:block;padding:15px 20px;border:2px solid #2563a6;border-radius:9px;background:#eef5fc;font-weight:650;text-decoration:none}.page-link:hover{background:#e1eefb}.page-link:focus-visible{outline:3px solid #b94b20;outline-offset:3px}.section-nav{display:flex;gap:10px 18px;flex-wrap:wrap;margin:20px 0;font-size:14px}section{scroll-margin-top:20px}select{max-width:100%}.controls label{min-width:0;max-width:100%}
</style></head><body><main>
<a class="page-link" href="REPORT.html">← Main explainer: how the agent capacity index works</a>
<p><a href="ACCELERATOR_COMPARISON.html">Compare all accelerators: score charts →</a></p>
<h1>Reading the knees in inference performance</h1>
<p class="lead">Adding more concurrent requests spreads weight reads over more output tokens—but also adds compute, cache traffic and communication. Several different effects can make a curve bend. They should not all be called “saturation.”</p>
<p class="small">Companion to the index · saved AISim SOL results plus explicitly labeled teaching examples · no new accelerator predictions</p>
<p>A <strong>knee point</strong> is a bend or change in slope. It can reflect a resource limit, but it can also arise from changing axes or selecting a different configuration. This page separates those explanations.</p>
<nav class="section-nav" aria-label="Knee-point guide sections"><a href="#native-curves">1 · Saved curves</a><a href="#axes">2 · Changing axes</a><a href="#experts">3 · Expert coverage</a><a href="#roofline">4 · Resource crossover</a><a href="#diagnosis">5 · Diagnose a knee</a></nav>
<div class="grid">
<div class="card"><h3>Expert-weight coverage</h3><p>Once a batch touches the resident expert inventory, more tokens no longer require new <em>distinct expert weights</em> for that operation. Expert computation still grows.</p></div>
<div class="card"><h3>Bandwidth ↔ compute</h3><p>An operation changes bottleneck when its compute time exceeds its memory-transfer time, or vice versa. Different operations can be limited by different resources in the same step.</p></div>
<div class="card"><h3>Memory-capacity ceiling</h3><p>Weights, KV and reservations fill the admitted HBM budget. This stops the batch from growing; it does not itself make each step compute-bound or bandwidth-bound.</p></div>
</div>

<section id="native-curves"><h2>1. A fixed configuration: latency rises; throughput can flatten</h2>
<p>These are <strong>saved native AISim results</strong>, not a fitted illustration. Hold parallelism fixed while increasing batch. Both hardware profiles use 32 allocated chips; these teaching configurations are not necessarily index winners.</p>
<div class="controls"><label>Model <select id="model" onchange="render()"></select></label><label>Layout <select id="tp" onchange="render()"><option value="8">TP8 DP4 · EP32 ETP1</option><option value="1">TP1 DP32 · EP32 ETP1</option></select></label></div>
<p id="layout" class="math"></p>
<div class="legend"><span class="blue">━ GB300</span><span class="orange">━ VR200</span><span>● integer-batch results; ○ last admitted batch</span><span class="purple">┆ expert-weight cap in verification</span></div>
<div class="plots"><div><h3>Time per output token (TPOT)</h3><div id="latency"></div></div><div><h3>Aggregate output throughput per chip</h3><div id="throughput"></div></div></div>
<h3>The same saved points, now against interactivity</h3>
<p>Here the horizontal axis is each configuration’s <strong>actual modeled TPS/user on a log scale</strong>: equal horizontal distances represent equal proportional speed changes. The left plot counts output tokens; the right counts concurrent requests. Neither re-optimizes across layouts. Moving to the right generally means selecting a smaller batch.</p>
<div class="plots"><div><h3>Token throughput versus interactivity</h3><div id="native-rate-speed"></div></div><div><h3>Concurrency versus interactivity</h3><div id="native-capacity-speed"></div></div></div>
<p class="small">These are the same integer-batch observations as above, connected to guide the eye—not additional simulated points or an optimized capacity frontier. The open endpoint still marks the largest admitted batch, which is generally toward the slow end of the interactivity axis.</p>
<div id="actual-note" class="notice"></div>
<div class="scroll"><table><thead><tr><th>Hardware</th><th>Last admitted B / total requests</th><th>Requests/chip</th><th>Memory budget used / chip</th><th>TPOT</th><th>TPS/user</th><th>Output TPS/chip</th></tr></thead><tbody id="actual-rows"></tbody></table></div>
<p class="small">Memory percentages use physical HBM as the denominator. The table separates resident weights and KV cache; the remainder includes workspace, reservations and guards. Each row notes the applicable 90% admission setting.</p>
<p class="small">B is requests per attention-DP group, not requests per physical chip. Total requests = B × DP; requests/chip = B/TP for these PP1 layouts. DCP1 and ETP1 throughout. Each request has an existing 262,144-token context, with retained cache determined by its attention architecture. Native nextn3 verification and draft work are included; AL3 converts iteration time to average TPOT, not a guarantee of evenly spaced tokens. A point at the admission ceiling can be bandwidth-bound, compute-bound, or mixed. Endpoints reflect the configured 90% fraction, backend-specific memory-pool rules and guards—not 100% use of physical HBM.</p>
<details><summary>A useful rule for doubling concurrency</summary><p>Doubling concurrency changes throughput by 2T(n)/T(2n), and TPS/user by T(n)/T(2n), for fixed layout and accepted length. If TPOT rises 10%, throughput rises about 82%, while TPS/user falls about 9%. If TPOT doubles, throughput is unchanged and TPS/user halves. Use the actual T values; “all experts covered” alone cannot tell you which case applies.</p></details>
</section>

<section id="axes"><h2>2. One straight latency line, four different-looking plots</h2>
<p><strong>Interactivity = output TPS/user.</strong> Token throughput is output tokens per chip-second; concurrency is requests per chip. Confusing those vertical axes can make the same data appear to tell different stories.</p>
<p>For a fixed layout, suppose TPOT is approximately <strong>T(n) = a + bn</strong>, with n concurrent requests per chip and time in seconds. The intercept a represents approximately batch-shared time in the regime being examined; b represents the extra time per added request. For example, covered expert-weight reads can contribute a shared cost, while compute and cache traffic grow with n. These coefficients are a local approximation—not a universal physical decomposition.</p>
<div class="math">Interactivity s = 1/T(n) = 1/(a + bn)<br>Token throughput/chip Q = n/T(n) = n/(a + bn)<br>Eliminating n: <strong>Q(s) = (1 − as)/b</strong><br>Concurrency/chip n(s) = Q(s)/s = <strong>(1/s − a)/b</strong></div>
<p><strong>The surprising part:</strong> a straight TPOT–concurrency line produces a flattening throughput–concurrency curve, but a <em>straight</em> throughput–interactivity line. Concurrency–interactivity is curved because we divide by speed once more.</p>
<div class="controls"><label for="example-n">Follow one point: <strong><output id="example-n-value">10</output> concurrent requests/chip</strong></label><input id="example-n" type="range" min="1" max="80" value="10" step="1" oninput="renderAxes()"></div>
<p class="small">Illustrative only: T(n) = 10 + n milliseconds. No memory ceiling is applied here; the displayed n=1–80 range is chosen for teaching. Orange dots identify the same operating point on all four panels.</p>
<div class="plots">
<div><h3>A · TPOT versus concurrency: straight</h3><div id="axis-tpot"></div></div>
<div><h3>B · Throughput versus concurrency: flattening</h3><div id="axis-rate"></div></div>
<div><h3>C · Throughput versus interactivity: straight</h3><div id="axis-rate-speed"></div></div>
<div><h3>D · Concurrency versus interactivity: curved</h3><div id="axis-capacity-speed"></div></div>
</div>
<p id="axes-note" class="math" aria-live="polite"></p>
<div class="scroll"><table><thead><tr><th>Concurrency/chip n</th><th>TPOT</th><th>TPS/user s</th><th>Output TPS/chip Q</th></tr></thead><tbody>
<tr><td>1</td><td>11 ms</td><td>90.9</td><td>90.9</td></tr><tr><td>5</td><td>15 ms</td><td>66.7</td><td>333.3</td></tr><tr><td>10</td><td>20 ms</td><td>50.0</td><td>500.0</td></tr><tr><td>20</td><td>30 ms</td><td>33.3</td><td>666.7</td></tr><tr><td>40</td><td>50 ms</td><td>20.0</td><td>800.0</td></tr>
</tbody></table></div>
<details><summary>Why does throughput flatten while TPOT keeps rising?</summary>
<p>At low concurrency, more requests spread the fixed 10 ms over more output tokens. At large concurrency, the n milliseconds of per-request work dominate: doubling n nearly doubles elapsed time. Token throughput approaches <strong>1,000 tokens/chip/s</strong>, while TPS/user continues toward zero. This plateau can result from compute, cache bandwidth or another cost proportional to concurrency; its existence alone does not identify the resource.</p>
<p>For general positive a and b, Q(n) approaches 1/b, with slope a/(a+bn)². At n=a/b, the shared and growing time terms are equal, and throughput is half its asymptote. That is a smooth amortization knee—not necessarily a new hardware bottleneck. If a were zero, the linear-latency model would give constant throughput from the outset.</p>
</details>
<details><summary>Why is throughput versus interactivity straight, but concurrency versus interactivity curved?</summary>
<p>In this example, Q(s) = <strong>1,000 − 10s</strong>. Every extra 10 TPS/user costs 100 output tokens/chip/s along this fixed-layout relationship. But concurrency is Q/s = <strong>1,000/s − 10</strong>, an inverse curve. These formulas describe the modeled operating range, not a valid extrapolation to negative throughput or zero requests.</p>
<p>Even with completely constant throughput of 1,000 tokens/s, the same system supports 20 decoding requests at 50 TPS/user, 10 at 100 TPS/user, or 5 at 200 TPS/user. The inverse shape appears without any resource transition at all. Conversely, the token-throughput/interactivity line bends when the latency law changes, such as expert-coverage saturation, a roofline crossover or layout selection.</p>
</details>
<details open><summary>How does this relate to the index’s capacity-versus-interactivity frontier?</summary>
<p>The native plots above use <strong>modeled SOL speed for each batch</strong>, not measured achieved performance. The index instead asks for the largest concurrency meeting a <strong>minimum speed threshold</strong>. For a continuous fixed-layout approximation with memory ceiling n<sub>max</sub>, decode capacity would be min[n<sub>max</sub>, (1/s − a)/b], within the range where at least one request fits and meets the target.</p>
<p>In this fixed-layout example, sufficiently low thresholds admit the largest memory-feasible batch: relaxing the target further cannot add requests, so decode capacity is flat. At higher thresholds, the speed requirement forces lower concurrency. The index combines integer batches and multiple layouts, giving steps and configuration switches rather than a smooth inverse curve. A configuration with modeled speed above the threshold can remain the winner across an entire interval; a flat region alone does not prove that physical HBM is full.</p>
<p>The current combined index then counts prefill chips too: <strong>C<sub>combined</sub>(s) = C<sub>decode</sub>(s) / [1 + r × s × C<sub>decode</sub>(s)/P]</strong>. Even when decode capacity is flat, increasing the target provisions more prefill chips, making combined concurrency per total chip slope downward. This is resource accounting—not evidence that the decode kernels changed bottleneck. Multiplying combined capacity by s gives the corresponding <em>paced</em> output throughput per total chip; the index plots capacity, not that token-throughput quantity.</p>
</details>
<p class="notice"><strong>Read the axes before diagnosing a knee.</strong> Curvature can come from a reciprocal, a capacity cutoff, or changing the selected configuration. A bottleneck claim needs evidence from the operation costs.</p>
</section>

<section id="experts"><h2>3. When are all experts activated?</h2>
<p><strong>In the current simulator, this means a cap on expert-weight reads—not a prediction of routing coverage.</strong> Let E be experts per routed layer, k experts selected per token, and N the token positions entering that layer across the expert-parallel group.</p>
<p class="math">Approximate native weight-read count = min(E, N × k)<br>Integer rank form = min(E/EP, floor(N × k/EP)) per expert rank<br>Main verification: N = B × DP × 4; ordinary decode: N = B × DP</p>
<p>The cap is reached at N ≥ E/k for valid evenly divisible layouts. It does not change the per-token top-k: each token still selects k experts, not E. All weights are already allocated in memory before this point; coverage changes which weights are read during an operation, not which weights must fit.</p>
<div class="controls"><label>Coverage illustration <select id="width" onchange="renderCoverage()"><option value="4">4 verification positions / request</option><option value="1">1 position / request (no-spec illustration)</option></select></label></div>
<div class="legend"><span class="purple">━ Native per-rank read-count proxy</span><span class="green">━ Expected distinct experts: independent uniform routing</span></div>
<div id="coverage"></div><p id="coverage-note" class="math"></p>
<div class="scroll"><table><thead><tr><th>Architecture</th><th>E / k</th><th>Native cap: token positions</th><th>Requests with 4 positions</th><th>Independent-routing positions for ≥95% expected coverage</th></tr></thead><tbody id="expert-table"></tbody></table></div>
<details><summary>Why actual routing need not touch every expert at that point</summary>
<p>If each position independently selects k distinct experts uniformly, the expected number touched is <strong>E × [1 − (1 − k/E)<sup>N</sup>]</strong>. At roughly E/k positions, that is only around 63% of the expert inventory, not all of it. Approximately three times as many independent positions are needed for 95% expected coverage. “95% expected coverage” is not a 95% probability that every expert is touched.</p>
<p>Real routing can be nonuniform and correlated, especially between adjacent speculative positions. That can reduce distinct coverage or concentrate work on a few ranks; the most loaded rank can determine step time. The green curve is a mathematical comparison, <strong>not measured routing and not used in the score</strong>. There is no universal request count at which all real experts become active.</p>
<p>The native rank formula also floors token assignments. At very small batches it can undercount rank traffic, including assigning zero modeled expert-weight reads when N × k is below EP. This is a modeling artifact, not physically free execution. The purple marker concerns the main verification MoE; separately represented draft operations need not reach their cap at the same request count. AL3 affects accepted output/time, not the four-position verification input size.</p>
</details>
</section>

<section id="roofline"><h2>4. Expert coverage is not the compute crossover</h2>
<p>For each roofline-modeled operation, use <strong>max(compute time, HBM time)</strong>. After expert-weight coverage saturates, the fixed weight-read term stops growing, but compute and activation traffic continue to grow with token count. That can produce a later bandwidth-to-compute crossover.</p>
<div class="controls"><label>Illustrative HBM bandwidth multiplier <input id="bw" type="range" min="1" max="3" step=".05" value="1" oninput="renderToy()"> <output id="bw-value">1.00×</output></label></div>
<div class="legend"><span class="orange">━ MoE HBM time</span><span class="blue">━ MoE compute time</span><span>━ Total toy step time</span><span>┆ Memory admission ceiling</span></div>
<div id="toy"></div><p id="toy-note" class="math"></p>
<p class="notice"><strong>Teaching example only.</strong> Hand-chosen coefficients, not a model fit or a GB300/VR200 estimate. Solid curves are inside the arbitrary capacity limit; dashed continuations ask “what if memory were unlimited?” Moving the bandwidth slider changes bandwidth only—not memory capacity, compute or interconnect.</p>
<details><summary>The exact illustrative equations</summary><p class="math">HBM time = [2 × min(B/8, 1) + 0.005B] / bandwidth multiplier<br>Compute time = 0.015B<br>Total time = max(HBM time, compute time) + 0.3 + 0.02B<br>Admission ceiling: B = 96; all times in illustrative milliseconds</p><p>Expert reads cap at B=8. The +0.005B term represents activation traffic; +0.3+0.02B represents sequential work outside this MoE operation. At bandwidth multiplier 1, the post-coverage crossover is B=200, beyond the capacity ceiling. Faster HBM moves that crossover to a smaller B; capacity itself stays at 96. This illustrates why a compute-bound regime may exist mathematically but never be reached before memory fills.</p></details>
</section>

<section id="diagnosis"><h2>5. How to diagnose a knee in practice</h2>
<div class="scroll"><table><thead><tr><th>What changed?</th><th>What it can mean</th><th>What to check</th></tr></thead><tbody>
<tr><th>Weight-read slope flattens</th><td>More requests reuse an already covered expert inventory.</td><td>Distinct-weight byte counts versus token assignments. Compute and cache costs may still rise.</td></tr>
<tr><th>An operation changes slope</th><td>Compute overtakes HBM time, or an integer partition changes.</td><td>Separate FLOP and byte times for that operation—not just end-to-end latency.</td></tr>
<tr><th>A fixed-layout curve ends</th><td>The next batch fails memory admission.</td><td>Weights + KV + workspace + reservations + guards versus usable HBM. This is not a throughput plateau.</td></tr>
<tr><th>The optimized capacity frontier jumps</th><td>A different layout or integer batch wins at the new speed target.</td><td>The selected TP/DP/EP/ETP/DCP/PP and batch on both sides. A frontier kink need not be a resource crossover.</td></tr>
<tr><th>More bandwidth stops helping much</th><td>Remaining compute, communication or other sequential work limits the gain.</td><td>Per-operation times and collective costs. Faster HBM does not automatically improve either.</td></tr>
</tbody></table></div>
<p>Cache <em>storage</em> and cache <em>reads per step</em> are separate. Full attention revisits a long prefix; sparse attention can read a subset while still retaining much of the cache, with index work adding its own cost. Sliding-window layers can reduce both retained cache and reads, but a minority of global layers can still matter. More capacity extends feasible concurrency; more bandwidth changes time for bandwidth-limited work.</p>
<p class="small">The saved fixed-layout CSVs above contain total timing and memory admission, not enough operation-level breakdown to certify a specific compute/communication crossover. The expert-cap markers are derived from the inspected native formula. We do not infer a measured bottleneck solely from the shape of a total-time curve.</p>
<details><summary>Sources and reproducibility</summary><p>AISim commit <code>2f0f4cd8606d580567b253d02f3d699fe36db865</code>. The native plots use the same frozen October 10 engine and saved candidate sets used by the current index, restricted to two explicit layouts. No old independent analytical saturation study is substituted for native timing.</p><div id="sources"></div><p>Generator: <code>studies/performance_knees_visual.py</code>. Input hashes and plotted values are embedded in this standalone HTML and in <code>performance_knees_data.json</code>. No simulator call or index recalculation is performed by this generator.</p></details>
</section>
<footer>Modeled SOL, not measured serving performance. Native plots include speculative decoding; the optional one-position coverage illustration does not change those plots. No prefill resource adjustment is included in these fixed-layout decode charts; the main explainer covers that separate calculation.</footer>
</main><script id="data" type="application/json">%%DATA%%</script><script>
const D=JSON.parse(document.getElementById('data').textContent),colors={GB300:'#2563a6',VR200:'#b94b20'};
const el=id=>document.getElementById(id);
el('model').innerHTML=D.models.map((m,i)=>`<option value="${i}">${m.label}</option>`).join('');
const selected=()=>D.models[Number(el('model').value)];
function plot(series,xmax,ymax,xlabel,ylabel,markers=[],cut=null,logMin=null){
 const W=720,H=350,L=70,R=18,T=18,B=58,x=n=>L+(logMin===null?n/xmax:Math.log(n/logMin)/Math.log(xmax/logMin))*(W-L-R),y=n=>H-B-n/ymax*(H-T-B);
 let svg=`<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="${ylabel} versus ${xlabel}">`;
 for(let i=0;i<=4;i++){let yy=ymax*i/4;svg+=`<path d="M${L},${y(yy)}H${W-R}" stroke="#dce3e8"/><text x="${L-9}" y="${y(yy)+4}" text-anchor="end">${yy.toFixed(yy<10?1:0)}</text>`;}
 const ticks=logMin===null?Array.from({length:5},(_,i)=>xmax*i/4):Array.from({length:Math.round(Math.log2(xmax/logMin))+1},(_,i)=>logMin*2**i);
 for(const xx of ticks)svg+=`<text x="${x(xx)}" y="${H-B+22}" text-anchor="middle">${xx.toFixed(xx<10?1:0)}</text>`;
 for(const s of series){
  function path(pts,dash){if(!pts.length)return '';return `<path d="M${pts.map(p=>`${x(p[0])},${y(p[1])}`).join(' L')}" fill="none" stroke="${s.color}" stroke-width="2.5" ${dash?'stroke-dasharray="6 5"':''}/>`;}
  if(cut!==null){svg+=path(s.points.filter(p=>p[0]<=cut),false)+path(s.points.filter(p=>p[0]>=cut),true);}else svg+=path(s.points,false);
  if(s.dots) for(const p of s.points)svg+=`<circle cx="${x(p[0])}" cy="${y(p[1])}" r="2" fill="${s.color}"/>`;
  if(s.end){const p=s.points[s.points.length-1];svg+=`<circle cx="${x(p[0])}" cy="${y(p[1])}" r="5" fill="white" stroke="${s.color}" stroke-width="2"/>`;}
 }
 for(const m of markers)if(m.x<=xmax){svg+=`<path d="M${x(m.x)},${T}V${H-B}" stroke="${m.color||'#8756a1'}" stroke-dasharray="4 4"/><text x="${Math.min(W-100,Math.max(100,x(m.x)+5))}" y="${m.y||31}" fill="${m.color||'#8756a1'}">${m.label}</text>`;}
 svg+=`<text x="${(L+W-R)/2}" y="${H-10}" text-anchor="middle">${xlabel}</text><text transform="translate(18,${(T+H-B)/2}) rotate(-90)" text-anchor="middle">${ylabel}</text></svg>`;
 return svg;
}
function render(){
 const m=selected(),tp=Number(el('tp').value),dp=32/tp,capB=Math.ceil(m.experts/(4*dp*m.topk));
 const sets=Object.keys(colors).map(hw=>({hw,rows:m.curves[`${hw}-${tp}`]}));
 const xmax=Math.max(...sets.map(s=>s.rows.at(-1).b))*1.07;
 for(const [id,key,label] of [['latency','tpot','TPOT (ms / output token)'],['throughput','throughput','Output tokens / chip / second']]){
  const series=sets.map(s=>({color:colors[s.hw],points:s.rows.map(r=>[r.b,r[key]]),dots:true,end:true}));
  el(id).innerHTML=plot(series,xmax,Math.max(...series.flatMap(s=>s.points.map(p=>p[1])))*1.13,'Requests / attention-DP group (B)',label,[{x:capB,label:`Read cap B=${capB}`}]);
 }
 for(const [id,key,label] of [['native-rate-speed','throughput','Output tokens / chip / second'],['native-capacity-speed','density','Concurrent requests / chip']]){
  const series=sets.map(s=>({color:colors[s.hw],points:s.rows.map(r=>[r.speed,r[key]]),dots:true,end:true}));
  const speeds=series.flatMap(s=>s.points.map(p=>p[0])),lo=25*2**Math.floor(Math.log2(Math.min(...speeds)/25)),hi=25*2**Math.ceil(Math.log2(Math.max(...speeds)/25));
  el(id).innerHTML=plot(series,hi,Math.max(...series.flatMap(s=>s.points.map(p=>p[1])))*1.13,'Actual SOL TPS / user (log scale)',label,[],null,lo);
 }
 el('layout').innerHTML=`TP${tp} × DP${dp} = EP32 × ETP1 = <strong>32 chips</strong>; DCP1, PP1. Total requests = ${dp}B; requests/chip = B/${tp}.`;
 el('actual-note').innerHTML=`The main verification MoE reaches its native expert-weight cap at <strong>B=${capB}</strong> (${capB*dp} requests, ${capB*dp*4} verification positions). ${capB===1?'It is already capped at the first plotted batch: there is no later expert-coverage knee to discover in this layout.':'This marker is calculated from expert counts and the native read rule; it is not a measured bottleneck classification.'} Increasing B beyond it can still slow the overall step substantially.`;
 el('actual-rows').innerHTML=sets.map(s=>{const r=s.rows.at(-1),mem=D.memory_profiles[s.hw],limit=m.key==='Kimi-K3'?'90% static-pool setting; displayed total includes transient activations':'Ceiling: 90% ('+(mem.physical_gib*.9).toFixed(2)+' GiB)';return `<tr><th>${s.hw}</th><td>${r.b} / ${r.requests}</td><td>${r.density.toFixed(3)}</td><td>${r.memory.toFixed(2)} GiB · <b>${(100*r.memory/mem.physical_gib).toFixed(1)}%</b><br><span class="small">Weights: ${r.weights.toFixed(1)} GiB<br>KV cache: ${r.cache.toFixed(1)} GiB<br>Other: ${(r.memory-r.weights-r.cache).toFixed(1)} GiB<br>Physical: ${mem.physical_gib.toFixed(2)} GiB<br>${limit}</span></td><td>${r.tpot.toFixed(3)} ms</td><td>${r.speed.toFixed(1)}</td><td>${r.throughput.toFixed(1)}</td></tr>`}).join('');
 renderCoverage();
}
function renderCoverage(){
 const m=selected(),tp=Number(el('tp').value),dp=32/tp,v=Number(el('width').value),cap=Math.ceil(m.experts/(v*dp*m.topk));
 const xmax=Math.max(8,cap*5),native=[],uniform=[];
 for(let b=0;b<=xmax;b++){const n=b*dp*v;native.push([b,100*Math.min(m.experts/32,Math.floor(n*m.topk/32))/(m.experts/32)]);uniform.push([b,100*(1-Math.pow(1-m.topk/m.experts,n))]);}
 el('coverage').innerHTML=plot([{points:native,color:'#8756a1',dots:true},{points:uniform,color:'#368269',dots:true}],xmax,105,'Requests / attention-DP group (B)','Expert inventory covered (%)',[{x:cap,label:`Native cap B=${cap}`}]);
 el('coverage-note').innerHTML=`${m.label}: E=${m.experts}, k=${m.topk}. ${dp} DP groups × ${v} positions/request. Native cap: ${m.ncap} token positions in total, rounded up to B=${cap} for this layout. Green curve assumes independent routing across <em>all</em> positions. Lines join integer-B illustrations.`;
}
function renderToy(){
 const bw=Number(el('bw').value),series=[{color:'#b94b20',points:[]},{color:'#2563a6',points:[]},{color:'#20313d',points:[]}];
 for(let b=0;b<=256;b++){const mem=(2*Math.min(b/8,1)+.005*b)/bw,comp=.015*b;[mem,comp,Math.max(mem,comp)+.3+.02*b].forEach((v,i)=>series[i].points.push([b,v]));}
 el('toy').innerHTML=plot(series,256,10,'Illustrative batch B','Illustrative time (ms)',[{x:8,label:'Weight cap',y:31},{x:96,label:'Capacity ceiling',y:49,color:'#647e91'}],96);
 el('bw-value').textContent=bw.toFixed(2)+'×';
 const cross=2/(.015*bw-.005);
 el('toy-note').innerHTML=`Expert-weight reads cap at B=8. MoE compute overtakes its HBM time at B≈${cross.toFixed(1)}—<strong>${cross>96?'beyond':'inside'}</strong> the fixed B=96 capacity ceiling. Other work still adds time after the MoE operation.`;
}
function renderAxes(){
 const n=Number(el('example-n').value),point=x=>({n:x,t:10+x,s:1000/(10+x),q:1000*x/(10+x)});
 const rows=Array.from({length:317},(_,i)=>point(1+i*.25)),chosen=point(n);
 for(const [id,xk,yk,xmax,ymax,xlabel,ylabel] of [
  ['axis-tpot','n','t',80,100,'Concurrent requests / chip','TPOT (ms)'],
  ['axis-rate','n','q',80,1000,'Concurrent requests / chip','Output tokens / chip / second'],
  ['axis-rate-speed','s','q',100,1000,'TPS / user','Output tokens / chip / second'],
  ['axis-capacity-speed','s','n',100,80,'TPS / user','Concurrent requests / chip']]){
  el(id).innerHTML=plot([{points:rows.map(r=>[r[xk],r[yk]]),color:'#2563a6'},{points:[[chosen[xk],chosen[yk]]],color:'#b94b20',end:true}],xmax,ymax,xlabel,ylabel);
 }
 el('example-n-value').textContent=n;
 el('axes-note').innerHTML=`The same point in every panel: <strong>${n} requests/chip</strong> → ${chosen.t.toFixed(1)} ms TPOT → <strong>${chosen.s.toFixed(1)} TPS/user</strong> × ${n} requests/chip = <strong>${chosen.q.toFixed(1)} output tokens/chip/s</strong>. Only the axes change; the underlying performance does not.`;
}
el('expert-table').innerHTML=D.models.map(m=>`<tr><th>${m.label}</th><td>${m.experts} / ${m.topk}</td><td>${m.ncap}</td><td>${Math.ceil(m.ncap/4)} total requests</td><td>${m.n95}</td></tr>`).join('');
el('sources').innerHTML=D.source_refs.map(([p,n])=>`<p><code>${p}</code> — ${n}</p>`).join('');
render();renderToy();renderAxes();
</script></body></html>'''


if __name__ == '__main__':
    main()
