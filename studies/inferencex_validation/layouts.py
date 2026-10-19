"""Controlled layout/footprint comparison, NOT exact benchmark replay.

Hold index context, precision, nextn3/AL3 and native graph constant; constrain
parallelism and P/D pool counts to submitted recipes. Never use client conc as
batch. Keep unsupported physical-device mappings explicit.
"""
import json
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(HERE),str(HERE.parents[1]/'src')]
from analyze import REF, OUT, HW, snapshots, save
from agent_index.current_native import Native, doctor
from agent_index.index import envelope

def main():
    runtime=doctor(REF);catalog=json.loads((REF/'catalog.json').read_text())
    logs={str(r['result_id']):r for r in json.loads((OUT/'kimi_config_audit.json').read_text())['records']}
    engines={};cache={};recipes=[];excluded=[]
    def run(model,hw,phase,layout):
        key=(model,hw,phase,tuple(layout.items()))
        if key not in cache:
            engine=engines.setdefault((model,hw),Native(REF,catalog[hw],model))
            n=engine.batches(phase,layout)
            cache[key]=[engine.evaluate(phase,layout,b) for b in range(1,n+1)]
        return cache[key]
    def layout(r,phase,dcp):
        world=int(r[phase+'_tp']);dp=world if r[phase+'_dp_attention'] else 1
        ep=int(r[phase+'_ep'])
        if world%ep:raise ValueError('Nonintegral expert TP')
        return dict(tp=world//dp,dp=dp,dcp=dcp,etp=world//ep,ep=ep,pp=1)
    seen=set()
    for model,rows in snapshots().items():
        for r in rows:
            if r['benchmark_type']!='agentic_traces' or r['hardware'] not in HW:continue
            hw=HW[r['hardware']];m=r['metrics'];notes=[]
            if r['hardware']=='tpuv7':
                excluded.append(dict(result_id=r['id'],hardware=hw,model=model,reason='Logical TP32 vs 16 physical chips per pool: no audited native logical-device mapping.'));continue
            dc=int(m.get('decode_dcp_size',m.get('dcp_size',1)))
            if not r['disagg']:dc=int(m.get('prefill_dcp_size',dc))
            if str(r['id']) in logs:dc=logs[str(r['id'])]['dcp'];notes.append('Decode DCP verified from startup log.')
            else:notes.append('Parallelism from API; unverified against full startup log.')
            try:
                dl=layout(r,'decode',dc);pl=layout(r,'prefill',1)
                nd=int(r['num_decode_gpu']);np=int(r['num_prefill_gpu']);world=dl['tp']*dl['dp'];pw=pl['tp']*pl['dp']
                if nd%world or np%pw:raise ValueError('Pool count not divisible by execution world')
                key=(model,hw,tuple(dl.items()),tuple(pl.items()),nd,np,bool(r['disagg']))
                if key in seen:continue
                seen.add(key)
                ds=run(model,hw,'decode',dl);ps=run(model,hw,'prefill',pl)
                if not ds or not ps:raise ValueError('No admitted 256Ki request under the fixed index precision/memory rules')
                prefill=max(ps,key=lambda p:p['input_tokens_chip_second'])
                recipes.append(dict(model=model,hardware=hw,representative_result_id=r['id'],decode_layout=dl,prefill_layout=pl,
                    decode_chips=nd,prefill_chips=np if r['disagg'] else 0,allocated_chips=nd+np if r['disagg'] else max(nd,np),
                    disaggregated=r['disagg'],prefill_rate=prefill['input_tokens_chip_second'],prefill_winner=prefill,
                    decode_frontier=envelope(ds),decode_candidates=len(ds),notes=notes))
                print(model,hw,r['id'],dl,'batches',len(ds),flush=True)
            except (ValueError,RuntimeError) as e:
                excluded.append(dict(result_id=r['id'],hardware=hw,model=model,reason=str(e)))
    save('layout_restricted.json',dict(runtime=runtime,context=262144,new_input_tokens=1024,nextn=3,al=3,
        scope='Observed deployment layouts and pool counts, optimized admitted batches; fixed index workload/precision/graph. Not exact iso-config replay.',
        recipes=recipes,excluded=excluded))
    print('Saved',len(recipes),'recipes;',len(excluded),'exclusions',flush=True)

if __name__=='__main__':main()
