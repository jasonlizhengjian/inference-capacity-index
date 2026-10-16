"""CPU-only native replay/search for the four-model 15 October release."""
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path

from . import simulation
from . import current_models as rules
from .topology import penalty_ms

KEYS = ('tp','dp','dcp','etp','ep','pp')


def doctor(reference):
    expected = json.loads((reference/'runtime.json').read_text())
    simulation.PINNED_AISIM = expected['version']
    simulation._sdk()
    import aisimulate._runtime as runtime
    import aisimulate_core
    actual = hashlib.sha256(Path(runtime.__file__).read_bytes()).hexdigest()
    if actual != expected['runtime_sha256']:
        raise RuntimeError('AISim runtime differs from the frozen wheel; use environment/setup_current.py')
    for model, digest in expected['model_config_sha256'].items():
        path = Path(aisimulate_core.__file__).parent/'model_configs'/(model.replace('/','--')+'_config.json')
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise RuntimeError('Bundled model config mismatch: '+model)
    return dict(version=importlib.metadata.version('aisimulate'), runtime_sha256=actual,
                source_commit=expected['source_commit'], bundled_model_hashes_checked=3)


class Native:
    def __init__(self, reference, hardware, model):
        self.ref, self.hw, self.model = reference, hardware, model
        self.records = json.loads((reference/'models.json').read_text())
        self.cache = {}
        if model.startswith('MAI2'):
            rules.install_adapter()

    def layouts(self, phase):
        if self.model in simulation.MODELS:
            return (rules.prefill_layouts(self.hw,self.model) if phase=='prefill'
                    else simulation.enumerate_layouts(self.hw,self.model))
        return (p for p in rules.gqa_layouts(self.hw,self.records[self.model])
                if p['dcp']==1 or (phase=='decode' and self.model=='Qwen-480B'))

    def graph(self, phase, layout):
        key = (phase, *(layout[k] for k in KEYS))
        if key in self.cache:
            return self.cache[key]
        p, hw = layout, self.hw
        if self.model in simulation.MODELS:
            c = simulation.make_config(hw,self.model,**p,systems_path=self.ref/'hardware')
        else:
            c = dict(model=str(self.ref/'models'/self.model),system=hw['profile'],backend='vllm',
                backend_version='0.24.0',tp=p['tp'],pp=p['pp'],attention_dp=p['dp'],dcp=p['dcp'],
                cp_size=1,moe_tp_size=p['etp'],moe_ep_size=p['ep'],enable_shared_layer=False,
                gemm_quant_mode='bfloat16',moe_quant_mode=hw['quant'],fmha_quant_mode='bfloat16',
                kvcache_quant_mode='fp8',comm_quant_mode='half',estimation_mode='op_level',
                database_mode='SOL',fallback_policy='deny',systems_paths=(str(self.ref/'hardware'),),
                estimator_config={'correction':{'enabled':False}})
        nn = 3 if phase=='decode' else 0
        c.update(worker_type=phase,nextn=nn)
        perf,_,build,apply,get = simulation._sdk()
        mc = build(tp_size=p['tp'],pp_size=p['pp'],attention_dp_size=p['dp'],dcp_size=p['dcp'],
            cp_size=1,moe_tp_size=p['etp'],moe_ep_size=p['ep'],**{k:c[k] for k in
            ('gemm_quant_mode','moe_quant_mode','fmha_quant_mode','kvcache_quant_mode','comm_quant_mode')})
        apply(mc,nn);g=get(c['model'],mc,c['backend']);engine=perf.best_available(c)
        ops=g.generation_ops if phase=='decode' else g.context_ops
        hop=penalty_ms([json.loads(o._spec_json()) for o in ops]) if hw.get('hop_overlay') else 0.
        self.cache[key]=(c,g,engine,hop)
        return self.cache[key]

    def memory(self, phase, layout, batch):
        c,g,_,_=self.graph(phase,layout)
        try:
            if self.model.startswith('MAI2'):
                m=rules.mai_memory(c,g,batch,3) if phase=='decode' else rules.mai_prefill_memory(c,g,batch)
            elif phase=='prefill':
                m=rules.prefill_memory(c,g,batch,1024)
            elif self.model=='Qwen-480B':
                m=rules.gqa_memory(c,g,batch,3)
            else:
                m=simulation._memory(c,g,batch)
        except ValueError as exc:
            if 'no KV budget:' in str(exc):return {'fits':False}
            raise
        return {k.replace('GiB','gib'):v for k,v in m.items()}

    def evaluate(self, phase, layout, batch):
        c,g,engine,hop=self.graph(phase,layout)
        mem=self.memory(phase,layout,batch)
        if not mem['fits']:raise ValueError('Candidate exceeds memory admission')
        ops=engine.static_phase_diagnostics(batch_size=batch,context_length=262144+(1024 if phase=='prefill' else 0),
                                            prefix=262144 if phase=='prefill' else 0,prefill=phase=='prefill')
        if any(not math.isfinite(o['latency_ms']) or o['latency_ms']<0 for o in ops):
            raise ValueError('Invalid operation timing')
        ms=sum(o['latency_ms'] for o in ops)+hop
        if ms<=0:raise ValueError('Nonpositive timing')
        chips=layout['tp']*layout['dp']*layout['pp'];requests=batch*layout['dp']
        return dict(hardware=self.hw['name'],model=self.model,phase=phase,**layout,batch=batch,
                    chips=chips,requests=requests,capacity=requests/chips,speed=3000/ms if phase=='decode' else 0.,
                    raw_ms=ms,hop_ms=hop,input_tokens_chip_second=requests*1024*1000/chips/ms if phase=='prefill' else 0.,
                    memory=mem['memory_gib'],**mem)

    def batches(self, phase, layout):
        lo,hi=0,1
        while hi<65536 and self.memory(phase,layout,hi)['fits']:lo,hi=hi,hi*2
        if hi>=65536:raise RuntimeError('Batch guard reached')
        while lo+1<hi:
            mid=(lo+hi)//2
            if self.memory(phase,layout,mid)['fits']:lo=mid
            else:hi=mid
        return lo
