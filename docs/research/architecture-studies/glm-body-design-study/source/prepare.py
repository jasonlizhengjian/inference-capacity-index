"""Emit controlled model definitions/configs for installation with apply_patch."""
import json
from pathlib import Path
HERE=Path(__file__).resolve().parent
BASE=json.loads((HERE.parent/'inferencex-agentx-grounding/models/glm/config.json').read_text())
def prepare():
    defs=[];configs={}
    def add(family,target,L=78,H=6144,M=2048,E=None,K=None,S=1,tag=''):
        E=E or max(64,round(target*1e12/(3*H*M*L)/64)*64)
        K=K or max(1,round(E/32))
        ident=f'{family}_T{target}_L{L}_H{H}_M{M}_E{E}_K{K}_S{S}'+tag
        cfg=dict(BASE)
        cfg.pop('quantization_config',None)
        cfg.update(num_hidden_layers=L,hidden_size=H,moe_intermediate_size=M,n_routed_experts=E,num_experts=E,num_experts_per_tok=K,n_shared_experts=S,
                   intermediate_size=2*H,first_k_dense_replace=0,indexer_types=None,index_topk_pattern=None,layer_types=['sparse']*L)
        # Per-layer DSA internal geometry is fixed. Only residual projections depend on H.
        per_expert=3*H*M
        routed=L*E*per_expert;active=L*K*per_expert
        full=sum(i<3 or (i-2)%4==0 for i in range(L))
        attn=L*(H*2048+2048*64*256+H*576+512*64*448+64*256*H)+full*(2048*32*128+H*32)
        other=attn+2*H*154880+L*(3*H*M*S+H*E)
        defs.append(dict(id=ident,family=family,target_t=target,layers=L,hidden=H,expert_width=M,experts=E,active_experts=K,shared_experts=S,
                         sparsity_fraction=K/E,expert_params_m=per_expert/1e6,routed_total_t=routed/1e12,routed_active_b=active/1e9,
                         approximate_total_t=(routed+other)/1e12,approximate_active_b=(active+other)/1e9,
                         model=str(HERE/'models'/ident),dsa_dims={'hidden_size':H}))
        configs[ident]=cfg
    for t in (3,5,10):
        for k in (8,16,32,64,128):add('sparsity',t,K=k)
        base=max(128,round(t*1e12/(3*6144*2048*78)/128)*128)
        # Exact routed total and active budgets across expert granularity. No shared experts avoids a changing shared-expert budget.
        for m,f in ((1024,2),(2048,1),(4096,.5),(8192,.25)):
            add('granularity',t,M=m,E=int(base*f),K=int(base*f/32),S=0)
        for s in (0,1,2,4):add('shared',t,S=s)
    for t in (5,10):
        for l in (48,64,78,96,128):
            for h in (4096,6144,8192,12288):
                experts=1792 if t==5 else 3584
                m=max(128,round(t*1e12/(3*h*l*experts)/128)*128)
                add('geometry',t,L=l,H=h,M=m,E=experts,K=experts//32)
    for t in (1,3,5,10):
        add('capacity_fixed_active',t,K=8)
        add('capacity_fixed_ratio',t)
    return {'definitions':defs,'configs':configs}
if __name__=='__main__':print(json.dumps(prepare()))
