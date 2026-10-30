"""Generate attention one-variable controls and paired coordinated scale-ups."""
import copy
import json
from pathlib import Path
HERE=Path(__file__).resolve().parent
BASE=json.loads((HERE.parent/'inferencex-agentx-grounding/models/glm/config.json').read_text())
DEFAULT=dict(heads=64,q_lora_rank=2048,kv_lora_rank=512,qk_nope_head_dim=192,qk_rope_head_dim=64,v_head_dim=256,index_n_heads=32,index_head_dim=128)
def prepare():
    defs=[];configs={}
    def add(t,family,profile,L=78,H=6144,M=2048,**change):
        dims=dict(DEFAULT,**change);E=1728 if t==5 else 3392;K=E//32
        ident=f'T{t}_{family}_{profile}_L{L}_H{H}'
        cfg=copy.deepcopy(BASE);cfg.pop('quantization_config',None)
        cfg.update({k:v for k,v in dims.items() if k!='heads'})
        cfg.update(hidden_size=H,num_hidden_layers=L,num_attention_heads=dims['heads'],num_key_value_heads=dims['heads'],head_dim=dims['qk_nope_head_dim'],qk_head_dim=dims['qk_nope_head_dim']+dims['qk_rope_head_dim'],
                   moe_intermediate_size=M,n_routed_experts=E,num_experts=E,num_experts_per_tok=K,n_shared_experts=1,first_k_dense_replace=0,intermediate_size=2*H,
                   indexer_types=None,index_topk_pattern=None,layer_types=['sparse']*L)
        n,q,kv,nope,rope,v,ih,idx=[dims[k] for k in DEFAULT]
        routed=3*L*H*M*E;active=3*L*H*M*K
        full=sum(i<3 or (i-2)%4==0 for i in range(L))
        attn=L*(H*q+q*n*(nope+rope)+H*(kv+rope)+kv*n*(nope+v)+n*v*H)+full*(q*ih*idx+H*ih)
        other=attn+2*H*154880+L*(3*H*M+H*E)
        defs.append(dict(id=ident,family=family,profile=profile,target_t=t,layers=L,hidden=H,expert_width=M,experts=E,active_experts=K,shared_experts=1,
                         routed_total_t=routed/1e12,routed_active_b=active/1e9,approximate_total_t=(routed+other)/1e12,approximate_active_b=(active+other)/1e9,
                         attention_params_b=attn/1e9,kv_bytes_per_token=L*(kv+2*rope+idx+4*((idx+127)//128)),full_index_layers=full,
                         model=str(HERE/'models'/ident),dsa_dims=dict(hidden_size=H,index_topk=2048,**{k:v for k,v in dims.items() if k!='heads'}),**dims))
        configs[ident]=cfg
    for t in (5,10):
        add(t,'one_axis','baseline')
        for field,values in [('heads',(32,128)),('q_lora_rank',(1024,4096)),('kv_lora_rank',(256,1024)),('qk_nope_head_dim',(96,384)),('v_head_dim',(128,512)),('index_n_heads',(16,64)),('index_head_dim',(64,256))]:
            for value in values:add(t,'one_axis',f'{field}_{value}',**{field:value})
        add(t,'bundle','compact',heads=32,q_lora_rank=1024,kv_lora_rank=256,qk_nope_head_dim=128,v_head_dim=128,index_n_heads=16,index_head_dim=64)
        add(t,'bundle','expanded',heads=128,q_lora_rank=4096,kv_lora_rank=1024,qk_nope_head_dim=256,v_head_dim=512,index_n_heads=64,index_head_dim=256)
        for L in (64,78,96):
            for H in (8192,12288):
                E=1728 if t==5 else 3392
                M=max(128,round(t*1e12/(3*H*L*E)/128)*128)
                add(t,'joint','fixed_inner',L=L,H=H,M=M)
                scale=H/6144
                add(t,'joint','scaled_inner',L=L,H=H,M=M,heads=round(64*scale/32)*32,q_lora_rank=round(2048*scale/512)*512,
                    kv_lora_rank=round(512*scale/256)*256,index_n_heads=round(32*scale/16)*16)
    return dict(definitions=defs,configs=configs)
if __name__=='__main__':print(json.dumps(prepare()))
