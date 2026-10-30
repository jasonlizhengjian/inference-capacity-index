"""Isolate ideal mean-rank versus synthetic uniform-routing critical rank."""
import itertools
import engine as e
import run

def main():
    s=run.Study();rows=[]
    selected=[d for d in s.defs if d.get('id') in ('k3_2p8t','k3_5t_trend','deepseek_v4_pro_5t_expert','glm52_ratio_preserved_5t')]
    for d,dep,cpr,hw in itertools.product(selected,(16,64),(1,4,16),('GB300','VR200-dense')):
        a=s.aic.estimate(d['model'],hw,dep,cpr,occupancy='expected')
        b=s.aic.estimate(d['model'],hw,dep,cpr,occupancy='maxrank')
        assert b['status']=='ok',b
        rows.append(dict(aid=d['aid'],system=hw,dep=dep,cpr=cpr,mean_rank_ms=a['step_ms'],
            rank_tail_ms=b['step_ms'],latency_increase_fraction=b['step_ms']/a['step_ms']-1,
            mean_moe_ms=a['moe_ms'],rank_tail_moe_ms=b['moe_ms'],admit90=a['admit90']))
    e.write_csv(e.OUT/'rank_balance.csv',rows)

if __name__=='__main__':main()
