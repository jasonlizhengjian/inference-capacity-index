"""Three-curve comparison at fixed assumptions; no new native timings."""
from bisect import bisect_left
import json
import math
from analyze import OUT, save, csvout
from agent_index.index import envelope, area

def lookup(points):
    p=envelope(points);xs=[x['speed'] for x in p];ys=[x['capacity'] for x in p]
    return lambda x:ys[i] if (i:=bisect_left(xs,x))<len(xs) else 0.

def functions(g,recipes,de,pe,metric):
    r=g['input_output_ratio']
    best=envelope(g['sol_frontier']+[p for q in recipes for p in q['decode_frontier']])
    pmax=max([g['prefill_rate']]+[q['prefill_rate'] for q in recipes])*pe
    def adjusted(c,s,p):return c/(1+r*s*c/p)
    def recipe_point(q,c,s):
        if q['disaggregated']:
            return min(c*q['decode_chips']/q['allocated_chips'],q['prefill_rate']*pe*q['prefill_chips']/(r*s*q['allocated_chips']))
        return adjusted(c,s,q['prefill_rate']*pe)
    if metric=='capacity':
        get=lookup(best)
        ideal=lambda s:adjusted(get(s/de),s,pmax)
        lookups=[(q,lookup(q['decode_frontier'])) for q in recipes]
        restricted=lambda s:max((recipe_point(q,f(s/de),s) for q,f in lookups),default=0.)
        measured=lookup(g['frontier'])
    else:
        ideal=lookup([dict(x,speed=x['speed']*de,capacity=x['speed']*de*adjusted(x['capacity'],x['speed']*de,pmax)) for x in best])
        restricted=lookup([dict(x,speed=x['speed']*de,capacity=x['speed']*de*recipe_point(q,x['capacity'],x['speed']*de)) for q in recipes for x in q['decode_frontier']])
        measured=lookup(g['throughput_frontier'])
    return measured,restricted,ideal,best,pmax/pe

def integrate(fn,edges,n):
    return math.fsum(math.log(b/a)/n*math.fsum(fn(math.exp(math.log(a)+(j+.5)*math.log(b/a)/n)) for j in range(n)) for a,b in zip(edges,edges[1:]))

def main():
    data=json.loads((OUT/'curves.json').read_text());layouts=json.loads((OUT/'layout_restricted.json').read_text());groups=[];rows=[]
    for g in data['groups']:
        recipes=[q for q in layouts['recipes'] if q['model']==g['model'] and q['hardware']==g['hardware']]
        summaries=[]
        for de,pe in [(.5,.5),(.5,1.),(1.,1.)]:
            for band,lo,hi in [('observed_range',g['observed_min_speed'],g['measured_max_speed']),('index_band',50.,400.)]:
                for metric in ['capacity','output']:
                    measured,restricted,ideal,best,pmax=functions(g,recipes,de,pe,metric)
                    edges=sorted({lo,hi,*(p['speed']*de for p in best if lo<p['speed']*de<hi),*(p['speed']*de for q in recipes for p in q['decode_frontier'] if lo<p['speed']*de<hi)})
                    ma=area(g['frontier'] if metric=='capacity' else g['throughput_frontier'],lo,hi)
                    results=[];errors=[]
                    for fn in [restricted,ideal]:
                        n=16;old=integrate(fn,edges,n)
                        while True:
                            n*=2;val=integrate(fn,edges,n);error=abs(val-old)/max(abs(val),1e-12)
                            if error<1e-6:break
                            if n>=4096:raise ValueError('Quadrature did not converge')
                            old=val
                        results.append(val);errors.append(error)
                    ra,ia=results
                    row=dict(model=g['model'],hardware=g['hardware'],band=band,low=lo,high=hi,decode_efficiency=de,prefill_efficiency=pe,metric=metric,
                        measured_auc=ma,restricted_auc=ra if recipes else None,ideal_auc=ia,
                        measured_over_restricted=ma/ra if ra else None,measured_over_ideal=ma/ia if ia else None,restricted_over_ideal=ra/ia if recipes and ia else None,
                        integration_relative_change=max(errors),restricted_recipes=len(recipes))
                    assert ra<=ia*(1+1e-6),(g['hardware'],ra,ia)
                    summaries.append(row);rows.append(row)
        groups.append(dict(model=g['model'],hardware=g['hardware'],best_frontier=best,best_prefill_rate=pmax,summaries=summaries))
    save('three_curve_comparison.json',dict(groups=groups,description='Observed recipes restricted vs best-known including those recipes, fixed index workload; exact replay unavailable.'))
    csvout('three_curve_summary.csv',rows)
    print('Computed',len(rows),'three-curve comparisons',flush=True)

if __name__=='__main__':main()
