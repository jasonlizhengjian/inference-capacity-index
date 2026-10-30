"""Re-evaluate existing matched GLM strata with index-reuse-aware decode SOL."""
import csv
import study as s
def main():
    source=s.HERE.parent/'inferencex-agentx-grounding/output/matched_decode_strata.csv'
    rows=list(csv.DictReader(source.open()));est=s.e.Estimator(s.body.HW);out=[]
    model=s.HERE.parent/'inferencex-agentx-grounding/models/glm'
    for r in rows:
        if r['name']!='GLM-5.2':continue
        args=dict(model=model,system='GB300',dep=8,context=int(float(r['context_mean_per_request'])),nextn=3,al=1,quant=None,gemm=None)
        live=est.estimate(cpr=int(float(r['batch_per_rank'])),**args)
        pad=est.estimate(cpr=int(float(r['padded_batch_per_rank'])),**args)
        assert live['status']==pad['status']=='ok'
        hybrid=live['attention_ms']+pad['step_ms']-pad['attention_ms']
        out.append(dict(batch=int(float(r['batch_per_rank'])),padded_batch=int(float(r['padded_batch_per_rank'])),context=args['context'],count=int(float(r['stratum_count'])),
                        observed_ms=float(r['observed_device_ms']),old_sol_ms=float(r['aic_step_ms']),reuse_sol_ms=live['step_ms'],reuse_padded_ms=pad['step_ms'],reuse_hybrid_ms=hybrid))
    s.e.write_csv(s.OUT/'native_recalibration.csv',out)
    denom=sum(r['count'] for r in out)
    summary={k:sum(r[k]*r['count'] for r in out)/denom for k in ('observed_ms','old_sol_ms','reuse_sol_ms','reuse_padded_ms','reuse_hybrid_ms')}
    summary.update(iterations=denom,strata=len(out));s.e.write_json(s.OUT/'native_recalibration.json',summary);print(summary)
if __name__=='__main__':main()
