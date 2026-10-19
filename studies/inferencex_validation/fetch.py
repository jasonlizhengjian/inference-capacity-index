"""Snapshot public benchmark metadata; never infer live batch from client concurrency."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import urllib.parse
import urllib.request

HERE = Path(__file__).resolve().parent

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,default=HERE/'evidence');args=ap.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    manifest=[]
    for model in ('GLM-5.2','Kimi-K3'):
        url='https://inferencex.semianalysis.com/api/v1/benchmarks?'+urllib.parse.urlencode({'model':model})
        with urllib.request.urlopen(url,timeout=90) as response:
            body=response.read();etag=response.headers.get('ETag')
        data=json.loads(body);assert isinstance(data,list)
        name=model+'.json';(args.output/name).write_bytes(body)
        manifest.append(dict(file=name,url=url,retrieved_at=datetime.now(timezone.utc).isoformat(),sha256=hashlib.sha256(body).hexdigest(),etag=etag,rows=len(data)))
        print(model,len(data),Counter((r['hardware'],r.get('benchmark_type')) for r in data),flush=True)
        for hw in ('gb300','b300','mi355x','tpuv7'):
            rows=[r for r in data if r['hardware']==hw]
            if rows:
                print(hw,'dates',sorted({r['date'] for r in rows}),'shapes',Counter((r['framework'],r['decode_tp'],r['decode_ep'],r.get('decode_dp_attention'),r.get('spec_method'),r['disagg']) for r in rows),'example',rows[0]['id'],flush=True)
    (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')

if __name__=='__main__':main()
