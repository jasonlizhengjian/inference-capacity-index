"""Cache public log evidence locally; raw logs are not publication inputs."""
from concurrent.futures import ThreadPoolExecutor
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import urllib.parse
import urllib.request

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/inferencex-validation-2026-10-19/raw_logs'
IDS=(444604,444605,444607,444602,445227,445218)
BASE='https://inferencex.semianalysis.com/api/v1/'

def get(path,query):
    url=BASE+path+'?'+urllib.parse.urlencode(query)
    for attempt in range(3):
        try:
            with urllib.request.urlopen(url,timeout=45) as r:return json.load(r),url
        except Exception:
            if attempt==2:raise

def fetch(id):
    out=OUT/str(id);out.mkdir(parents=True,exist_ok=True)
    files,url=get('server-log-files',{'id':id})
    (out/'files.json').write_text(json.dumps(files,indent=2)+'\n')
    names=[f for f in files if ('decode_w' in f or 'agg_w' in f)]
    chosen=names[:1]+[next(f for f in files if Path(f).name=='benchmark.out')]
    manifest=[]
    for name in chosen:
        target=out/Path(name).name
        if target.exists() and (out/(target.name+'.meta.json')).exists():
            manifest.append(json.loads((out/(target.name+'.meta.json')).read_text()));continue
        chunks=[];offset=0;complete=False
        for _ in range(100):
            d,link=get('server-log',{'id':id,'file':name,'limit':262144,'offset':offset})
            assert d['fileName']==name,(name,d['fileName'])
            body=d.get('serverLog','');chunks.append(body)
            nxt=d.get('nextOffset')
            if not body or nxt is None:complete=True;break
            assert nxt>offset
            offset=nxt
        raw=''.join(chunks).encode();target.write_bytes(raw)
        m=dict(result_id=id,file=name,url=BASE+'server-log?'+urllib.parse.urlencode({'id':id,'file':name}),retrieved_at=datetime.now(timezone.utc).isoformat(),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest(),complete=complete)
        (out/(target.name+'.meta.json')).write_text(json.dumps(m,indent=2)+'\n');manifest.append(m)
        print(id,name,len(raw),'complete',complete,flush=True)
    return manifest

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--ids',type=int,nargs='+',default=IDS);args=ap.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    with ThreadPoolExecutor(max_workers=3) as pool:result=list(pool.map(fetch,args.ids))
    (OUT/'manifest.json').write_text(json.dumps([r for rows in result for r in rows],indent=2)+'\n')
