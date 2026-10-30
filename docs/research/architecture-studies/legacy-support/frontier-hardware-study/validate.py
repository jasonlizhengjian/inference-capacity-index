"""Validate generated data and offline artifact without claiming silicon accuracy."""
import csv
import json
import math
import re
import subprocess
from html.parser import HTMLParser
import engine as e

def main():
    results=[]
    expected={'unified_results.csv':48576,'saturation.csv':20240,'topology_spec.csv':4920,
              'width_depth_results.csv':2304,'capacity_slo_boundaries.csv':12144,'parallelism_screen.csv':8096}
    for file in e.OUT.glob('*.csv'):
        rows=list(csv.DictReader(file.open()))
        if file.name in expected:assert len(rows)==expected[file.name],(file,len(rows))
        assert all(r.get('status')!='error' for r in rows),file
        for r in rows:
            if 'memory_gib' in r:
                assert math.isfinite(float(r['memory_gib'])) and float(r['memory_gib'])>=0
            if r.get('phase') in ('prefill','decode') and 'step_ms' in r:
                assert float(r['step_ms'])>0
            if 'world' in r:assert float(r['world'])<=72
        results.append(dict(dataset=file.name,rows=len(rows),passed=True))
    class Links(HTMLParser):
        def __init__(self):super().__init__();self.ids=set();self.links=[]
        def handle_starttag(self,tag,attrs):
            d=dict(attrs)
            if 'id' in d:assert d['id'] not in self.ids;self.ids.add(d['id'])
            if tag=='a':self.links.append(d.get('href',''))
    document=(e.OUT/'FRONTIER_INFERENCE_STUDY.html').read_text();p=Links();p.feed(document)
    for link in p.links:
        if link.startswith('#'):assert link[1:] in p.ids,link
        elif '://' not in link:assert (e.OUT/link).exists(),link
    js=re.search(r'<script>(.*?)</script>',document,re.S).group(1)
    checked=subprocess.run(['node','--check'],input=js,text=True,capture_output=True)
    assert checked.returncode==0,checked.stderr
    results.append(dict(test='HTML internal/download links and JavaScript syntax',passed=True))
    cal=list(csv.DictReader((e.OUT/'calibration_recheck.csv').open()))
    for r in cal:
        if r['adapter_policy']=='legacy':
            assert abs(float(r['updated_step_ms'])-float(r['aic_sol_tpot_ms']))<1e-8
    results.append(dict(test='Legacy calibration numerical reproduction',passed=True))
    e.write_json(e.OUT/'artifact_validation.json',results)
    print(json.dumps(results,indent=2))

if __name__=='__main__':main()
