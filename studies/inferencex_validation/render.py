"""Render the public, offline-capable validation explorer from saved outputs."""
import argparse
import json
from pathlib import Path
import shutil

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
OUT=ROOT/'docs/research/inferencex-validation'

def main(site=None):
    payload={key:json.loads((OUT/file).read_text()) for key,file in
             [('curves','curves.json'),('glm','matched_glm.json'),('kimi','kimi_config_audit.json'),('layouts','layout_restricted.json'),('comparison','three_curve_comparison.json'),('agreement','point_agreement.json')]}
    page=(HERE/'template.html').read_text().replace('__PAYLOAD__',json.dumps(payload,allow_nan=False).replace('</',r'<\/'))
    (OUT/'index.html').write_text(page)
    if site:
        site=Path(site);site.mkdir(parents=True,exist_ok=True)
        for f in OUT.iterdir():
            if f.is_file():shutil.copy2(f,site/f.name)
    print(OUT/'index.html')

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--site-dist');args=ap.parse_args();main(args.site_dist)
