"""Seed a full resumable extraction cache from one completed rich checkpoint."""
from __future__ import annotations
import argparse, json
from pathlib import Path
def main():
 p=argparse.ArgumentParser();p.add_argument('--manifest',required=True);p.add_argument('--checkpoint',required=True);p.add_argument('--output',required=True);a=p.parse_args()
 rows=list(__import__('csv').DictReader(open(a.manifest))); known={r['path'] for r in rows}
 done=json.loads(Path(a.checkpoint).read_text()); paths=[r['path'] for r in done]
 if len(paths)!=len(set(paths)) or not set(paths)<=known: raise SystemExit('checkpoint paths must be unique members of manifest')
 out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(done,indent=2));print({'seeded':len(done),'remaining':len(rows)-len(done),'output':str(out)})
if __name__=='__main__':main()
