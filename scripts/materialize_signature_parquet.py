"""Flatten cached JSON signature rows into the durable Parquet feature table."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import pandas as pd
def main():
 p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--output',default='results/features/sd15_signature.parquet');a=p.parse_args()
 rows=[]
 for r in json.loads(Path(a.input).read_text()):
  # Caption is retained in the debug cache, never included in feature columns.
  rows.append({k:v for k,v in r.items() if k not in {'features','caption'}} | r['features'])
 df=pd.DataFrame(rows);Path(a.output).parent.mkdir(parents=True,exist_ok=True);df.to_parquet(a.output,index=False);print({'rows':len(df),'columns':len(df.columns),'output':a.output})
if __name__=='__main__':main()
