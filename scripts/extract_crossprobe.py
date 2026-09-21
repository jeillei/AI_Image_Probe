"""Resumable cross-probe extraction (class-B quantities) for a probe on a manifest.  Output records mirror v1."""
from __future__ import annotations
import argparse, csv, json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from src.probes.base import cross_probe_features
def main():
    p = argparse.ArgumentParser(); p.add_argument("--probe", default="cifar32"); p.add_argument("--manifest", default="data/stage_b_cross_source/manifest.csv"); p.add_argument("--output"); p.add_argument("--steps", type=int, default=6)
    p.add_argument("--batch", type=int, default=16); p.add_argument("--limit", type=int); p.add_argument("--device", default="cpu"); a = p.parse_args()
    if a.probe == "dit":
        from src.probes.dit import DiTProbe; probe = DiTProbe(steps=a.steps, device=a.device)
    elif a.probe == "cifar32":
        from src.probes.pixel_ddpm import PixelDDPMProbe; probe = PixelDDPMProbe(steps=a.steps, device=a.device)
    else:
        from src.probes.pixel_ddpm import PixelDDPMProbe
        ids = {"church256": "google/ddpm-ema-church-256", "bedroom256": "google/ddpm-ema-bedroom-256", "celebahq256": "google/ddpm-ema-celebahq-256"}
        probe = PixelDDPMProbe(ids[a.probe], steps=a.steps, size=256, device=a.device, tag=a.probe)
    rows = list(csv.DictReader(open(a.manifest)))[: a.limit]; out = Path(a.output or f"results/crossprobe/{probe.name}.json"); out.parent.mkdir(parents=True, exist_ok=True)
    done = json.load(open(out)) if out.exists() else []; have = {r["path"] for r in done}; todo = [r for r in rows if r["path"] not in have]
    for s in range(0, len(todo), a.batch):
        g = todo[s:s + a.batch]; t = time.time(); xs = [probe.preprocess(r["path"]) for r in g]; res = probe.measure(xs)
        for r, x, y in zip(g, xs, res): done.append({"path": r["path"], "label": int(r["label"]), "generator": r["generator"], "probe": probe.name, "protocol": probe.protocol_version, "features": cross_probe_features(y, x)})
        out.write_text(json.dumps(done)); print(len(done), "/", len(rows), round((time.time() - t) / len(g), 2), "s/img", flush=True)
main()
