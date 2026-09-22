"""PASS 2 of the v2 stage-panel extraction: LPIPS-only (S1.1 lpips_ae, S4.2 lpips_roundtrip).  Never touches the
SD1.5 UNet -- reads the image/ae_recon/roundtrip_recon pixel arrays cached by scripts/extract_stage_panel.py
(PASS 1) and merges the two LPIPS scores into the same feature record.  See extract_stage_panel.py's docstring
for why this must be a separate process from any UNet call."""
from __future__ import annotations
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, torch, lpips
from src.features.panel_v2 import lpips_layer_score

def rowkey(r):
    return (r["generator"], r["content_id"], r.get("transform") or "clean", str(r.get("transform_value") or ""))


def main():
    ap = __import__("argparse").ArgumentParser()
    ap.add_argument("--pass1", default="results/stage_decomposition/panel_features_pass1.json")
    ap.add_argument("--output", default="results/stage_decomposition/panel_features.json")
    ap.add_argument("--recon-cache", default="data/stage_panel_cache")
    a = ap.parse_args()
    pass1 = Path(a.pass1); out = Path(a.output); cache = Path(a.recon_cache)
    rows = json.loads(pass1.read_text())
    done = json.loads(out.read_text()) if out.exists() else []
    have = {rowkey(r) for r in done}
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    net = lpips.LPIPS(net="vgg").to(device).eval()
    for i, r in enumerate(rows, 1):
        key = rowkey(r)
        if key in have:
            continue
        cache_key = f"{r['generator']}__{r['content_id']}__{r.get('transform') or 'clean'}__{r.get('transform_value') or ''}"
        npz_path = cache / f"{cache_key}.npz"
        if not npz_path.exists(): npz_path = cache / f"{r['generator']}__{r['content_id']}.npz"  # back-compat: pre-transform-support cache files
        d = np.load(npz_path)
        lpips_ae = lpips_layer_score(net, d["image"], d["ae_recon"].transpose(1, 2, 0), device, torch.float32)
        lpips_rt = lpips_layer_score(net, d["image"], d["roundtrip_recon"].transpose(1, 2, 0), device, torch.float32)
        rec = dict(r); rec["features"] = dict(r["features"]); rec["features"]["lpips_ae"] = lpips_ae; rec["features"]["lpips_roundtrip"] = lpips_rt
        done.append(rec)
        out.write_text(json.dumps(done, indent=1))
        if i % 20 == 0 or i == len(rows): print(i, "/", len(rows), flush=True)
    print("done:", len(done), "rows")

if __name__ == "__main__":
    main()
