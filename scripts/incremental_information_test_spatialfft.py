"""Follow-up to E86 (same script/estimator, one more pre-specified-by-audit column set): does the
rich_score_spatial+rich_score_fft family (66 cols -- identified by the E88 feature audit as where aMUSEd's
concentrated signal lives) add information beyond VAE+CIFAR-32, where guidance/eps did not?  No new features."""
from __future__ import annotations
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from sklearn.metrics import roc_auc_score
from src.analysis.evalkit import cols_for
from scripts.incremental_information_test import load_matched, paired_grouped_cv, paired_boot
out = {}
for gen in ["sd15", "amused"]:
    d = load_matched(gen)
    BASE = ["mse256", "mae256", "mse512", "mae512"] + [c for c in d.columns if c.startswith("cifar_")]
    EXT = BASE + cols_for(d, ["rich_score_spatial", "rich_score_fft"])
    a, b, oa, ob = paired_grouped_cv(d, BASE, EXT); md, ci = paired_boot(d, oa, ob)
    out[gen] = {"baseline(VAE+CIFAR)_auroc": a, "extended(+spatial+fft,66cols)_auroc": b, "paired_diff_mean": md, "paired_diff_ci": ci, "adds_information": bool(ci[0] > 0)}
    print(gen, json.dumps(out[gen], indent=1))
json.dump(out, open("results/content_matched/analysis/incremental_information_test_spatialfft.json", "w"), indent=1)
