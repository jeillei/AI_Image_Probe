"""E81: is the guidance-family crossed signal explained by the BLIP caption text itself?
(a) text-only baselines: word count; TF-IDF(1-2gram) logistic.  (b) guidance family after residualising on geometry AND caption length/word-count/TF-IDF-SVD."""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from src.analysis.evalkit import *
from src.analysis.families import COARSE
d = load_features("results/features/stage_b_cross_source_rich.json"); d["log_min_side"] = np.log(d.min_side); d["ncap"] = d.caption.str.split().str.len(); d["nchar"] = d.caption.str.len()
NUI = ["log_min_side", "aspect", "is_square", "is_pow2_side", "is_png"]; gens = sorted(d[d.label == 1].generator.unique()); srcs = sorted(d[d.label == 0].source.unique()); G = cols_for(d, COARSE["guidance"]); rows = []
print("caption words real vs fake:", d.groupby("label").ncap.mean().round(2).to_dict()); print("top fake-vs-real words:")
for rs in srcs:
    for g in gens:
        te = d[(d.generator == g) | ((d.label == 0) & (d.source == rs))]; tr = d.drop(te.index); y = te.label.values; rec = {"held_real": rs, "held_generator": g}
        rec["caption_wordcount_only"] = roc_auc_score(y, make(.1).fit(tr[["ncap", "nchar"]], tr.label).predict_proba(te[["ncap", "nchar"]])[:, 1])
        tv = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True).fit(tr.caption); lr = LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000).fit(tv.transform(tr.caption), tr.label); rec["caption_tfidf_only"] = roc_auc_score(y, lr.predict_proba(tv.transform(te.caption))[:, 1])
        svd = TruncatedSVD(20, random_state=0).fit(tv.transform(tr.caption)); ctr = svd.transform(tv.transform(tr.caption)); cte = svd.transform(tv.transform(te.caption))
        Xtr = np.nan_to_num(tr[G].values); Xte = np.nan_to_num(te[G].values); rec["guidance_raw"] = roc_auc_score(y, make(.1).fit(Xtr, tr.label).predict_proba(Xte)[:, 1])
        for nm, (Ntr, Nte) in {"guidance_resid_geometry": (tr[NUI].values, te[NUI].values), "guidance_resid_geometry+caption": (np.c_[tr[NUI].values, tr[["ncap", "nchar"]].values, ctr], np.c_[te[NUI].values, te[["ncap", "nchar"]].values, cte])}.items():
            sc = StandardScaler().fit(Ntr); r = Ridge(alpha=1.0).fit(sc.transform(Ntr), Xtr); rec[nm] = roc_auc_score(y, make(.1).fit(Xtr - r.predict(sc.transform(Ntr)), tr.label).predict_proba(Xte - r.predict(sc.transform(Nte)))[:, 1])
        rows.append(rec)
r = pd.DataFrame(rows); r.to_csv("results/audit/crossed/caption_baseline_cells.csv", index=False)
s = r.drop(columns=["held_real", "held_generator"]).agg(["mean", "median", "min"]).T; s["frac_above_0.5"] = (r.drop(columns=["held_real", "held_generator"]) > .5).mean(); s.round(3).to_csv("results/audit/crossed/caption_baseline_summary.csv"); print(s.round(3).to_string())
# most fake-indicative caption words
tv = TfidfVectorizer(min_df=3).fit(d.caption); lr = LogisticRegression(C=1, class_weight="balanced", max_iter=2000).fit(tv.transform(d.caption), d.label); w = pd.Series(lr.coef_[0], index=tv.get_feature_names_out()).sort_values(); print("real-indicative:", list(w.index[:12])); print("fake-indicative:", list(w.index[-12:]))
