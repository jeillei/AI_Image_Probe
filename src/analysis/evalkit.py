"""Small, dependency-light evaluation helpers.  All classifiers are
StandardScaler+LogisticRegression with a FIXED C (no per-cell tuning) so that
differences between experiments are attributable to the representation."""
from __future__ import annotations
import json, warnings
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from .families import family
warnings.filterwarnings("ignore")
C_DEFAULT = 0.1

def load_features(path: str, meta_csv: str = "results/audit/file_metadata.csv") -> pd.DataFrame:
    rows = json.load(open(path))
    d = pd.DataFrame([{"path": r["path"], "label": r["label"], "generator": r["generator"], "caption": r.get("caption", ""),
                       **r["features"]} for r in rows])
    if meta_csv:
        m = pd.read_csv(meta_csv).drop(columns=["label", "generator"], errors="ignore")
        d = d.merge(m, on="path", how="left")
    d["source"] = np.where(d.label == 0, d.get("real_source", pd.Series(index=d.index, dtype=object)).fillna("real"), d.generator)
    return d

def feature_cols(d: pd.DataFrame) -> list[str]:
    return [c for c in d.columns if family(c) != "unassigned"]

def cols_for(d, families): 
    fs = set(families); return [c for c in feature_cols(d) if family(c) in fs]

def make(c=C_DEFAULT):
    return make_pipeline(StandardScaler(), LogisticRegression(C=c, max_iter=5000, class_weight="balanced", random_state=17))

def boot_auc(y, p, n=1000, seed=0):
    y = np.asarray(y); p = np.asarray(p); rng = np.random.default_rng(seed); pos = np.where(y == 1)[0]; neg = np.where(y == 0)[0]; out = []
    if len(pos) == 0 or len(neg) == 0: return (np.nan, np.nan)
    for _ in range(n):
        i = np.r_[rng.choice(pos, len(pos)), rng.choice(neg, len(neg))]; out.append(roc_auc_score(y[i], p[i]))
    return tuple(np.percentile(out, [2.5, 97.5]))

def _prep(X):
    X = np.asarray(X, float); return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

def cv_binary(X, y, c=C_DEFAULT, reps=5, folds=5, seed=0):
    """Repeated stratified CV; returns per-rep AUROC list and mean OOF score."""
    X = _prep(X); y = np.asarray(y); aucs = []; oof = np.zeros(len(y))
    for r in range(reps):
        p = np.zeros(len(y))
        for tr, te in StratifiedKFold(folds, shuffle=True, random_state=seed + r).split(X, y):
            p[te] = make(c).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
        aucs.append(roc_auc_score(y, p)); oof += p / reps
    return np.array(aucs), oof

def cv_multiclass(X, y, c=C_DEFAULT, reps=5, folds=5, seed=0):
    X = _prep(X); y = np.asarray(y); classes = np.unique(y); bal = []; conf = np.zeros((len(classes), len(classes)))
    for r in range(reps):
        pred = np.empty(len(y), dtype=object)
        for tr, te in StratifiedKFold(folds, shuffle=True, random_state=seed + r).split(X, y):
            m = make(c).fit(X[tr], y[tr]); pred[te] = m.predict(X[te])
        bal.append(balanced_accuracy_score(y, pred)); conf += confusion_matrix(y, pred, labels=classes)
    conf = conf / conf.sum(1, keepdims=True)
    return np.array(bal), conf, classes

def thresholds_from_oof(X, y, c, qs=(.01, .05), seed=0):
    """Score thresholds at the 1%/5% FPR of OUT-OF-FOLD scores of the training reals."""
    aucs, oof = cv_binary(X, y, c, reps=1, seed=seed); neg = np.sort(oof[y == 0])
    return {q: float(neg[max(0, int(np.ceil((1 - q) * len(neg))) - 1)]) for q in qs}

def tpr_fpr(y, p, thr):
    y = np.asarray(y); p = np.asarray(p); o = {}
    for q, t in thr.items():
        o[f"tpr@{int(q*100)}"] = float(np.mean(p[y == 1] >= t)); o[f"realized_fpr@{int(q*100)}"] = float(np.mean(p[y == 0] >= t))
    return o

def crossed_matrix(d, cols, real_sources, generators, c=C_DEFAULT, n_boot=500):
    """Rows: held-out real source; columns: held-out generator.  Neither is seen in training."""
    res = []
    for rs in real_sources:
        for g in generators:
            te = d[(d.generator == g) | ((d.label == 0) & (d.source == rs))]; tr = d.drop(te.index)
            m = make(c).fit(_prep(tr[cols]), tr.label.values); p = m.predict_proba(_prep(te[cols]))[:, 1]; y = te.label.values
            lo, hi = boot_auc(y, p, n_boot); thr = thresholds_from_oof(_prep(tr[cols]), tr.label.values, c)
            res.append({"held_real": rs, "held_generator": g, "auroc": float(roc_auc_score(y, p)), "ci_lo": lo, "ci_hi": hi,
                        "n_pos": int(y.sum()), "n_neg": int((1 - y).sum()), **tpr_fpr(y, p, thr)})
    return pd.DataFrame(res)

def summarize_matrix(m: pd.DataFrame) -> dict:
    a = m.auroc.values; return {"macro_mean": float(a.mean()), "median": float(np.median(a)), "worst": float(a.min()),
        "frac_above_0.5": float(np.mean(a > .5)), "frac_ci_lo_above_0.5": float(np.mean(m.ci_lo > .5)), "frac_ci_hi_below_0.5": float(np.mean(m.ci_hi < .5)),
        "mean_tpr@1": float(m["tpr@1"].mean()), "mean_realized_fpr@1": float(m["realized_fpr@1"].mean()),
        "mean_tpr@5": float(m["tpr@5"].mean()), "mean_realized_fpr@5": float(m["realized_fpr@5"].mean())}
