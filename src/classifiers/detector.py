"""Train and apply regularized generative-feature detectors."""
from __future__ import annotations
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingClassifier

def fit_logistic(x,y,c=1.0):
    return make_pipeline(StandardScaler(),LogisticRegression(C=c,max_iter=5000,class_weight="balanced",random_state=17)).fit(x,y)
def fit_boosted(x,y):
    return HistGradientBoostingClassifier(max_depth=2,l2_regularization=1.,max_iter=80,random_state=17).fit(x,y)
def score(model,x): return model.predict_proba(x)[:,1]
