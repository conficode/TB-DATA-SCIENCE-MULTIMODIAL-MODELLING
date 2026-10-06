"""Controlled evaluation of CNN alone vs clinical alone vs fusion, on CLINICIAN-CONFIRMED cases.

Run manually (never automatically):  python scripts/evaluate_fusion.py
* Uses only cases with a confirmed outcome of 'TB confirmed' or 'TB excluded'.
* Splits them 50/50 (stratified) into a VALIDATION part (choose fusion weight + threshold)
  and a TEST part (reported once, untouched by any choice).
"""
import sqlite3, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

MIN_CASES = 60

def metrics(y, p, t):
    yh = (p >= t).astype(int); tn, fp, fn, tp = confusion_matrix(y, yh, labels=[0, 1]).ravel()
    return {"threshold": round(t, 3), "accuracy": accuracy_score(y, yh), "precision": precision_score(y, yh, zero_division=0),
            "sensitivity": recall_score(y, yh), "specificity": tn / (tn + fp), "f1": f1_score(y, yh), "roc_auc": roc_auc_score(y, p),
            "confusion [TN FP FN TP]": [int(tn), int(fp), int(fn), int(tp)]}

def thr_for_sensitivity(y, p, target=0.90):
    f, t, th = roc_curve(y, p); return float(np.clip(th, 0, 1)[t >= target].max())

con = sqlite3.connect(C.DATABASE_PATH)
rows = con.execute("""SELECT p.cnn_probability, p.clinical_probability, o.outcome FROM predictions p
                      JOIN confirmed_outcomes o ON o.case_id = p.case_id WHERE o.outcome IN ('TB confirmed','TB excluded')""").fetchall()
if len(rows) < MIN_CASES:
    sys.exit(f"Only {len(rows)} confirmed cases - need at least {MIN_CASES} for a meaningful validation/test evaluation.")
cnn, clin = np.array([r[0] for r in rows]), np.array([r[1] for r in rows])
y = np.array([1 if r[2] == "TB confirmed" else 0 for r in rows])
if y.min() == y.max():
    sys.exit("Confirmed cases contain only one class - cannot evaluate.")
idx_val, idx_test = train_test_split(np.arange(len(y)), test_size=0.5, stratify=y, random_state=42)

# 1) choose the fusion weight on VALIDATION only (maximise ROC-AUC)
grid = np.round(np.arange(0, 1.01, 0.05), 2)
aucs = [roc_auc_score(y[idx_val], w * cnn[idx_val] + (1 - w) * clin[idx_val]) for w in grid]
w = float(grid[int(np.argmax(aucs))])
print(f"Validation-selected CNN weight = {w} (clinical weight = {1 - w:.2f}); validation AUC = {max(aucs):.3f}")
fused = w * cnn + (1 - w) * clin

# 2) thresholds chosen on VALIDATION (>= 90 % sensitivity), 3) report on TEST once
for name, p in [("CNN alone", cnn), ("Clinical alone", clin), ("Fusion", fused)]:
    t = thr_for_sensitivity(y[idx_val], p[idx_val])
    m = metrics(y[idx_test], p[idx_test], t)
    print(f"\n{name}:"); [print(f"  {k:24s} {v if isinstance(v, list) else round(v, 3)}") for k, v in m.items()]
print("\nIf the fusion weight differs clearly from 0.5 and test results support it, update FUSION_WEIGHTS in config.py.")
