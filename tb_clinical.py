"""tb_clinical.py - feature engineering + prediction for the TB clinical (structured-data) model.
Written by TB_Clinical_Model.ipynb. Keep it next to tb_clinical_model.joblib."""
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

CORE_SYMPTOMS = ["fever", "night_sweats", "weight_loss", "hemoptysis", "loss_of_appetite"]
ENGINEERED = ["core_symptom_count", "cough_2wk_plus", "underweight"]
YES_NO = {"yes": 1, "y": 1, "true": 1, "1": 1, "no": 0, "n": 0, "false": 0, "0": 0}


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineered features (identical at training and inference time)."""
    df = df.copy()
    for c in CORE_SYMPTOMS + ["bmi", "cough_duration_weeks"]:
        if c not in df:
            df[c] = np.nan
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["core_symptom_count"] = df[CORE_SYMPTOMS].sum(axis=1, min_count=len(CORE_SYMPTOMS))
    df["cough_2wk_plus"] = (df["cough_duration_weeks"] >= 2).astype(float).where(df["cough_duration_weeks"].notna())
    df["underweight"] = (df["bmi"] < 18.5).astype(float).where(df["bmi"].notna())
    return df


class FeatureEngineer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return add_features(pd.DataFrame(X))


_BUNDLE = {}


def load(path=Path(__file__).with_name("tb_clinical_model.joblib")):
    if not _BUNDLE:
        import joblib
        _BUNDLE.update(joblib.load(path))
    return _BUNDLE


def predict_tb(patient: dict, threshold: float = None) -> dict:
    """Raw patient dict -> structured output. `tb_probability` is the value for the future fusion layer."""
    b = load(); meta = b["metadata"]
    df = pd.DataFrame([patient]).astype(object)
    for c in meta["binary_inputs"]:
        if c in df:
            df[c] = df[c].map(lambda v: YES_NO.get(str(v).strip().lower(), np.nan) if pd.notna(v) else np.nan)
    thr = meta["threshold"] if threshold is None else float(threshold)
    p = float(b["pipeline"].predict_proba(df)[:, 1][0])
    missing = [c for c in meta["required_inputs"] if c not in patient or patient[c] in ("", None)]
    return {"prediction": "TB" if p >= thr else "No TB", "tb_probability": round(p, 4),
            "negative_probability": round(1 - p, 4), "threshold": thr, "model_version": meta["model_version"],
            "risk_category": "Low" if p < thr else ("Elevated - refer for GeneXpert" if p < 0.5 else "High - prioritise GeneXpert"),
            "missing_inputs": missing,
            "note": "Screening support on SYNTHETIC training data - not a diagnosis."}
