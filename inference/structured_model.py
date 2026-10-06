"""Structured (clinical) model inference + exact linear SHAP explanation.

The saved pipeline (tb_clinical_model.joblib) already contains feature engineering, imputation,
standardisation, one-hot encoding and the logistic regression, so inputs are passed RAW, exactly
as in TB_Clinical_Model.ipynb. Nothing is re-implemented here (no training-serving skew)."""
import json
import logging
import joblib
import numpy as np
import pandas as pd
import config as C

log = logging.getLogger(__name__)

# Input specification taken from the saved model metadata (required_inputs) + notebook valid ranges
NUMERIC_FIELDS = {"age": (15, 100, "Age (years)"), "bmi": (10, 60, "BMI (kg/m²)"),
                  "cough_duration_weeks": (0, 52, "Cough duration (weeks)"),
                  "temperature_c": (34.0, 42.5, "Temperature (°C)"), "spo2": (60, 100, "SpO₂ (%)")}
BINARY_FIELDS = {"fever": "Fever", "night_sweats": "Night sweats", "weight_loss": "Weight loss",
                 "hemoptysis": "Haemoptysis", "loss_of_appetite": "Loss of appetite",
                 "productive_cough": "Productive cough", "fatigue": "Fatigue",
                 "household_tb_contact": "Household TB contact"}
HIV_VALUES = ["Negative", "Positive"]     # the only levels seen in training
PRETTY = {"core_symptom_count": "Core symptom count", "bmi": "BMI", "cough_duration_weeks": "Cough duration",
          "age": "Age", "temperature_c": "Temperature", "spo2": "SpO₂", "household_tb_contact": "Household TB contact",
          "fever": "Fever", "productive_cough": "Productive cough", "weight_loss": "Weight loss",
          "fatigue": "Fatigue", "hiv_status_Positive": "HIV positive"}


class ClinicalModel:
    def __init__(self, path=C.CLINICAL_MODEL_PATH):
        self.pipeline, self.meta, self.error = None, {}, None
        try:
            if not path.exists():
                raise FileNotFoundError(f"Clinical model not found at {path}. Copy tb_clinical_model.joblib into models/structured/.")
            bundle = joblib.load(path)            # requires tb_clinical.py in the project root
            self.pipeline, self.meta = bundle["pipeline"], bundle["metadata"]
            bg = json.loads(C.SHAP_BACKGROUND_PATH.read_text())
            self.bg_mean = np.array(bg["background_mean"])
            self.feature_names = bg["feature_names"]
            log.info("Clinical model loaded: %s", self.meta.get("model"))
        except Exception as e:
            self.error = str(e)
            log.error("Clinical model loading failed: %s", e)

    @property
    def ready(self):
        return self.pipeline is not None

    @property
    def threshold(self):
        return float(self.meta.get("threshold", 0.147))

    @property
    def version(self):
        return f"{self.meta.get('model_version', '?')} ({self.meta.get('model', '?')})"

    @staticmethod
    def validate(form):
        """Return (clean_inputs, errors). All 14 inputs are required."""
        clean, errors = {}, []
        for f, (lo, hi, label) in NUMERIC_FIELDS.items():
            v = (form.get(f) or "").strip()
            if v == "":
                errors.append(f"{label} is required.")
                continue
            try:
                x = float(v)
            except ValueError:
                errors.append(f"{label} must be a number.")
                continue
            if not lo <= x <= hi:
                errors.append(f"{label} must be between {lo} and {hi}.")
            clean[f] = x
        for f, label in BINARY_FIELDS.items():
            v = form.get(f)
            if v not in ("0", "1"):
                errors.append(f"{label}: please select Yes or No.")
            else:
                clean[f] = int(v)
        hiv = form.get("hiv_status")
        if hiv not in HIV_VALUES:
            errors.append("HIV status must be Positive or Negative (the model was not trained on 'unknown').")
        else:
            clean["hiv_status"] = hiv
        if clean.get("cough_duration_weeks", 0) == 0 and (clean.get("productive_cough") or clean.get("hemoptysis")):
            errors.append("Productive cough / haemoptysis recorded but cough duration is 0 weeks.")
        return clean, errors

    def _frame(self, inputs):
        return pd.DataFrame([{k: inputs[k] for k in self.meta["required_inputs"]}])

    def predict(self, inputs):
        return float(self.pipeline.predict_proba(self._frame(inputs))[:, 1][0])

    def explain(self, inputs, top=8):
        """Exact SHAP for logistic regression (interventional, mean background):
        contribution_j = coef_j * (x_j - mean_j) in log-odds. Equivalent to shap.LinearExplainer."""
        df = self._frame(inputs)
        xt = self.pipeline[:-1].transform(df)[0]
        model = self.pipeline.named_steps["model"]
        coef = model.coef_[0]
        contrib = coef * (xt - self.bg_mean)
        base_logit = float(model.intercept_[0] + coef @ self.bg_mean)
        engineered = self.pipeline.named_steps["features"].transform(df).iloc[0]
        rows = []
        for name, c in zip(self.feature_names, contrib):
            raw = name.split("__", 1)[1]
            key = "hiv_status" if raw.startswith("hiv_status") else raw
            val = engineered.get(key, None)
            if key in BINARY_FIELDS:
                val = "Yes" if val == 1 else "No"
            elif isinstance(val, (int, float, np.floating)):
                val = f"{val:g}"
            rows.append({"feature": PRETTY.get(raw, raw), "value": val, "contribution": float(c)})
        rows.sort(key=lambda r: abs(r["contribution"]), reverse=True)
        mx = max(abs(r["contribution"]) for r in rows) or 1
        for r in rows:
            r["width"] = round(abs(r["contribution"]) / mx * 100, 1)
        return {"base_probability": float(1 / (1 + np.exp(-base_logit))), "rows": rows[:top]}
