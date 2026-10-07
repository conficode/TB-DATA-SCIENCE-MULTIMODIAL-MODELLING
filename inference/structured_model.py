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
            prep = self.pipeline.named_steps["prep"]
            scaler = prep.named_transformers_["num"].named_steps["scale"]
            num_cols = next(cols for name, _, cols in prep.transformers_ if name == "num")
            self.input_range = {}
            for col, mu, sd in zip(num_cols, scaler.mean_, scaler.scale_):
                if col in NUMERIC_FIELDS:
                    lo, hi = NUMERIC_FIELDS[col][:2]
                    self.input_range[col] = (max(lo, mu - C.CLINICAL_CLIP_SD * sd), min(hi, mu + C.CLINICAL_CLIP_SD * sd))
            log.info("Clinical model loaded: %s", self.meta.get("model"))
        except Exception as e:
            self.error = str(e)
            log.error("Clinical model loading failed: %s", e)

    @property
    def ready(self):
        return self.pipeline is not None

    @property
    def threshold(self):
        return float(C.CLINICAL_THRESHOLD)

    @property
    def version(self):
        return f"{self.meta.get('model_version', '?')} ({self.meta.get('model', '?')})"

    @staticmethod
    def validate(form):
        """Return (clean_inputs, errors). All 14 inputs are required."""
        clean, errors = {}, []
        for f, (lo, hi, label) in NUMERIC_FIELDS.items():
            v = str(form.get(f, "") or "").strip()
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
            v = str(form.get(f, "") or "").strip()
            if v not in ("0", "1"):
                errors.append(f"{label}: please select Yes or No.")
            else:
                clean[f] = int(v)
        hiv = str(form.get("hiv_status", "") or "").strip()
        hiv_norm = hiv.strip().capitalize()
        if hiv_norm not in HIV_VALUES:
            errors.append("HIV status must be Positive or Negative (the model was not trained on 'unknown').")
        else:
            clean["hiv_status"] = hiv_norm
        if clean.get("cough_duration_weeks", 0) == 0 and (clean.get("productive_cough") or clean.get("hemoptysis")):
            errors.append("Productive cough / haemoptysis recorded but cough duration is 0 weeks.")
        return clean, errors

    def _clip(self, inputs):
        return {k: (min(max(v, self.input_range[k][0]), self.input_range[k][1]) if k in self.input_range else v)
                for k, v in inputs.items()}

    def input_notes(self, inputs):
        """Human-readable notes for values outside the training range (they are limited before prediction)."""
        notes = []
        for k, (lo, hi) in self.input_range.items():
            v = inputs.get(k)
            if v is not None and not lo <= v <= hi:
                edge = lo if v < lo else hi
                notes.append(f"{NUMERIC_FIELDS[k][2]} {v:g} is outside the range the clinical model was trained on "
                             f"({lo:.1f}-{hi:.1f}); it was treated as {edge:.1f}.")
        return notes

    def _frame(self, inputs):
        clipped = self._clip(inputs)
        return pd.DataFrame([{k: clipped[k] for k in self.meta["required_inputs"]}])

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
