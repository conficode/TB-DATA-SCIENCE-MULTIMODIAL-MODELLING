# TB-DATA-SCIENCE-MULTIMODIAL-MODELLING
this contains TB classification and prediction multimodal system

LungLens is an **AI-assisted screening / decision-support prototype** that combines a chest X-ray CNN and a clinical
(structured-data) model. It is **not** a diagnostic system: every result requires clinical confirmation.

---
## 1. What I found when inspecting your models

| | CNN (`tb_xray_best.keras`) | Clinical model (`tb_clinical_model.joblib`) |
|---|---|---|
| **Input** | 224 × 224 × 3 RGB image, raw pixels 0–255 | 14 raw inputs: age, bmi, cough_duration_weeks, temperature_c, spo2, hiv_status, fever, night_sweats, weight_loss, hemoptysis, loss_of_appetite, productive_cough, fatigue, household_tb_contact |
| **Preprocessing** | decode as 3-channel RGB → **bilinear** resize to 224×224 (exactly `image_dataset_from_directory`). `Rescaling(1/255)` is **inside** the model; augmentation layers are inactive at inference | All inside the saved pipeline: `FeatureEngineer` (core-symptom count etc.) → median/mode imputation → `StandardScaler` → `OneHotEncoder` → `LogisticRegression` |
| **Output** | sigmoid = **P(TB)** (classes alphabetical: NORMAL = 0, TB = 1) | **P(TB)**, well calibrated (~30 % base rate) |
| **Threshold used** | **0.55**: your validation table, highest threshold keeping sensitivity 0.990 (spec. 0.978) | **0.147**: ≥ 90 % sensitivity rule from the clinical notebook |
| **Test performance** | Acc 0.954 · Sens 0.943 · Spec 0.963 · AUC 0.986 (n = 240, @0.5) | Acc 0.681 · Sens 0.901 · Spec 0.587 · AUC 0.855 (n = 2,031, @0.147) |

**Compatibility notes**
1. **Resize interpolation.** Your notebook's `predict_xray()` used `load_img` (nearest-neighbour by default), but training used bilinear resizing. The app uses **bilinear**, verified pixel-identical (max difference 0.0) to the training loader.
2. **`tb_clinical.py` must stay in the project root.** The joblib pipeline references `tb_clinical.FeatureEngineer`.
3. **scikit-learn must be exactly 1.8.0** to unpickle the clinical model.
4. **TensorFlow/Keras version.** The `.keras` file is Keras-3 format. If loading fails, install the TensorFlow version you used in Colab (`print(tf.__version__)` in Colab).
5. **Fusion cannot be fitted yet (most important).** The X-ray images and the clinical records are **different, unpaired datasets**, so no patient has both. Fusion weights and fused test metrics therefore **cannot be estimated** from existing data. Equal weights are used and clearly labelled as an assumption. Once clinicians confirm outcomes for real paired cases, `scripts/evaluate_fusion.py` chooses the weight on a validation split and reports CNN vs clinical vs fusion on an untouched test split.
6. **HIV status** accepts only Positive/Negative (the only levels in training).

The SHAP explanation is computed **exactly** for the logistic regression (`coef × (x − training mean)` in log-odds, equivalent to `shap.LinearExplainer`). The training means are stored in `models/structured/shap_background.json`, and the contributions add up to the model's probability exactly.

---
## 2. Project structure
```
tb_multimodal/
├── app.py                     Flask routes only (loads models ONCE at start-up)
├── config.py                  paths, thresholds, fusion weights (all documented)
├── tb_clinical.py             feature engineering required by the clinical pipeline (do not move)
├── inference/
│   ├── cnn_model.py           X-ray preprocessing, prediction, Grad-CAM
│   └── structured_model.py    input validation, prediction, exact SHAP
├── fusion/fusion.py           agreement-aware fusion rule
├── database/db.py             SQLite schema + queries (AI predictions and confirmed outcomes kept separate)
├── scripts/evaluate_fusion.py controlled CNN vs clinical vs fusion evaluation on confirmed cases
├── templates/  static/        dashboard UI (HTML/CSS/JS)
├── models/cnn/                ← put tb_xray_best.keras here
├── models/structured/         ← put tb_clinical_model.joblib here (shap_background.json included)
├── uploads/                   uploaded X-rays
├── requirements.txt  Procfile  README.md
```

---
## 3. How data flows through the system
```
 Clinician form (14 findings) ──► validate ranges / required fields ──► clinical pipeline ──► p_clinical ─┐
                                                                         └► exact SHAP contributions      │
 Chest X-ray upload ──► check file is an image ──► RGB · bilinear 224×224 ──► CNN ──► p_cnn ──────────────┤
                                                                         └► Grad-CAM heat-map              │
                                                                                                           ▼
                        fusion.fuse():  each model vs its OWN threshold → agreement (concordant/discordant)
                                        fused = 0.5·p_cnn + 0.5·p_clinical  vs  fused threshold 0.5·0.55 + 0.5·0.147
                                        uncertainty = high (disagree or gap ≥ 0.5) / moderate (borderline) / low
                                                                                                           ▼
                        SQLite: cases + predictions (+ later: confirmed_outcomes) ──► results dashboard
```
**Final assessment logic**
* Both above threshold → **High suspicion of TB**: refer for GeneXpert/culture.
* Both below → **Low suspicion of TB**.
* Disagree → **Indeterminate: models disagree**, with *high uncertainty*. Clinician review is required and confirmatory testing is advised. Disagreement is **never hidden** by the averaged number.

Examples: CNN 0.91 + clinical 0.87 → concordant, low uncertainty. CNN 0.15 + clinical 0.82 → discordant, high uncertainty.

---
## 4. Database (SQLite, `database/tb_screening.db`, created automatically)
* `cases`: case_id, patient_ref (pseudonym), sex, clinician, notes, created_at
* `predictions`: clinical inputs (JSON), X-ray path, CNN/clinical/fused probabilities, individual predictions, fusion result, agreement, uncertainty, thresholds/weights/explanations (JSON), model versions, timestamp
* `confirmed_outcomes`: **separate** table holding the clinician-confirmed outcome (TB confirmed / excluded / inconclusive), method, confirmer and time

There is **no automatic retraining**. Confirmed cases are only used manually via `scripts/evaluate_fusion.py`.

---
## 5. Run locally in VS Code (Windows / macOS / Linux)
1. Install **Python 3.11 or 3.12** and **VS Code** with the *Python* extension.
2. Unzip `tb_multimodal.zip`, then in VS Code use **File → Open Folder → tb_multimodal**.
3. Copy your model files into place:
   * `tb_xray_best.keras` → `models/cnn/`
   * `tb_clinical_model.joblib` → `models/structured/`
4. Open a terminal (**Terminal → New Terminal**) and create a virtual environment:
   ```bash
   python -m venv .venv
   # Windows:
   .venv\Scripts\activate
   # macOS/Linux:
   source .venv/bin/activate
   ```
   When VS Code asks *"Select the new environment?"*, click **Yes**. Or use **Ctrl+Shift+P → Python: Select Interpreter → .venv**.
5. Install the dependencies (the TensorFlow download is large):
   ```bash
   pip install -r requirements.txt
   ```
6. Start the app:
   ```bash
   python app.py
   ```
7. Open **http://127.0.0.1:5000**. The sidebar *Model status* should show two green dots. If a dot is red, the banner at the top explains which file is missing. You can also check **http://127.0.0.1:5000/health**.
8. Click **New screening**, fill in the form, drop an X-ray, and press **Run multimodal screening**.

Stop the server with `Ctrl+C`. Saved cases persist in `database/tb_screening.db`.

---
## 6. Free deployment
TensorFlow plus a 134 MB model needs about 1–1.5 GB RAM, so pick a host with enough memory.

**Render**
1. Push the repository to GitHub. The CNN model is over GitHub's 100 MB regular-file limit, so install Git LFS and make sure the `.keras` model is uploaded as an LFS object.
2. In Render, choose **New → Blueprint**, connect this repository, and deploy the `render.yaml` blueprint.
3. Render creates the web service and generates `SECRET_KEY`. Open the service URL and check `/health` for model-loading errors.

The included blueprint intentionally uses Render's **free** instance as requested. Its 512 MB RAM is usually too small for TensorFlow and this model, so the service may fail to start or run out of memory during inference. Upgrade the instance to a paid plan with at least 2 GB RAM if this happens. Free instances also have ephemeral storage: the SQLite database and uploaded files can be lost on restart or redeploy.

---
## 7. Presenting the system (talking points)
* **Separation of concerns:** inference / fusion / database / routes / UI live in separate modules.
* **No training-serving skew:** the clinical pipeline is used exactly as saved. The CNN preprocessing was verified pixel-identical to training.
* **Honest fusion:** a transparent, documented rule. There is no claim of optimal weights, because no paired data exist yet. Disagreement is surfaced, not averaged away.
* **Explainability:** Grad-CAM shows which regions influenced the CNN, and exact SHAP shows which findings drove the clinical risk. Both are **model explanations**, not proof of lesions or causes.
* **Learning loop without risk:** confirmed outcomes are stored separately and evaluated under control, with no automatic retraining.
