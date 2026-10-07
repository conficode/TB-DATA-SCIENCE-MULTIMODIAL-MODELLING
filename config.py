"""Central configuration. Every number here is traceable to a notebook result."""
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

# ---------------- model files (place your files here) ----------------
CNN_MODEL_PATH = Path(os.getenv("CNN_MODEL_PATH", BASE_DIR / "models" / "cnn" / "tb_xray_best.keras"))
CNN_WEIGHTS_PATH = Path(os.getenv("CNN_WEIGHTS_PATH", BASE_DIR / "models" / "cnn" / "tb_xray_weights.npz"))  # NumPy export used at runtime
CLINICAL_MODEL_PATH = Path(os.getenv("CLINICAL_MODEL_PATH", BASE_DIR / "models" / "structured" / "tb_clinical_model.joblib"))
SHAP_BACKGROUND_PATH = BASE_DIR / "models" / "structured" / "shap_background.json"

# ---------------- CNN (from xray notebook) ----------------
CNN_IMG_SIZE = (224, 224)          # image_dataset_from_directory(image_size=(224, 224))
CNN_CLASS_NAMES = ["NORMAL", "TB"] # alphabetical folder order -> sigmoid output = P(TB)
CNN_THRESHOLD = 0.55               # validation table: highest threshold keeping sensitivity 0.990 (spec 0.978)
CNN_VERSION = "tb_xray_best.keras (custom 3-block CNN, test AUC 0.986)"

# ---------------- clinical model ----------------
# 0.35 maximises both F1 (0.685) and Youden's J (sens 0.732 + spec 0.828 - 1) in the notebook's out-of-fold
# threshold table. The notebook's 0.147 (>= 90% sensitivity) flagged ~41% of non-TB patients as positive.
CLINICAL_THRESHOLD = 0.35
# Numeric inputs are limited to the training range (mean +- 3 SD) before prediction: a logistic regression
# extrapolates linearly, so e.g. SpO2 65% (17 SD below the training mean) would otherwise force P(TB) to ~0.
CLINICAL_CLIP_SD = 3.0
# Clinical plausibility guard: model terms whose learned direction contradicts clinical knowledge are
# neutralised (held at the average training patient's value, contribution 0). In the training data fever's
# own term is negative (fever is already counted in core_symptom_count, and the model offsets it), and LOW
# SpO2 lowered P(TB). Without the guard a feverish, hypoxic patient scores LOWER than the same patient
# without those findings. Fever still counts through the core-symptom count. Set to () to disable.
CLINICAL_NEUTRAL_TERMS = ("bin__fever", "num__spo2")

# ---------------- fusion (documented assumption, see fusion/fusion.py) ----------------
FUSION_WEIGHTS = {"cnn": 0.5, "clinical": 0.5}   # equal weights: NO paired data exists to estimate them
BORDERLINE_MARGIN = 0.10           # a model within +-0.10 of its own threshold = borderline
CONFLICT_GAP = 0.50                # |p_cnn - p_clinical| >= 0.50 = strong conflict
FUSION_VERSION = "fusion-rule-1.1 (equal-weight average, agreement-aware)"

# ---------------- app ----------------
DATABASE_PATH = BASE_DIR / "database" / "tb_screening.db"
UPLOAD_DIR = BASE_DIR / "uploads"
GRADCAM_DIR = BASE_DIR / "static" / "gradcam"
ALLOWED_IMAGE_EXT = {".png", ".jpg", ".jpeg"}
MAX_UPLOAD_MB = 15
SECRET_KEY = os.getenv("SECRET_KEY", "dev-only-change-me")
