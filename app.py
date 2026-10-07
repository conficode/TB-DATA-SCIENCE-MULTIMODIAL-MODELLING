"""Flask entry point: routes only. Inference, fusion and storage live in their own packages.

Data flow:  form + X-ray -> validation -> ClinicalModel.predict / CNNModel.predict
            -> fusion.fuse -> explanations (SHAP, Grad-CAM) -> database.save_case -> results page
"""
import logging
import uuid
from pathlib import Path
from flask import Flask, render_template, request, redirect, url_for, flash, abort, jsonify
from werkzeug.utils import secure_filename

import config as C
from database import db
from fusion.fusion import fuse
from inference.cnn_model import CNNModel, validate_image
from inference.structured_model import ClinicalModel, NUMERIC_FIELDS, BINARY_FIELDS

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("app")

app = Flask(__name__)
app.config.update(SECRET_KEY=C.SECRET_KEY, MAX_CONTENT_LENGTH=C.MAX_UPLOAD_MB * 1024 * 1024)
for d in (C.UPLOAD_DIR, C.GRADCAM_DIR, C.DATABASE_PATH.parent):
    Path(d).mkdir(parents=True, exist_ok=True)

# ---- models are loaded ONCE at start-up ----
CNN = CNNModel()
CLINICAL = ClinicalModel()
db.init_db()


def model_status():
    return {"cnn": {"ready": CNN.ready, "error": CNN.error, "loaded_from": CNN.source, "version": C.CNN_VERSION, "threshold": C.CNN_THRESHOLD},
            "clinical": {"ready": CLINICAL.ready, "error": CLINICAL.error,
                         "version": CLINICAL.version if CLINICAL.ready else None,
                         "threshold": CLINICAL.threshold if CLINICAL.ready else None}}


@app.context_processor
def inject():
    return {"status": model_status()}


@app.route("/")
def dashboard():
    try:
        return render_template("dashboard.html", stats=db.stats(), cases=db.list_cases(8))
    except db.DatabaseError as e:
        flash(str(e), "error")
        return render_template("dashboard.html", stats={}, cases=[])


@app.route("/screening/new", methods=["GET"])
def new_screening():
    return render_template("new.html", numeric=NUMERIC_FIELDS, binary=BINARY_FIELDS, form={})


@app.route("/screening/run", methods=["POST"])
def run_screening():
    form = request.form
    errors = []
    if not CNN.ready:
        errors.append(f"X-ray model unavailable: {CNN.error}")
    if not CLINICAL.ready:
        errors.append(f"Clinical model unavailable: {CLINICAL.error}")
    inputs, field_errors = ClinicalModel.validate(form)
    errors += field_errors

    file = request.files.get("xray")
    if not file or file.filename == "":
        errors.append("Please upload a chest X-ray image.")
    elif Path(file.filename).suffix.lower() not in C.ALLOWED_IMAGE_EXT:
        errors.append("X-ray must be a PNG or JPEG image.")
    if errors:
        for e in errors:
            flash(e, "error")
        return render_template("new.html", numeric=NUMERIC_FIELDS, binary=BINARY_FIELDS, form=form), 400

    case_id = db.new_case_id()
    fname = f"{case_id}_{uuid.uuid4().hex[:6]}_{secure_filename(file.filename)}"
    path = C.UPLOAD_DIR / fname
    file.save(path)
    try:
        validate_image(path)
    except ValueError as e:
        path.unlink(missing_ok=True)
        flash(str(e), "error")
        return render_template("new.html", numeric=NUMERIC_FIELDS, binary=BINARY_FIELDS, form=form), 400

    try:
        p_clin = CLINICAL.predict(inputs)
        shap_exp = CLINICAL.explain(inputs)
        p_cnn, x = CNN.predict(path)
        cam_path = C.GRADCAM_DIR / f"{case_id}_cam.png"
        try:
            CNN.gradcam(x, cam_path)
            cam_ok = True
        except Exception as e:
            log.warning("Grad-CAM failed: %s", e)
            cam_ok = False
        result = fuse(p_cnn, p_clin, t_cnn=C.CNN_THRESHOLD, t_clin=CLINICAL.threshold)
    except Exception as e:
        log.exception("Inference failed")
        flash(f"Prediction failed: {e}", "error")
        return render_template("new.html", numeric=NUMERIC_FIELDS, binary=BINARY_FIELDS, form=form), 500

    details = {"fusion": result.to_dict(), "weights": C.FUSION_WEIGHTS, "shap": shap_exp, "gradcam": cam_ok,
               "thresholds": {"cnn": C.CNN_THRESHOLD, "clinical": CLINICAL.threshold, "fused": result.fused_threshold}}
    case = {"case_id": case_id, "patient_ref": form.get("patient_ref", "").strip() or case_id,
            "sex": form.get("sex"), "clinician": form.get("clinician", "").strip(), "notes": form.get("notes", "").strip()}
    try:
        db.save_case(case, inputs, str(path.relative_to(C.BASE_DIR)), result,
                     {"cnn": C.CNN_VERSION, "clinical": CLINICAL.version, "fusion": C.FUSION_VERSION}, details)
    except db.DatabaseError as e:
        flash(f"Results computed but could not be saved: {e}", "error")
    return redirect(url_for("view_case", case_id=case_id))


@app.route("/case/<case_id>")
def view_case(case_id):
    data = db.get_case(case_id)
    if not data:
        abort(404)
    return render_template("case.html", d=data, binary=BINARY_FIELDS, numeric=NUMERIC_FIELDS)


@app.route("/case/<case_id>/outcome", methods=["POST"])
def save_outcome(case_id):
    outcome = request.form.get("outcome")
    if outcome not in ("TB confirmed", "TB excluded", "Inconclusive"):
        flash("Please choose a confirmed outcome.", "error")
    else:
        try:
            db.save_outcome(case_id, outcome, request.form.get("method"), request.form.get("confirmed_by"), request.form.get("notes"))
            flash("Confirmed outcome saved (stored separately from the AI prediction).", "success")
        except db.DatabaseError as e:
            flash(str(e), "error")
    return redirect(url_for("view_case", case_id=case_id))


@app.route("/cases")
def cases():
    return render_template("cases.html", cases=db.list_cases())


@app.route("/models")
def models_page():
    return render_template("models.html", cfg=C)


@app.route("/health")
def health():
    return jsonify(model_status())


@app.errorhandler(413)
def too_large(_):
    flash(f"File too large (max {C.MAX_UPLOAD_MB} MB).", "error")
    return redirect(url_for("new_screening"))


@app.errorhandler(404)
def not_found(_):
    return render_template("base.html", not_found=True), 404


if __name__ == "__main__":
    app.run(debug=True, port=5000)
