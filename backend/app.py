from functools import wraps
import os
from pathlib import Path
import sys
from uuid import uuid4

from flask import Flask, abort, flash, redirect, render_template, request, send_from_directory, session, url_for
from werkzeug.utils import secure_filename

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    load_dotenv = None


BASE_DIR = Path(__file__).resolve().parents[1]
if load_dotenv is not None:
    load_dotenv(BASE_DIR / ".env")

MODEL_DIR = BASE_DIR / "model"
MODEL_PATH = MODEL_DIR / "bone_fracture_model.h5"
DATA_DIR = Path(os.getenv("FRACTUREAI_DATA_DIR", str(BASE_DIR)))
UPLOAD_DIR = DATA_DIR / "uploads"
ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg"}

sys.path.insert(0, str(MODEL_DIR))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from db import create_patient, delete_patient, get_patient, init_db, list_patients, update_patient, verify_user  # noqa: E402


app = Flask(
    __name__,
    template_folder=str(BASE_DIR / "backend" / "templates"),
    static_folder=str(BASE_DIR / "static"),
)
app.secret_key = os.getenv("FRACTUREAI_SECRET_KEY", "fractureai-secret-key")

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
init_db()


def login_required(view_function):
    @wraps(view_function)
    def wrapped_view(*args, **kwargs):
        if "user_id" not in session:
            return redirect(url_for("login"))
        return view_function(*args, **kwargs)

    return wrapped_view


def read_patient_form(form):
    age = form.get("age", "").strip()
    return {
        "name": form.get("name", "").strip(),
        "age": int(age) if age.isdigit() else None,
        "gender": form.get("gender", "").strip(),
        "phone": form.get("phone", "").strip(),
        "notes": form.get("notes", "").strip(),
    }


def process_xray(uploaded_file):
    """Save the uploaded X-ray and run the model. Returns the fields to store."""
    if Path(uploaded_file.filename).suffix.lower() not in ALLOWED_EXTENSIONS:
        raise ValueError("Please upload a PNG, JPG, or JPEG image.")

    filename = f"{uuid4().hex}_{secure_filename(uploaded_file.filename)}"
    uploaded_file.save(UPLOAD_DIR / filename)
    fields = {"xray_image": filename, "result": None, "confidence": None}

    if not MODEL_PATH.exists():
        flash("Model file not found, so the X-ray was saved without a prediction.", "error")
        return fields

    from predict_single_image import predict_fracture

    result, confidence = predict_fracture(UPLOAD_DIR / filename)
    fields.update(result=result, confidence=round(float(confidence), 2))
    return fields


def remove_xray(filename):
    if filename:
        (UPLOAD_DIR / filename).unlink(missing_ok=True)


@app.route("/login", methods=["GET", "POST"])
def login():
    error = None
    if request.method == "POST":
        user = verify_user(request.form.get("username", "").strip(), request.form.get("password", ""))
        if user:
            session["user_id"] = user["id"]
            session["username"] = user["username"]
            return redirect(url_for("index"))
        error = "Invalid username or password."
    return render_template("login.html", error=error)


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


# Read (list)
@app.route("/")
@login_required
def index():
    search = request.args.get("q", "").strip()
    return render_template("patients.html", patients=list_patients(search), search=search)


# Create
@app.route("/patients/new", methods=["GET", "POST"])
@login_required
def new_patient():
    if request.method == "POST":
        data = read_patient_form(request.form)
        if not data["name"]:
            flash("Patient name is required.", "error")
            return render_template("patient_form.html", patient=data)

        xray = request.files.get("xray_image")
        if xray and xray.filename:
            try:
                data.update(process_xray(xray))
            except Exception as exc:
                flash(f"X-ray could not be processed: {exc}", "error")
                return render_template("patient_form.html", patient=data)

        patient_id = create_patient(data)
        flash("Patient record created.", "success")
        return redirect(url_for("view_patient", patient_id=patient_id))

    return render_template("patient_form.html", patient={})


# Read (single)
@app.route("/patients/<int:patient_id>")
@login_required
def view_patient(patient_id):
    patient = get_patient(patient_id) or abort(404)
    return render_template("patient_detail.html", patient=patient)


# Update
@app.route("/patients/<int:patient_id>/edit", methods=["GET", "POST"])
@login_required
def edit_patient(patient_id):
    patient = get_patient(patient_id) or abort(404)

    if request.method == "POST":
        data = read_patient_form(request.form)
        if not data["name"]:
            flash("Patient name is required.", "error")
            return render_template("patient_form.html", patient={**patient, **data})

        xray = request.files.get("xray_image")
        if xray and xray.filename:
            try:
                data.update(process_xray(xray))
            except Exception as exc:
                flash(f"X-ray could not be processed: {exc}", "error")
                return render_template("patient_form.html", patient={**patient, **data})
            remove_xray(patient["xray_image"])

        update_patient(patient_id, data)
        flash("Patient record updated.", "success")
        return redirect(url_for("view_patient", patient_id=patient_id))

    return render_template("patient_form.html", patient=patient)


# Delete
@app.route("/patients/<int:patient_id>/delete", methods=["POST"])
@login_required
def remove_patient(patient_id):
    patient = get_patient(patient_id) or abort(404)
    delete_patient(patient_id)
    remove_xray(patient["xray_image"])
    flash(f"Patient '{patient['name']}' deleted.", "success")
    return redirect(url_for("index"))


@app.route("/uploads/<path:filename>")
@login_required
def uploaded_file(filename):
    return send_from_directory(str(UPLOAD_DIR), filename)
