# FractureAI – Bone Fracture Detection (MVP)

A simple Flask app for storing patient records in SQLite and screening X-ray images
for bone fractures with a MobileNetV2 deep learning model.

## Features

- Login (default: `admin` / `admin123`)
- Patient records with full CRUD in SQLite (`fractureai.db`):
  - **Create**: add a patient, optionally with an X-ray image
  - **Read**: list and search patients, view one patient
  - **Update**: edit details or upload a new X-ray (re-runs the prediction)
  - **Delete**: remove the patient and their X-ray image
- Fracture / Normal prediction with a confidence score

## Project Structure

```
app.py                Entry point: python app.py
backend/
  app.py              Flask routes
  db.py               SQLite setup and patient CRUD functions
  templates/          HTML pages
model/
  bone_fracture_model.h5
  evaluate_model.py
  metrics.json
  predict_single_image.py
  preprocess_data.py
  train_mobilenetv2.py
dataset/              train/val/test images (fractured, non_fractured)
static/css/style.css
uploads/              saved X-ray images
```

## Database

Table `patients`:

| Column | Type |
|---|---|
| id | INTEGER PRIMARY KEY |
| name | TEXT (required) |
| age | INTEGER |
| gender | TEXT |
| phone | TEXT |
| notes | TEXT |
| xray_image | TEXT |
| result | TEXT (`Fracture` / `Normal`) |
| confidence | REAL |
| created_at, updated_at | TEXT |

## Run Locally

```bash
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt
copy .env.example .env         # then set FRACTUREAI_SECRET_KEY
python app.py
```

Open http://localhost:5000.

## Train the Model

Put images in `dataset/{train,val,test}/{fractured,non_fractured}` and run:

```bash
python model\train_mobilenetv2.py
```

This writes `model/bone_fracture_model.h5`.

## Evaluate the Model

```bash
python model\evaluate_model.py
```

This writes validation and test metrics (accuracy, precision, recall, specificity,
F1, ROC-AUC, confusion matrix) to `model/metrics.json`.

> AI screening only. Results must be reviewed by a qualified doctor.
