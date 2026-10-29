import os
import sqlite3
from datetime import datetime
from pathlib import Path

from werkzeug.security import check_password_hash, generate_password_hash


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.getenv("FRACTUREAI_DATA_DIR", str(BASE_DIR)))
DATA_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH = DATA_DIR / "fractureai.db"

PATIENT_FIELDS = ("name", "age", "gender", "phone", "notes", "xray_image", "result", "confidence")


def get_connection():
    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row
    return connection


def _now():
    return datetime.now().strftime("%Y-%m-%d %H:%M")


def init_db():
    with get_connection() as connection:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL UNIQUE,
                password_hash TEXT NOT NULL,
                role TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS patients (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                age INTEGER,
                gender TEXT,
                phone TEXT,
                notes TEXT,
                xray_image TEXT,
                result TEXT,
                confidence REAL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            """
        )
        connection.execute(
            "INSERT OR IGNORE INTO users (username, password_hash, role) VALUES (?, ?, ?)",
            ("admin", generate_password_hash("admin123"), "admin"),
        )


def verify_user(username, password):
    with get_connection() as connection:
        user = connection.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()
    if user and check_password_hash(user["password_hash"], password):
        return dict(user)
    return None


# ---------- Patient CRUD ----------

def create_patient(data):
    values = [data.get(field) for field in PATIENT_FIELDS]
    timestamp = _now()
    with get_connection() as connection:
        cursor = connection.execute(
            f"INSERT INTO patients ({', '.join(PATIENT_FIELDS)}, created_at, updated_at) "
            f"VALUES ({', '.join('?' * len(PATIENT_FIELDS))}, ?, ?)",
            (*values, timestamp, timestamp),
        )
        return cursor.lastrowid


def list_patients(search=""):
    sql = "SELECT * FROM patients"
    params = ()
    if search:
        sql += " WHERE name LIKE ? OR CAST(id AS TEXT) = ?"
        params = (f"%{search}%", search)
    sql += " ORDER BY id DESC"
    with get_connection() as connection:
        return [dict(row) for row in connection.execute(sql, params).fetchall()]


def get_patient(patient_id):
    with get_connection() as connection:
        row = connection.execute("SELECT * FROM patients WHERE id = ?", (patient_id,)).fetchone()
    return dict(row) if row else None


def update_patient(patient_id, data):
    fields = [field for field in PATIENT_FIELDS if field in data]
    assignments = ", ".join(f"{field} = ?" for field in fields)
    with get_connection() as connection:
        connection.execute(
            f"UPDATE patients SET {assignments}, updated_at = ? WHERE id = ?",
            (*[data[field] for field in fields], _now(), patient_id),
        )


def delete_patient(patient_id):
    with get_connection() as connection:
        connection.execute("DELETE FROM patients WHERE id = ?", (patient_id,))
