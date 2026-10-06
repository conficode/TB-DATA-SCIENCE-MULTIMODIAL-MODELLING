"""SQLite storage. AI predictions and the clinician-confirmed outcome live in SEPARATE tables,
so confirmed results can later be used for controlled evaluation/retraining (never automatic)."""
import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
import config as C


def _json(obj):
    """JSON that also accepts NumPy scalars."""
    return json.dumps(obj, default=lambda o: o.item() if hasattr(o, "item") else str(o))

SCHEMA = """
CREATE TABLE IF NOT EXISTS cases (
    case_id        TEXT PRIMARY KEY,
    patient_ref    TEXT,            -- pseudonymised reference, never a real name
    sex            TEXT,
    clinician      TEXT,
    notes          TEXT,
    created_at     TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS predictions (
    prediction_id        INTEGER PRIMARY KEY AUTOINCREMENT,
    case_id              TEXT NOT NULL REFERENCES cases(case_id),
    clinical_inputs      TEXT NOT NULL,     -- JSON of the 14 model inputs
    xray_path            TEXT NOT NULL,
    cnn_probability      REAL NOT NULL,
    cnn_prediction       TEXT NOT NULL,
    clinical_probability REAL NOT NULL,
    clinical_prediction  TEXT NOT NULL,
    fused_probability    REAL NOT NULL,
    fusion_result        TEXT NOT NULL,     -- assessment text
    agreement            TEXT NOT NULL,
    uncertainty          TEXT NOT NULL,
    details              TEXT,              -- JSON: thresholds, weights, reasons, explanations
    cnn_version          TEXT, clinical_version TEXT, fusion_version TEXT,
    created_at           TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS confirmed_outcomes (
    case_id          TEXT PRIMARY KEY REFERENCES cases(case_id),
    outcome          TEXT NOT NULL CHECK (outcome IN ('TB confirmed', 'TB excluded', 'Inconclusive')),
    method           TEXT,               -- e.g. GeneXpert, culture, clinical diagnosis
    confirmed_by     TEXT,
    notes            TEXT,
    confirmed_at     TEXT NOT NULL
);
"""


class DatabaseError(Exception):
    pass


def now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


@contextmanager
def connect():
    try:
        con = sqlite3.connect(C.DATABASE_PATH)
        con.row_factory = sqlite3.Row
        con.execute("PRAGMA foreign_keys = ON")
        yield con
        con.commit()
    except sqlite3.Error as e:
        raise DatabaseError(f"Database error: {e}") from e
    finally:
        try:
            con.close()
        except Exception:
            pass


def init_db():
    with connect() as con:
        con.executescript(SCHEMA)


def new_case_id():
    with connect() as con:
        n = con.execute("SELECT COUNT(*) FROM cases").fetchone()[0] + 1
    return f"TB-{datetime.now().strftime('%Y%m%d')}-{n:04d}"


def save_case(case, inputs, xray_path, fusion, versions, details):
    with connect() as con:
        con.execute("INSERT INTO cases VALUES (?,?,?,?,?,?)",
                    (case["case_id"], case.get("patient_ref"), case.get("sex"), case.get("clinician"), case.get("notes"), now()))
        con.execute("""INSERT INTO predictions (case_id, clinical_inputs, xray_path, cnn_probability, cnn_prediction,
                       clinical_probability, clinical_prediction, fused_probability, fusion_result, agreement, uncertainty,
                       details, cnn_version, clinical_version, fusion_version, created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (case["case_id"], _json(inputs), xray_path, fusion.cnn_probability,
                     "TB" if fusion.cnn_positive else "No TB", fusion.clinical_probability,
                     "TB" if fusion.clinical_positive else "No TB", fusion.fused_probability, fusion.assessment,
                     fusion.agreement, fusion.uncertainty, _json(details), versions["cnn"], versions["clinical"],
                     versions["fusion"], now()))


def save_outcome(case_id, outcome, method, confirmed_by, notes):
    with connect() as con:
        con.execute("INSERT OR REPLACE INTO confirmed_outcomes VALUES (?,?,?,?,?,?)",
                    (case_id, outcome, method, confirmed_by, notes, now()))


def get_case(case_id):
    with connect() as con:
        case = con.execute("SELECT * FROM cases WHERE case_id=?", (case_id,)).fetchone()
        pred = con.execute("SELECT * FROM predictions WHERE case_id=? ORDER BY prediction_id DESC", (case_id,)).fetchone()
        out = con.execute("SELECT * FROM confirmed_outcomes WHERE case_id=?", (case_id,)).fetchone()
    if not case:
        return None
    pred = dict(pred) if pred else None
    if pred:
        pred["clinical_inputs"] = json.loads(pred["clinical_inputs"])
        pred["details"] = json.loads(pred["details"] or "{}")
    return {"case": dict(case), "prediction": pred, "outcome": dict(out) if out else None}


def list_cases(limit=200):
    with connect() as con:
        rows = con.execute("""SELECT c.case_id, c.patient_ref, c.created_at, p.cnn_probability, p.clinical_probability,
                              p.fused_probability, p.agreement, p.uncertainty, p.fusion_result, o.outcome
                              FROM cases c LEFT JOIN predictions p ON p.case_id = c.case_id
                              LEFT JOIN confirmed_outcomes o ON o.case_id = c.case_id
                              ORDER BY c.created_at DESC LIMIT ?""", (limit,)).fetchall()
    return [dict(r) for r in rows]


def stats():
    with connect() as con:
        r = con.execute("""SELECT COUNT(*) n, SUM(agreement='discordant') disc, SUM(agreement='concordant_positive') pos
                           FROM predictions""").fetchone()
        conf = con.execute("SELECT COUNT(*) FROM confirmed_outcomes").fetchone()[0]
    return {"cases": r["n"] or 0, "discordant": r["disc"] or 0, "high_suspicion": r["pos"] or 0, "confirmed": conf}
