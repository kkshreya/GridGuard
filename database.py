"""SQLite layer (no server needed - the DB is a single file in instance/)."""
import sqlite3
from flask import g, current_app

SCHEMA = """
CREATE TABLE IF NOT EXISTS user (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT NOT NULL UNIQUE,
    email TEXT NOT NULL,
    phone TEXT,
    place TEXT,
    password TEXT NOT NULL,
    role TEXT NOT NULL DEFAULT 'inspector',      -- 'admin' | 'inspector'
    api_key TEXT UNIQUE,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE TABLE IF NOT EXISTS batch (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER REFERENCES user(id),
    filename TEXT,
    total INTEGER, flagged INTEGER, skipped INTEGER DEFAULT 0,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE TABLE IF NOT EXISTS prediction (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    batch_id INTEGER REFERENCES batch(id) ON DELETE CASCADE,
    user_id INTEGER REFERENCES user(id),
    source TEXT NOT NULL,                        -- single | batch | api
    customer_id TEXT, meter_id TEXT,
    consumer_type TEXT, season TEXT, energy REAL,
    inputs TEXT NOT NULL,                        -- JSON of the 9 raw inputs
    probability REAL, risk_score REAL, risk_level TEXT,
    label INTEGER, anomaly_score INTEGER,
    model_version TEXT,
    explanation TEXT,                            -- JSON
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_pred_batch ON prediction(batch_id);
CREATE INDEX IF NOT EXISTS idx_pred_created ON prediction(created_at);
CREATE TABLE IF NOT EXISTS inspection_case (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    prediction_id INTEGER NOT NULL UNIQUE REFERENCES prediction(id) ON DELETE CASCADE,
    status TEXT NOT NULL DEFAULT 'Open',         -- Open|Assigned|Inspecting|Confirmed Theft|False Alarm
    priority TEXT NOT NULL,
    assigned_to INTEGER REFERENCES user(id),
    notes TEXT DEFAULT '',
    outcome_label INTEGER,                       -- 1 confirmed theft, 0 false alarm
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_case_status ON inspection_case(status);
CREATE TABLE IF NOT EXISTS model_version (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    version TEXT, note TEXT, n_feedback INTEGER DEFAULT 0,
    before_f1 REAL, after_f1 REAL,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);
"""


def get_db():
    if "db" not in g:
        g.db = sqlite3.connect(current_app.config["DATABASE"])
        g.db.row_factory = sqlite3.Row
        g.db.execute("PRAGMA foreign_keys = ON")
    return g.db


def close_db(_exc=None):
    db = g.pop("db", None)
    if db is not None:
        db.close()


def init_db(app):
    with app.app_context():
        db = get_db()
        db.executescript(SCHEMA)
        db.commit()
