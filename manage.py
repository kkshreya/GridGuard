"""
Command-line helper.

    python manage.py train          # (re)train all models (full CV, ~1-2 min)
    python manage.py seed           # create demo users + demo scan history
    python manage.py reset-db       # delete the database (asks first)

Demo accounts created by `seed` (for local demos ONLY - change/delete in real use):
    admin      / Admin@123
    inspector  / Inspector@123
"""
import argparse
import os
import random
import secrets
import sys

import pandas as pd
from werkzeug.security import generate_password_hash

from config import Config


def cmd_train(_):
    from ml.train import train_all
    train_all()


def cmd_reset(_):
    if input(f"Delete {Config.DATABASE}? [y/N] ").lower() == "y":
        if os.path.exists(Config.DATABASE):
            os.remove(Config.DATABASE)
        print("Database removed.")


def cmd_seed(args):
    from app import create_app
    from database import get_db
    from ml import features as F

    app = create_app()
    with app.app_context():
        db = get_db()
        for name, pw, role in (("admin", "Admin@123", "admin"), ("inspector", "Inspector@123", "inspector")):
            if not db.execute("SELECT 1 FROM user WHERE username=?", (name,)).fetchone():
                db.execute("INSERT INTO user (username,email,phone,place,password,role,api_key) VALUES (?,?,?,?,?,?,?)",
                           (name, f"{name}@gridguard.local", "", "Bengaluru", generate_password_hash(pw), role,
                            secrets.token_urlsafe(24)))
        db.commit()
        admin_id = db.execute("SELECT id FROM user WHERE username='admin'").fetchone()[0]
        insp_id = db.execute("SELECT id FROM user WHERE username='inspector'").fetchone()[0]

        raw = pd.read_csv(app.config["DATA_PATH"]).sample(args.rows, random_state=11).reset_index(drop=True)
        truth = raw["Anomaly"].values
        clean, _ = F.normalize_raw(raw)
        results = app.predictor.predict(clean, explain="flagged")

        # split into 3 "historical" scans on different days so the trend chart has shape
        cuts = [0, len(clean) // 4, len(clean) * 5 // 8, len(clean)]
        offsets = [10, 5, 0]
        import json
        for b in range(3):
            lo, hi = cuts[b], cuts[b + 1]
            part, res = clean.iloc[lo:hi].reset_index(drop=True), results[lo:hi]
            bid = db.execute("INSERT INTO batch (user_id,filename,total,flagged,skipped,created_at) "
                             "VALUES (?,?,?,?,0, datetime('now', ?))",
                             (admin_id, f"demo_scan_{b + 1}.csv", len(res), sum(r['label'] for r in res),
                              f"-{offsets[b]} days")).lastrowid
            for i, r in enumerate(res):
                row = part.iloc[i]
                inputs = {c: (row[c] if isinstance(row[c], str) else float(row[c])) for c in F.RAW_REQUIRED}
                pid = db.execute(
                    """INSERT INTO prediction (batch_id,user_id,source,customer_id,meter_id,consumer_type,season,energy,
                       inputs,probability,risk_score,risk_level,label,anomaly_score,model_version,explanation,created_at)
                       VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?, datetime('now', ?))""",
                    (bid, admin_id, "batch", str(row.get("CustomerID", "")), str(row.get("MeterID", "")),
                     row["ConsumerType"], row["Season"], float(row["EnergyConsumption"]), json.dumps(inputs),
                     r["probability"], r["risk_score"], r["risk_level"], r["label"], r["anomaly_score"],
                     app.predictor.version, json.dumps({"explanation": r["explanation"], "model_probs": r["model_probs"]}),
                     f"-{offsets[b]} days")).lastrowid
                if r["label"] == 1 and r["risk_level"] in ("High", "Critical"):
                    # Field outcome taken from the dataset's ground truth (simulated inspections).
                    t = int(truth[lo + i])
                    roll = random.Random(pid).random()
                    if roll < 0.55:
                        status, outcome, assignee = ("Confirmed Theft", 1, insp_id) if t else ("False Alarm", 0, insp_id)
                    elif roll < 0.75:
                        status, outcome, assignee = "Inspecting", None, insp_id
                    elif roll < 0.85:
                        status, outcome, assignee = "Assigned", None, insp_id
                    else:
                        status, outcome, assignee = "Open", None, None
                    db.execute("INSERT INTO inspection_case (prediction_id,priority,status,assigned_to,outcome_label,notes) "
                               "VALUES (?,?,?,?,?,?)",
                               (pid, r["risk_level"], status, assignee, outcome,
                                "[demo] simulated inspection outcome\n" if outcome is not None else ""))
        db.commit()
        n_cases = db.execute("SELECT COUNT(*) FROM inspection_case").fetchone()[0]
        n_fb = db.execute("SELECT COUNT(*) FROM inspection_case WHERE outcome_label IS NOT NULL").fetchone()[0]
        print(f"Seeded {len(clean)} readings, {n_cases} cases ({n_fb} verified -> usable for retraining).")
        print("Log in with  admin / Admin@123   or   inspector / Inspector@123")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("train").set_defaults(fn=cmd_train)
    sub.add_parser("reset-db").set_defaults(fn=cmd_reset)
    s = sub.add_parser("seed"); s.add_argument("--rows", type=int, default=600); s.set_defaults(fn=cmd_seed)
    a = ap.parse_args()
    a.fn(a)
