"""
GridGuard - AI-based electricity theft detection & inspection management.

Run:  python app.py        (then open http://127.0.0.1:5000)
"""
from __future__ import annotations

import io
import json
import os
import re
import secrets
from functools import wraps

import pandas as pd
from flask import (Flask, Response, abort, flash, g, jsonify, redirect,
                   render_template, request, session, url_for)
from markupsafe import Markup
from werkzeug.security import check_password_hash, generate_password_hash

from config import Config
from database import close_db, get_db, init_db
from ml import features as F
from ml.predictor import Predictor

CASE_STATUSES = ["Open", "Assigned", "Inspecting", "Confirmed Theft", "False Alarm"]
PRIORITY_ORDER = "CASE priority WHEN 'Critical' THEN 0 WHEN 'High' THEN 1 ELSE 2 END"
USERNAME_RE = re.compile(r"^[A-Za-z0-9_]{3,30}$")
EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def create_app(test_config: dict | None = None) -> Flask:
    app = Flask(__name__)
    app.config.from_object(Config)
    if test_config:
        app.config.update(test_config)

    app.teardown_appcontext(close_db)
    init_db(app)
    app.predictor = _load_predictor(app)

    # ------------------------------------------------------------------ #
    # Security helpers: CSRF + auth decorators
    # ------------------------------------------------------------------ #
    def csrf_token() -> str:
        if "_csrf" not in session:
            session["_csrf"] = secrets.token_hex(16)
        return session["_csrf"]

    app.jinja_env.globals["csrf_token"] = csrf_token
    app.jinja_env.globals["csrf"] = lambda: Markup(
        f'<input type="hidden" name="csrf_token" value="{csrf_token()}">')

    @app.before_request
    def _before():
        g.user = None
        if "uid" in session:
            g.user = get_db().execute("SELECT * FROM user WHERE id = ?", (session["uid"],)).fetchone()
            if g.user is None:
                session.clear()
        if request.method == "POST" and not request.path.startswith("/api/"):
            sent = request.form.get("csrf_token", "")
            if not sent or not secrets.compare_digest(sent, session.get("_csrf", "")):
                abort(400, "Invalid or missing CSRF token. Reload the page and try again.")

    @app.context_processor
    def _inject():
        return {"user": g.get("user"), "model_version": app.predictor.version,
                "case_statuses": CASE_STATUSES}

    def login_required(view):
        @wraps(view)
        def wrapped(*a, **kw):
            if g.user is None:
                flash("Please log in to continue.", "warning")
                return redirect(url_for("login", next=request.path))
            return view(*a, **kw)
        return wrapped

    def admin_required(view):
        @wraps(view)
        def wrapped(*a, **kw):
            if g.user is None:
                return redirect(url_for("login", next=request.path))
            if g.user["role"] != "admin":
                abort(403)
            return view(*a, **kw)
        return wrapped

    # ------------------------------------------------------------------ #
    # Prediction persistence
    # ------------------------------------------------------------------ #
    def save_predictions(raw: pd.DataFrame, results: list[dict], user_id, source, batch_id=None):
        db = get_db()
        ids = []
        for i, res in enumerate(results):
            row = raw.iloc[i]
            inputs = {c: (row[c] if isinstance(row[c], str) else float(row[c])) for c in F.RAW_REQUIRED}
            cur = db.execute(
                """INSERT INTO prediction (batch_id,user_id,source,customer_id,meter_id,consumer_type,season,
                   energy,inputs,probability,risk_score,risk_level,label,anomaly_score,model_version,explanation)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (batch_id, user_id, source,
                 _opt(row, "CustomerID"), _opt(row, "MeterID"),
                 row["ConsumerType"], row["Season"], float(row["EnergyConsumption"]),
                 json.dumps(inputs), res["probability"], res["risk_score"], res["risk_level"],
                 res["label"], res["anomaly_score"], app.predictor.version,
                 json.dumps({"explanation": res["explanation"], "model_probs": res["model_probs"]})))
            pid = cur.lastrowid
            ids.append(pid)
            if res["label"] == 1 and res["risk_level"] in ("High", "Critical"):
                db.execute("INSERT INTO inspection_case (prediction_id, priority) VALUES (?, ?)",
                           (pid, res["risk_level"]))
        db.commit()
        return ids

    # ------------------------------------------------------------------ #
    # Public pages + auth
    # ------------------------------------------------------------------ #
    @app.route("/")
    def home():
        return render_template("index.html", metrics=app.predictor.metrics)

    @app.route("/about")
    def about():
        return render_template("about.html")

    @app.route("/register", methods=["GET", "POST"])
    def register():
        msg, ok = "", False
        if request.method == "POST":
            f = request.form
            username, email, password = f.get("username", "").strip(), f.get("email", "").strip(), f.get("password", "")
            db = get_db()
            if not USERNAME_RE.match(username):
                msg = "Username must be 3-30 characters: letters, numbers or underscore."
            elif not EMAIL_RE.match(email):
                msg = "Invalid email address."
            elif len(password) < 8:
                msg = "Password must be at least 8 characters."
            elif db.execute("SELECT 1 FROM user WHERE username = ?", (username,)).fetchone():
                msg = "That username is already taken."
            else:
                first = db.execute("SELECT COUNT(*) c FROM user").fetchone()["c"] == 0
                db.execute("INSERT INTO user (username,email,phone,place,password,role,api_key) VALUES (?,?,?,?,?,?,?)",
                           (username, email, f.get("phone", "").strip(), f.get("place", "").strip(),
                            generate_password_hash(password), "admin" if first else "inspector",
                            secrets.token_urlsafe(24)))
                db.commit()
                ok = True
                msg = "Account created" + (" - you are the administrator (first user)." if first else ".") + " Please log in."
        return render_template("register.html", msg=msg, ok=ok)

    @app.route("/login", methods=["GET", "POST"])
    def login():
        msg = ""
        if request.method == "POST":
            row = get_db().execute("SELECT * FROM user WHERE username = ?",
                                   (request.form.get("username", "").strip(),)).fetchone()
            if row and check_password_hash(row["password"], request.form.get("password", "")):
                session.clear()
                session["uid"] = row["id"]
                nxt = request.args.get("next", "")
                return redirect(nxt if nxt.startswith("/") and not nxt.startswith("//") else url_for("dashboard"))
            msg = "Incorrect username or password."
        return render_template("login.html", msg=msg)

    @app.route("/logout", methods=["POST"])
    def logout():
        session.clear()
        return redirect(url_for("home"))

    @app.route("/profile", methods=["GET", "POST"])
    @login_required
    def profile():
        if request.method == "POST":
            get_db().execute("UPDATE user SET api_key = ? WHERE id = ?", (secrets.token_urlsafe(24), g.user["id"]))
            get_db().commit()
            flash("API key regenerated.", "success")
            return redirect(url_for("profile"))
        return render_template("profile.html")

    # ------------------------------------------------------------------ #
    # Dashboard
    # ------------------------------------------------------------------ #
    @app.route("/dashboard")
    @login_required
    def dashboard():
        db = get_db()
        one = lambda q, a=(): db.execute(q, a).fetchone()[0]
        kpi = {
            "scanned": one("SELECT COUNT(*) FROM prediction"),
            "flagged": one("SELECT COUNT(*) FROM prediction WHERE label = 1"),
            "open_cases": one("SELECT COUNT(*) FROM inspection_case WHERE status IN ('Open','Assigned','Inspecting')"),
            "confirmed": one("SELECT COUNT(*) FROM inspection_case WHERE status = 'Confirmed Theft'"),
            "false_alarms": one("SELECT COUNT(*) FROM inspection_case WHERE status = 'False Alarm'"),
            "feedback": one("SELECT COUNT(*) FROM inspection_case WHERE outcome_label IS NOT NULL"),
        }
        verified = kpi["confirmed"] + kpi["false_alarms"]
        kpi["field_precision"] = round(100 * kpi["confirmed"] / verified, 1) if verified else None

        rows = lambda q: [dict(r) for r in db.execute(q).fetchall()]
        charts = {
            "risk": rows("""SELECT risk_level k, COUNT(*) n FROM prediction GROUP BY risk_level"""),
            "by_type": rows("""SELECT consumer_type k, COUNT(*) total, SUM(label) flagged
                               FROM prediction GROUP BY consumer_type"""),
            "by_season": rows("""SELECT season k, COUNT(*) total, SUM(label) flagged
                                 FROM prediction GROUP BY season"""),
            "trend": rows("""SELECT date(created_at) d, COUNT(*) total, SUM(label) flagged
                             FROM prediction GROUP BY date(created_at) ORDER BY d DESC LIMIT 14"""),
            "cases": rows("SELECT status k, COUNT(*) n FROM inspection_case GROUP BY status"),
        }
        charts["trend"].reverse()
        top = db.execute(f"""SELECT c.id case_id, c.status, p.customer_id, p.meter_id, p.consumer_type,
                                    p.energy, p.risk_score, p.risk_level
                             FROM inspection_case c JOIN prediction p ON p.id = c.prediction_id
                             WHERE c.status IN ('Open','Assigned','Inspecting')
                             ORDER BY p.risk_score DESC LIMIT 8""").fetchall()
        return render_template("dashboard.html", kpi=kpi, charts=charts, top=top)

    # ------------------------------------------------------------------ #
    # Single prediction
    # ------------------------------------------------------------------ #
    DEFAULTS = {"EnergyConsumption": 45, "Voltage": 230, "Current": 0.25, "PowerFactor": 0.9,
                "AvgDemand": 60, "Temperature": 22, "Humidity": 55, "Season": "Summer",
                "ConsumerType": "Commercial"}

    @app.route("/predict", methods=["GET", "POST"])
    @login_required
    def predict():
        result, error, form = None, None, dict(DEFAULTS)
        if request.method == "POST":
            form = {k: request.form.get(k, "") for k in F.RAW_REQUIRED}
            clean, rep = F.normalize_raw(pd.DataFrame([form]))
            if clean.empty:
                error = "Invalid input: " + "; ".join(f"{k} ({v})" for k, v in rep["reasons"].items())
            else:
                res = app.predictor.predict(clean, explain="all")
                pid = save_predictions(clean, res, g.user["id"], "single")[0]
                result = {**res[0], "id": pid}
        return render_template("predict.html", result=result, error=error, form=form,
                               seasons=F.SEASONS, types=F.CONSUMER_TYPES, ranges=F.VALID_RANGES)

    # ------------------------------------------------------------------ #
    # Bulk upload
    # ------------------------------------------------------------------ #
    @app.route("/upload", methods=["GET", "POST"])
    @login_required
    def upload():
        if request.method == "POST":
            file = request.files.get("datasetfile")
            if not file or not file.filename:
                flash("Choose a CSV file first.", "danger")
                return redirect(url_for("upload"))
            if not file.filename.lower().endswith(".csv"):
                flash("Only .csv files are supported.", "danger")
                return redirect(url_for("upload"))
            try:
                df = pd.read_csv(file, encoding_errors="replace", nrows=app.config["MAX_BATCH_ROWS"] + 1)
                if len(df) > app.config["MAX_BATCH_ROWS"]:
                    raise F.ValidationError(f"File too large: max {app.config['MAX_BATCH_ROWS']} rows per upload.")
                clean, rep = F.normalize_raw(df)
                if clean.empty:
                    raise F.ValidationError("No valid rows found. " + "; ".join(
                        f"{k}: {v}" for k, v in rep["reasons"].items()))
            except (F.ValidationError, pd.errors.ParserError, pd.errors.EmptyDataError) as e:
                flash(f"Could not process file: {e}", "danger")
                return redirect(url_for("upload"))

            results = app.predictor.predict(clean, explain="flagged")
            flagged = sum(r["label"] for r in results)
            db = get_db()
            bid = db.execute("INSERT INTO batch (user_id,filename,total,flagged,skipped) VALUES (?,?,?,?,?)",
                             (g.user["id"], os.path.basename(file.filename)[:120], len(results), flagged,
                              rep["skipped"])).lastrowid
            db.commit()
            save_predictions(clean, results, g.user["id"], "batch", bid)
            if rep["skipped"]:
                flash(f"{rep['skipped']} row(s) skipped: " + "; ".join(f"{k}: {v}" for k, v in rep["reasons"].items()), "warning")
            return redirect(url_for("batch_detail", batch_id=bid))
        return render_template("upload.html", required=F.RAW_REQUIRED, optional=F.RAW_OPTIONAL)

    @app.route("/batches")
    @login_required
    def batches():
        rows = get_db().execute("""SELECT b.*, u.username FROM batch b LEFT JOIN user u ON u.id=b.user_id
                                   ORDER BY b.id DESC LIMIT 100""").fetchall()
        return render_template("batches.html", rows=rows)

    @app.route("/batch/<int:batch_id>")
    @login_required
    def batch_detail(batch_id):
        db = get_db()
        b = db.execute("SELECT * FROM batch WHERE id = ?", (batch_id,)).fetchone() or abort(404)
        level = request.args.get("level", "")
        page = max(1, request.args.get("page", 1, type=int))
        where, args = "p.batch_id = ?", [batch_id]
        if level in ("Low", "Medium", "High", "Critical"):
            where += " AND p.risk_level = ?"
            args.append(level)
        total_rows = db.execute(f"SELECT COUNT(*) FROM prediction p WHERE {where}", args).fetchone()[0]
        per = 50
        rows = db.execute(f"""SELECT p.*, c.id case_id FROM prediction p
                              LEFT JOIN inspection_case c ON c.prediction_id = p.id
                              WHERE {where}
                              ORDER BY p.risk_score DESC, p.id LIMIT ? OFFSET ?""",
                          args + [per, (page - 1) * per]).fetchall()
        dist = {r["risk_level"]: r["n"] for r in db.execute(
            "SELECT risk_level, COUNT(*) n FROM prediction WHERE batch_id = ? GROUP BY risk_level", (batch_id,))}
        by_type = [dict(r) for r in db.execute(
            """SELECT consumer_type k, COUNT(*) total, SUM(label) flagged FROM prediction
               WHERE batch_id = ? GROUP BY consumer_type""", (batch_id,))]
        cases = db.execute("""SELECT COUNT(*) FROM inspection_case c JOIN prediction p ON p.id=c.prediction_id
                              WHERE p.batch_id = ?""", (batch_id,)).fetchone()[0]
        parsed = []
        for r in rows:
            expl = json.loads(r["explanation"] or "{}").get("explanation")
            parsed.append({"row": r, "reason": (expl["reasons"][0] if expl else "")})
        return render_template("batch_detail.html", b=b, rows=parsed, dist=dist, by_type=by_type,
                               cases=cases, level=level, page=page, pages=max(1, -(-total_rows // per)),
                               total_rows=total_rows)

    @app.route("/batch/<int:batch_id>/download")
    @login_required
    def batch_download(batch_id):
        rows = get_db().execute("SELECT * FROM prediction WHERE batch_id = ? ORDER BY risk_score DESC",
                                (batch_id,)).fetchall()
        if not rows:
            abort(404)
        out = []
        for r in rows:
            inputs = json.loads(r["inputs"])
            expl = (json.loads(r["explanation"] or "{}").get("explanation") or {})
            out.append({"CustomerID": r["customer_id"], "MeterID": r["meter_id"], **inputs,
                        "RiskScore": r["risk_score"], "RiskLevel": r["risk_level"],
                        "Prediction": "Theft" if r["label"] else "Normal",
                        "AnomalyScore": r["anomaly_score"],
                        "TopReason": (expl.get("reasons") or [""])[0]})
        buf = io.StringIO()
        pd.DataFrame(out).to_csv(buf, index=False)
        return Response(buf.getvalue(), mimetype="text/csv",
                        headers={"Content-Disposition": f"attachment; filename=gridguard_batch_{batch_id}.csv"})

    @app.route("/sample.csv")
    def sample_csv():
        """Small valid example file users can download to try the upload."""
        df = pd.read_csv(app.config["DATA_PATH"]).sample(60, random_state=7)
        cols = ["CustomerID", "MeterID"] + F.RAW_REQUIRED
        buf = io.StringIO()
        df[cols].to_csv(buf, index=False)
        return Response(buf.getvalue(), mimetype="text/csv",
                        headers={"Content-Disposition": "attachment; filename=sample_upload.csv"})

    # ------------------------------------------------------------------ #
    # Inspection cases
    # ------------------------------------------------------------------ #
    @app.route("/cases")
    @login_required
    def cases():
        db = get_db()
        status = request.args.get("status", "")
        mine = request.args.get("mine") == "1"
        where, args = ["1=1"], []
        if status in CASE_STATUSES:
            where.append("c.status = ?"); args.append(status)
        if mine:
            where.append("c.assigned_to = ?"); args.append(g.user["id"])
        page = max(1, request.args.get("page", 1, type=int))
        per = 25
        total = db.execute(f"SELECT COUNT(*) FROM inspection_case c WHERE {' AND '.join(where)}", args).fetchone()[0]
        rows = db.execute(f"""SELECT c.*, p.customer_id, p.meter_id, p.consumer_type, p.energy, p.risk_score,
                                     u.username assignee
                              FROM inspection_case c JOIN prediction p ON p.id = c.prediction_id
                              LEFT JOIN user u ON u.id = c.assigned_to
                              WHERE {' AND '.join(where)}
                              ORDER BY {PRIORITY_ORDER}, p.risk_score DESC LIMIT ? OFFSET ?""",
                          args + [per, (page - 1) * per]).fetchall()
        return render_template("cases.html", rows=rows, status=status, mine=mine, page=page,
                               pages=max(1, -(-total // per)), total=total)

    @app.route("/cases/<int:case_id>", methods=["GET", "POST"])
    @login_required
    def case_detail(case_id):
        db = get_db()
        c = db.execute("""SELECT c.*, p.* , c.id AS case_id, c.status AS case_status, c.notes AS case_notes,
                                 u.username assignee
                          FROM inspection_case c JOIN prediction p ON p.id = c.prediction_id
                          LEFT JOIN user u ON u.id = c.assigned_to WHERE c.id = ?""", (case_id,)).fetchone() or abort(404)
        if request.method == "POST":
            status = request.form.get("status", "")
            if status not in CASE_STATUSES:
                abort(400)
            assignee = request.form.get("assigned_to", type=int)
            if g.user["role"] != "admin":
                assignee = g.user["id"] if assignee else c["assigned_to"]
            if assignee and not db.execute("SELECT 1 FROM user WHERE id = ?", (assignee,)).fetchone():
                abort(400)
            if status == "Open" and assignee:
                status = "Assigned"
            note = request.form.get("note", "").strip()[:1000]
            notes = c["case_notes"] or ""
            if note:
                notes += f"[{pd.Timestamp.now():%Y-%m-%d %H:%M} {g.user['username']}] {note}\n"
            outcome = 1 if status == "Confirmed Theft" else 0 if status == "False Alarm" else None
            db.execute("""UPDATE inspection_case SET status=?, assigned_to=?, notes=?, outcome_label=?,
                          updated_at=datetime('now') WHERE id=?""",
                       (status, assignee, notes, outcome, case_id))
            db.commit()
            flash("Case updated.", "success")
            return redirect(url_for("case_detail", case_id=case_id))
        expl = json.loads(c["explanation"] or "{}")
        users = db.execute("SELECT id, username FROM user ORDER BY username").fetchall()
        return render_template("case_detail.html", c=c, inputs=json.loads(c["inputs"]),
                               expl=expl.get("explanation"), model_probs=expl.get("model_probs", {}), users=users)

    # ------------------------------------------------------------------ #
    # Model page + admin
    # ------------------------------------------------------------------ #
    @app.route("/models")
    @login_required
    def models_page():
        versions = get_db().execute("SELECT * FROM model_version ORDER BY id DESC").fetchall()
        fb = get_db().execute("SELECT COUNT(*) FROM inspection_case WHERE outcome_label IS NOT NULL").fetchone()[0]
        return render_template("models.html", m=app.predictor.metrics, versions=versions, feedback=fb,
                               min_fb=app.config["MIN_FEEDBACK_FOR_RETRAIN"])

    @app.route("/admin/retrain", methods=["POST"])
    @admin_required
    def retrain():
        from ml.train import retrain_with_feedback
        db = get_db()
        rows = db.execute("""SELECT p.inputs, c.outcome_label FROM inspection_case c
                             JOIN prediction p ON p.id = c.prediction_id
                             WHERE c.outcome_label IS NOT NULL""").fetchall()
        if len(rows) < app.config["MIN_FEEDBACK_FOR_RETRAIN"]:
            flash(f"Need at least {app.config['MIN_FEEDBACK_FOR_RETRAIN']} inspector-verified cases "
                  f"(have {len(rows)}).", "warning")
            return redirect(url_for("models_page"))
        fb = pd.DataFrame([{**json.loads(r["inputs"]), "label": r["outcome_label"]} for r in rows])
        summary = retrain_with_feedback(fb, app.config["ARTIFACT_DIR"], app.config["DATA_PATH"])
        app.predictor = Predictor(app.config["ARTIFACT_DIR"])
        db.execute("INSERT INTO model_version (version,note,n_feedback,before_f1,after_f1) VALUES (?,?,?,?,?)",
                   (summary["version"], "Retrained on inspector feedback", summary["n_feedback"],
                    summary["before"]["f1"], summary["after"]["f1"]))
        db.commit()
        flash(f"Model retrained -> v{summary['version']} using {summary['n_feedback']} verified cases. "
              f"Hold-out F1: {summary['before']['f1']} -> {summary['after']['f1']}.", "success")
        return redirect(url_for("models_page"))

    @app.route("/admin/users", methods=["GET", "POST"])
    @admin_required
    def admin_users():
        db = get_db()
        if request.method == "POST":
            uid, role = request.form.get("user_id", type=int), request.form.get("role")
            if role not in ("admin", "inspector") or uid is None:
                abort(400)
            if uid == g.user["id"]:
                flash("You cannot change your own role.", "warning")
            else:
                db.execute("UPDATE user SET role = ? WHERE id = ?", (role, uid))
                db.commit()
                flash("Role updated.", "success")
            return redirect(url_for("admin_users"))
        users = db.execute("""SELECT u.*, (SELECT COUNT(*) FROM inspection_case WHERE assigned_to = u.id) cases
                              FROM user u ORDER BY u.id""").fetchall()
        return render_template("admin_users.html", users=users)

    # ------------------------------------------------------------------ #
    # REST API   (header:  X-API-Key: <key from your profile page>)
    # ------------------------------------------------------------------ #
    @app.route("/api/predict", methods=["POST"])
    def api_predict():
        key = request.headers.get("X-API-Key", "")
        owner = get_db().execute("SELECT id FROM user WHERE api_key = ?", (key,)).fetchone() if key else None
        if owner is None:
            return jsonify(error="Missing or invalid X-API-Key header"), 401
        payload = request.get_json(silent=True)
        if payload is None:
            return jsonify(error="Send a JSON body"), 400
        records = payload.get("records") if isinstance(payload, dict) and "records" in payload else \
            payload if isinstance(payload, list) else [payload]
        if not records or len(records) > 1000 or not all(isinstance(r, dict) for r in records):
            return jsonify(error="Provide 1-1000 JSON objects (or {'records': [...]})"), 400
        frame = pd.DataFrame(records)
        if not all(c in frame.columns for c in F.RAW_REQUIRED):
            return jsonify(error="Missing required fields", required=F.RAW_REQUIRED), 422
        clean, rep = F.normalize_raw(frame)
        if clean.empty:
            return jsonify(error="No valid records", details=rep["reasons"]), 422
        res = app.predictor.predict(clean, explain="all")
        save_predictions(clean, res, owner["id"], "api")
        out = [{"row": int(clean.loc[i, "_row"]) - 1,
                **{k: res[i][k] for k in ("probability", "risk_score", "risk_level", "label", "anomaly_score")},
                "prediction": "Theft" if res[i]["label"] else "Normal",
                "reasons": (res[i]["explanation"] or {}).get("reasons", [])}
               for i in range(len(res))]
        return jsonify(model_version=app.predictor.version, skipped=rep["skipped"], results=out)

    # ------------------------------------------------------------------ #
    @app.errorhandler(400)
    @app.errorhandler(403)
    @app.errorhandler(404)
    @app.errorhandler(413)
    def _err(e):
        msg = {413: "File too large (max 10 MB)."}.get(e.code, getattr(e, "description", ""))
        if request.path.startswith("/api/"):
            return jsonify(error=msg), e.code
        return render_template("error.html", code=e.code, msg=msg), e.code

    return app


def _opt(row, col):
    v = row.get(col) if hasattr(row, "get") else None
    return None if v is None or (isinstance(v, float) and pd.isna(v)) else str(v)


def _load_predictor(app: Flask) -> Predictor:
    """Load trained artifacts; train them automatically on first run if missing."""
    d = app.config["ARTIFACT_DIR"]
    try:
        return Predictor(d)
    except Exception as exc:
        print(f"[gridguard] Artifacts missing/incompatible ({exc.__class__.__name__}). Training now (~1 min)...")
        from ml.train import train_all
        train_all(app.config["DATA_PATH"], d, run_cv=False)
        return Predictor(d)


if __name__ == "__main__":
    create_app().run(debug=os.environ.get("FLASK_DEBUG") == "1",
                     host=os.environ.get("HOST", "127.0.0.1"), port=int(os.environ.get("PORT", 5000)))
