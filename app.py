"""
Venture Forge AI – Flask Backend
Converts research papers / ideas into structured startup concept reports.
"""

import os
import re
import json
import sqlite3
import tempfile
import traceback
from datetime import datetime
from functools import wraps
from pathlib import Path

from flask import Flask, request, jsonify, render_template, redirect, url_for, g, session
from werkzeug.security import generate_password_hash, check_password_hash
from dotenv import load_dotenv

load_dotenv()

# ── Optional heavy imports (fail gracefully) ───────────────────────────────
try:
    import pymupdf as fitz  # PyMuPDF 1.25+ preferred API
    HAS_FITZ = True
except ImportError:
    try:
        import fitz  # fallback for older versions
        HAS_FITZ = True
    except ImportError:
        HAS_FITZ = False

try:
    from google import genai as new_genai
    HAS_NEW_GENAI = True
except ImportError:
    HAS_NEW_GENAI = False

try:
    import google.generativeai as genai_legacy
    HAS_LEGACY_GENAI = True
except ImportError:
    HAS_LEGACY_GENAI = False

HAS_GENAI = HAS_NEW_GENAI or HAS_LEGACY_GENAI

# ── Configuration ──────────────────────────────────────────────────────────
app = Flask(__name__)
# Set SECRET_KEY in .env (any long random string). Needed to keep login sessions safe.
app.secret_key = os.environ.get("SECRET_KEY") or "dev-only-change-me"
app.config.update(SESSION_COOKIE_HTTPONLY=True, SESSION_COOKIE_SAMESITE="Lax")
DATABASE = Path("/tmp/database.db") if os.environ.get("VERCEL") else Path(__file__).parent / "database.db"
MAX_TEXT_CHARS = 15_000  # send at most this many chars to Gemini
MAX_PDF_BYTES = 10 * 1024 * 1024  # 10 MB

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "").strip()
DEMO_MODE = not GEMINI_API_KEY

if not DEMO_MODE and HAS_NEW_GENAI:
    _genai_client = new_genai.Client(api_key=GEMINI_API_KEY)
elif not DEMO_MODE and HAS_LEGACY_GENAI:
    genai_legacy.configure(api_key=GEMINI_API_KEY, transport="rest")
    _genai_client = None
else:
    _genai_client = None

# ── Demo-mode sample report ────────────────────────────────────────────────
DEMO_REPORT = {
    "research_summary": {
        "main_idea": (
            "Using machine learning (specifically convolutional neural networks) "
            "to detect crop diseases at an early stage by analyzing images of "
            "plant leaves captured by smartphones or drones."
        ),
        "technology": (
            "Convolutional Neural Networks (CNN), Transfer Learning (ResNet / MobileNet), "
            "Computer Vision, Mobile Image Processing, Edge AI"
        ),
        "problem_solved": (
            "Farmers often detect crop diseases too late, leading to large-scale crop "
            "loss, reduced income, and food insecurity. Traditional diagnosis requires "
            "agronomists who are expensive and scarce in rural areas."
        ),
        "other_application_fields": [
            "Livestock health monitoring",
            "Post-harvest quality inspection",
            "Forest disease and pest surveillance",
            "Greenhouse automation",
            "Precision agriculture dashboards",
        ],
    },
    "startup": {
        "name": "CropSense AI",
        "description": (
            "CropSense AI is a mobile-first platform that empowers farmers and agronomists "
            "to instantly identify crop diseases by photographing a leaf. Powered by a "
            "lightweight on-device ML model, the app works even without internet "
            "connectivity, making it ideal for rural and remote farming communities."
        ),
        "problem_statement": (
            "Over 20-40 % of global crop production is lost annually to pests and diseases. "
            "Smallholder farmers in developing countries lack access to timely, affordable "
            "agronomic expertise, resulting in delayed treatment and devastating yield losses."
        ),
        "proposed_solution": (
            "A smartphone app + web dashboard that lets farmers photograph affected crops. "
            "The on-device CNN model identifies the disease, severity level, and suggests "
            "treatment options in local languages. Agronomists can review flagged cases "
            "remotely, creating a hybrid AI + expert telemedicine model for agriculture."
        ),
        "target_customers": (
            "Smallholder and mid-scale farmers in South Asia, Sub-Saharan Africa, and "
            "Southeast Asia; agricultural cooperatives; government agriculture departments; "
            "agri-input distributors and pesticide companies."
        ),
        "unique_value_proposition": (
            "The only crop disease detection tool that works fully offline on a $50 "
            "Android phone, supports 12 regional languages, and connects farmers directly "
            "to vetted agronomists — all in under 30 seconds per diagnosis."
        ),
        "revenue_model": (
            "Freemium mobile app (basic disease ID free; premium treatment plans & expert "
            "chat subscription ₹199/month). B2B SaaS for agri-input companies wanting "
            "hyper-local disease data. Government contracts for early-warning surveillance "
            "dashboards. Affiliate revenue from recommended agri-input products."
        ),
        "mvp_features": [
            "Camera-based leaf disease detection (10 most common crops, 30 diseases)",
            "Offline inference using a quantized MobileNet model",
            "Disease severity score (mild / moderate / severe)",
            "Treatment recommendations with locally available products",
            "Multilingual UI (English + 3 regional languages at launch)",
            "Basic farmer profile and diagnosis history",
        ],
        "challenges_and_risks": [
            "Low smartphone penetration in the most rural areas",
            "Dataset bias – model may underperform on local crop varieties",
            "Regulatory and liability concerns around automated medical/agronomic advice",
            "Monetisation difficulty with low-income smallholder farmers",
            "Competition from well-funded agri-tech incumbents (Plantix, FarmSense)",
        ],
        "next_steps": [
            "Partner with 2–3 agricultural universities to build a labeled dataset of 50,000+ leaf images",
            "Train and benchmark MobileNetV3 vs EfficientNet-Lite for on-device accuracy",
            "Build a functional Android MVP and pilot with 200 farmers in one district",
            "Apply for DST-AGRI / NABARD / Bill & Melinda Gates Foundation grants",
            "Conduct structured interviews with 50 farmers and 10 agronomists to validate willingness-to-pay",
            "Establish an advisory board with one agronomist, one ML researcher, and one rural entrepreneur",
        ],
    },
}

# ── Database ───────────────────────────────────────────────────────────────
def get_db():
    if "db" not in g:
        if os.environ.get("VERCEL") and not Path(DATABASE).exists():
            init_db()
        g.db = sqlite3.connect(DATABASE)
        g.db.row_factory = sqlite3.Row
    return g.db

@app.teardown_appcontext
def close_db(exc=None):
    db = g.pop("db", None)
    if db is not None:
        db.close()

def init_db():
    db = sqlite3.connect(DATABASE)
    db.execute(
        """
        CREATE TABLE IF NOT EXISTS reports (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            input_type TEXT NOT NULL,
            startup_name TEXT NOT NULL,
            snippet    TEXT,
            report_json TEXT NOT NULL,
            demo_mode  INTEGER NOT NULL DEFAULT 0
        )
        """
    )
    db.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            name          TEXT NOT NULL,
            email         TEXT NOT NULL UNIQUE,
            password_hash TEXT NOT NULL,
            created_at    TEXT NOT NULL
        )
        """
    )
    # Older databases don't have reports.user_id yet - add it once.
    cols = [r[1] for r in db.execute("PRAGMA table_info(reports)").fetchall()]
    if "user_id" not in cols:
        db.execute("ALTER TABLE reports ADD COLUMN user_id INTEGER")
    db.commit()
    db.close()

# ── PDF extraction ─────────────────────────────────────────────────────────
def extract_pdf_text(file_bytes: bytes) -> str:
    if not HAS_FITZ:
        raise RuntimeError("PyMuPDF is not installed.")
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name
    try:
        doc = fitz.open(tmp_path)
        pages_text = []
        for page in doc:
            pages_text.append(page.get_text())
        doc.close()
    finally:
        os.unlink(tmp_path)
    full_text = "\n".join(pages_text).strip()
    if not full_text:
        raise ValueError(
            "No readable text found in the PDF. "
            "This may be a scanned / image-only PDF. "
            "Please try pasting the text manually in the textarea."
        )
    return full_text[:MAX_TEXT_CHARS]

# ── Gemini prompt & call ───────────────────────────────────────────────────
SYSTEM_PROMPT = """You are a startup strategy expert and technology commercialization advisor.
Analyze the research content provided and return ONLY a valid JSON object (no markdown, no code fences, no extra text) with exactly this structure:

{
  "research_summary": {
    "main_idea": "...",
    "technology": "...",
    "problem_solved": "...",
    "other_application_fields": ["...", "..."]
  },
  "startup": {
    "name": "...",
    "description": "...",
    "problem_statement": "...",
    "proposed_solution": "...",
    "target_customers": "...",
    "unique_value_proposition": "...",
    "revenue_model": "...",
    "mvp_features": ["...", "..."],
    "challenges_and_risks": ["...", "..."],
    "next_steps": ["...", "..."]
  }
}

Guidelines:
- startup.name: a catchy, memorable startup name (2-3 words max).
- mvp_features: 5-7 concrete features for version 1.
- challenges_and_risks: 4-5 realistic risks.
- next_steps: 5-6 actionable steps numbered logically.
- Be specific, practical, and grounded in the research content.
- Do NOT output anything other than the JSON object.
"""

def call_gemini(text: str) -> dict:
    if not HAS_GENAI:
        raise RuntimeError("No Gemini SDK installed.")

    prompt = f"{SYSTEM_PROMPT}\n\n--- RESEARCH CONTENT ---\n{text}"

    if HAS_NEW_GENAI and _genai_client is not None:
        # New google-genai SDK
        response = _genai_client.models.generate_content(
            model="gemini-flash-latest",
            contents=prompt,
        )
        raw = response.text.strip()
    else:
        # Legacy google-generativeai SDK
        model = genai_legacy.GenerativeModel("gemini-flash-latest")
        response = model.generate_content(prompt, request_options={"timeout": 90})
        raw = response.text.strip()

    # strip markdown code fences if present
    raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.IGNORECASE)
    raw = re.sub(r"\s*```$", "", raw)
    return json.loads(raw)

# ── API routes ─────────────────────────────────────────────────────────────
@app.route("/api/analyze", methods=["POST"])
def api_analyze():
    try:
        input_type = "text"
        text = ""

        if "pdf" in request.files and request.files["pdf"].filename:
            f = request.files["pdf"]
            if not f.filename.lower().endswith(".pdf"):
                return jsonify({"error": "Only PDF files are accepted."}), 400
            pdf_bytes = f.read()
            if len(pdf_bytes) > MAX_PDF_BYTES:
                return jsonify({"error": "PDF exceeds the 10 MB size limit."}), 400
            try:
                text = extract_pdf_text(pdf_bytes)
            except ValueError as ve:
                return jsonify({"error": str(ve)}), 422
            input_type = "pdf"
        elif request.form.get("idea_text", "").strip():
            text = request.form["idea_text"].strip()[:MAX_TEXT_CHARS]
            input_type = "text"
        else:
            return jsonify({"error": "Please upload a PDF or enter a research idea."}), 400

        # Generate report
        if DEMO_MODE:
            report_data = DEMO_REPORT
            is_demo = 1
        else:
            try:
                report_data = call_gemini(text)
                is_demo = 0
            except json.JSONDecodeError:
                return jsonify({
                    "error": (
                        "The AI returned an unexpected response. "
                        "Please try again or simplify your input."
                    )
                }), 502
            except Exception as api_err:
                return jsonify({"error": f"AI API error: {str(api_err)}"}), 502

        # Validate minimal structure
        startup_name = (
            report_data.get("startup", {}).get("name")
            or report_data.get("name")
            or "Unnamed Startup"
        )
        snippet = (
            report_data.get("startup", {}).get("description", "")[:200]
        )

        db = get_db()
        cur = db.execute(
            """
            INSERT INTO reports (created_at, input_type, startup_name, snippet, report_json, demo_mode, user_id)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
                input_type,
                startup_name,
                snippet,
                json.dumps(report_data),
                is_demo,
                session.get("user_id"),
            ),
        )
        db.commit()
        report_id = cur.lastrowid

        return jsonify({"report_id": report_id, "demo_mode": bool(is_demo)})

    except Exception:
        traceback.print_exc()
        return jsonify({"error": "An unexpected server error occurred. Please try again."}), 500


@app.route("/api/reports", methods=["GET"])
def api_get_reports():
    db = get_db()
    rows = db.execute(
        "SELECT id, created_at, input_type, startup_name, snippet, demo_mode "
        "FROM reports WHERE user_id = ? ORDER BY id DESC", (session["user_id"],)
    ).fetchall()
    return jsonify([dict(r) for r in rows])


@app.route("/api/reports/<int:report_id>", methods=["GET"])
def api_get_report(report_id):
    db = get_db()
    row = db.execute(
        "SELECT * FROM reports WHERE id = ? AND user_id = ?", (report_id, session["user_id"])
    ).fetchone()
    if not row:
        return jsonify({"error": "Report not found."}), 404
    data = dict(row)
    data["report_json"] = json.loads(data["report_json"])
    return jsonify(data)


@app.route("/api/reports/<int:report_id>", methods=["DELETE"])
def api_delete_report(report_id):
    db = get_db()
    db.execute("DELETE FROM reports WHERE id = ? AND user_id = ?", (report_id, session["user_id"]))
    db.commit()
    return jsonify({"success": True})


# ── Page routes ────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html", demo_mode=DEMO_MODE)

@app.route("/analyze")
def analyze():
    return render_template("analyze.html", demo_mode=DEMO_MODE)

@app.route("/report/<int:report_id>")
def report(report_id):
    return render_template("report.html", report_id=report_id, demo_mode=DEMO_MODE)

@app.route("/history")
def history():
    return render_template("history.html", demo_mode=DEMO_MODE)


# ── Authentication ─────────────────────────────────────────────────────────
EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")

def current_user():
    uid = session.get("user_id")
    if not uid:
        return None
    return get_db().execute(
        "SELECT id, name, email, created_at FROM users WHERE id = ?", (uid,)
    ).fetchone()

@app.context_processor
def inject_user():
    return {"current_user": current_user()}

# Pages anyone can open without logging in
PUBLIC_ENDPOINTS = {"login", "signup", "static"}

@app.before_request
def require_login():
    if request.endpoint in PUBLIC_ENDPOINTS or request.endpoint is None:
        return None
    uid = session.get("user_id")
    if uid and get_db().execute("SELECT 1 FROM users WHERE id = ?", (uid,)).fetchone():
        return None
    session.clear()  # logged out, or account no longer exists
    if request.path.startswith("/api/"):
        return jsonify({"error": "Please log in first."}), 401
    return redirect(url_for("login", next=request.path))

def login_required(view):
    @wraps(view)
    def wrapped(*args, **kwargs):
        if not session.get("user_id"):
            return redirect(url_for("login", next=request.path))
        return view(*args, **kwargs)
    return wrapped

def safe_next(target):
    # only allow redirects to pages inside this site
    if target and target.startswith("/") and not target.startswith("//"):
        return target
    return url_for("index")

@app.route("/signup", methods=["GET", "POST"])
def signup():
    if session.get("user_id"):
        return redirect(url_for("index"))
    error, form = None, {}
    if request.method == "POST":
        name = request.form.get("name", "").strip()
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        confirm = request.form.get("confirm", "")
        form = {"name": name, "email": email}
        db = get_db()
        if not name or len(name) > 60:
            error = "Please enter your name (up to 60 characters)."
        elif not EMAIL_RE.match(email):
            error = "Please enter a valid email address."
        elif len(password) < 8:
            error = "Password must be at least 8 characters."
        elif password != confirm:
            error = "Passwords do not match."
        elif db.execute("SELECT 1 FROM users WHERE email = ?", (email,)).fetchone():
            error = "An account with this email already exists. Try logging in."
        else:
            cur = db.execute(
                "INSERT INTO users (name, email, password_hash, created_at) VALUES (?, ?, ?, ?)",
                (name, email, generate_password_hash(password),
                 datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")),
            )
            db.commit()
            session.clear()
            session["user_id"] = cur.lastrowid
            return redirect(url_for("index"))
    return render_template("signup.html", error=error, form=form, demo_mode=DEMO_MODE)

@app.route("/login", methods=["GET", "POST"])
def login():
    if session.get("user_id"):
        return redirect(url_for("index"))
    error, email = None, ""
    nxt = request.values.get("next", "")
    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        user = get_db().execute("SELECT * FROM users WHERE email = ?", (email,)).fetchone()
        if user and check_password_hash(user["password_hash"], password):
            session.clear()
            session["user_id"] = user["id"]
            return redirect(safe_next(nxt))
        error = "Incorrect email or password."
    return render_template("login.html", error=error, email=email, next=nxt, demo_mode=DEMO_MODE)

@app.route("/logout", methods=["POST"])
def logout():
    session.clear()
    return redirect(url_for("index"))

@app.route("/profile", methods=["GET", "POST"])
@login_required
def profile():
    user = current_user()
    if user is None:  # account was deleted but cookie remained
        session.clear()
        return redirect(url_for("login"))
    db = get_db()
    msg = error = None
    if request.method == "POST":
        action = request.form.get("action")
        if action == "name":
            name = request.form.get("name", "").strip()
            if not name or len(name) > 60:
                error = "Please enter a name (up to 60 characters)."
            else:
                db.execute("UPDATE users SET name = ? WHERE id = ?", (name, user["id"]))
                db.commit()
                msg = "Name updated."
        elif action == "password":
            row = db.execute("SELECT password_hash FROM users WHERE id = ?", (user["id"],)).fetchone()
            old = request.form.get("old_password", "")
            new = request.form.get("new_password", "")
            if not check_password_hash(row["password_hash"], old):
                error = "Current password is incorrect."
            elif len(new) < 8:
                error = "New password must be at least 8 characters."
            else:
                db.execute("UPDATE users SET password_hash = ? WHERE id = ?",
                           (generate_password_hash(new), user["id"]))
                db.commit()
                msg = "Password changed."
        user = current_user()
    reports = db.execute(
        "SELECT id, created_at, startup_name, snippet, demo_mode FROM reports "
        "WHERE user_id = ? ORDER BY id DESC", (user["id"],)
    ).fetchall()
    return render_template("profile.html", user=user, reports=reports,
                           msg=msg, error=error, demo_mode=DEMO_MODE)


# ── Entry point ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    init_db()
    print("=" * 55)
    print("  Venture Forge AI")
    print(f"  Mode: {'DEMO (no Gemini key)' if DEMO_MODE else 'LIVE (Gemini connected)'}")
    print("  URL : http://127.0.0.1:5000")
    print("=" * 55)
    app.run(debug=True, port=5000)
