"""
Venture Forge AI - Flask Backend
Neon PostgreSQL database version.
"""

import os
import re
import json
import tempfile
import traceback
from datetime import datetime
from functools import wraps

import psycopg
from psycopg.rows import dict_row

from flask import (
    Flask, request, jsonify, render_template,
    redirect, url_for, g, session
)
from werkzeug.security import generate_password_hash, check_password_hash
from dotenv import load_dotenv

load_dotenv()

# Optional PDF support
try:
    import pymupdf as fitz
    HAS_FITZ = True
except ImportError:
    try:
        import fitz
        HAS_FITZ = True
    except ImportError:
        HAS_FITZ = False

# Optional Gemini SDK
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

# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

app = Flask(__name__)

app.secret_key = os.environ.get("SECRET_KEY") or "dev-only-change-me"

app.config.update(
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE="Lax",
    SESSION_COOKIE_SECURE=bool(os.environ.get("VERCEL")),
)

MAX_TEXT_CHARS = 15000
MAX_PDF_BYTES = 10 * 1024 * 1024

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "").strip()
DEMO_MODE = not GEMINI_API_KEY

if not DEMO_MODE and HAS_NEW_GENAI:
    _genai_client = new_genai.Client(api_key=GEMINI_API_KEY)
elif not DEMO_MODE and HAS_LEGACY_GENAI:
    genai_legacy.configure(api_key=GEMINI_API_KEY, transport="rest")
    _genai_client = None
else:
    _genai_client = None


# --------------------------------------------------
# DEMO REPORT
# --------------------------------------------------

DEMO_REPORT = {
    "research_summary": {
        "main_idea": (
            "Using machine learning to detect crop diseases early "
            "by analyzing plant leaf images."
        ),
        "technology": (
            "CNN, Transfer Learning, Computer Vision, Mobile AI"
        ),
        "problem_solved": (
            "Farmers often detect crop diseases too late, resulting "
            "in crop losses and reduced income."
        ),
        "other_application_fields": [
            "Livestock health monitoring",
            "Post-harvest quality inspection",
            "Forest disease surveillance",
            "Greenhouse automation",
            "Precision agriculture",
        ],
    },
    "startup": {
        "name": "CropSense AI",
        "description": (
            "A mobile-first platform that helps farmers identify "
            "crop diseases using photographs of plant leaves."
        ),
        "problem_statement": (
            "Farmers lack affordable and timely access to crop "
            "disease diagnosis."
        ),
        "proposed_solution": (
            "A smartphone application that identifies crop diseases "
            "and provides treatment recommendations."
        ),
        "target_customers": (
            "Farmers, agricultural cooperatives, government "
            "departments and agri-input companies."
        ),
        "unique_value_proposition": (
            "Affordable crop disease identification using smartphones."
        ),
        "revenue_model": (
            "Freemium subscriptions, B2B SaaS and agricultural partnerships."
        ),
        "mvp_features": [
            "Leaf disease detection",
            "Disease severity score",
            "Treatment recommendations",
            "Multilingual interface",
            "Farmer profile",
            "Diagnosis history",
        ],
        "challenges_and_risks": [
            "Limited smartphone access",
            "Dataset bias",
            "Incorrect disease predictions",
            "Difficulty monetising the service",
            "Competition from existing products",
        ],
        "next_steps": [
            "Collect crop leaf images",
            "Train a machine learning model",
            "Build an Android MVP",
            "Pilot with local farmers",
            "Interview farmers and agronomists",
            "Establish agricultural partnerships",
        ],
    },
}


# --------------------------------------------------
# NEON POSTGRESQL DATABASE
# --------------------------------------------------

def init_db(db=None):
    """Create database tables in Neon PostgreSQL."""

    owns_connection = db is None

    if owns_connection:
        database_url = os.environ.get("DATABASE_URL")

        if not database_url:
            raise RuntimeError(
                "DATABASE_URL is missing. Add your Neon connection "
                "string in Vercel Environment Variables."
            )

        db = psycopg.connect(
            database_url,
            row_factory=dict_row,
            connect_timeout=10,
        )

    try:
        db.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id BIGSERIAL PRIMARY KEY,
                name TEXT NOT NULL,
                email TEXT NOT NULL UNIQUE,
                password_hash TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )

        db.execute(
            """
            CREATE TABLE IF NOT EXISTS reports (
                id BIGSERIAL PRIMARY KEY,
                created_at TEXT NOT NULL,
                input_type TEXT NOT NULL,
                startup_name TEXT NOT NULL,
                snippet TEXT,
                report_json TEXT NOT NULL,
                demo_mode INTEGER NOT NULL DEFAULT 0,
                user_id BIGINT
            )
            """
        )

        # Add user_id if the reports table already exists.
        db.execute(
            "ALTER TABLE reports ADD COLUMN IF NOT EXISTS user_id BIGINT"
        )

        db.commit()

    except Exception:
        db.rollback()
        raise

    finally:
        if owns_connection:
            db.close()


def get_db():
    """Get one PostgreSQL connection for the current request."""

    if "db" not in g:
        database_url = os.environ.get("DATABASE_URL")

        if not database_url:
            raise RuntimeError(
                "DATABASE_URL is not configured in Vercel."
            )

        g.db = psycopg.connect(
            database_url,
            row_factory=dict_row,
            connect_timeout=10,
        )
        
    return g.db


@app.teardown_appcontext
def close_db(exc=None):
    db = g.pop("db", None)

    if db is not None:
        if exc is not None:
            db.rollback()

        db.close()


# --------------------------------------------------
# PDF TEXT EXTRACTION
# --------------------------------------------------

def extract_pdf_text(file_bytes):
    if not HAS_FITZ:
        raise RuntimeError("PyMuPDF is not installed.")

    with tempfile.NamedTemporaryFile(
        suffix=".pdf",
        delete=False
    ) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    try:
        doc = fitz.open(tmp_path)
        pages_text = [
            page.get_text()
            for page in doc
        ]
        doc.close()

    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

    full_text = "\n".join(pages_text).strip()

    if not full_text:
        raise ValueError(
            "No readable text found in this PDF. "
            "Try pasting the text manually."
        )

    return full_text[:MAX_TEXT_CHARS]


# --------------------------------------------------
# GEMINI AI
# --------------------------------------------------

SYSTEM_PROMPT = """
You are a startup strategy expert and technology
commercialization advisor.

Analyze the research content and return ONLY a valid JSON object
with this structure:

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

Be specific, practical and grounded in the research.
Do not return markdown or code fences.
"""



def call_gemini(text):
    if not HAS_GENAI:
        raise RuntimeError("No Gemini SDK installed.")

    prompt = (
        f"{SYSTEM_PROMPT}\n\n"
        f"--- RESEARCH CONTENT ---\n{text}"
    )

def call_gemini(text: str) -> dict:
    if not HAS_GENAI:
        raise RuntimeError("No Gemini SDK installed.")

    # Limit very long research papers to reduce API processing time.
    text = text[:20000]
    prompt = f"{SYSTEM_PROMPT}\n\n--- RESEARCH CONTENT ---\n{text}"


    model_name = "gemini-2.5-flash"

    if HAS_NEW_GENAI and _genai_client is not None:
        response = _genai_client.models.generate_content(
            model=model_name,
            contents=prompt,
            config={
                "response_mime_type": "application/json",
                "temperature": 0.2,
                "max_output_tokens": 4096,
            },
        )
        raw = response.text.strip()

    else:
        model = genai_legacy.GenerativeModel(

            model_name,
            generation_config={
                "response_mime_type": "application/json",
                "temperature": 0.2,
                "max_output_tokens": 4096,
            },
        )

        response = model.generate_content(
            prompt,
            request_options={"timeout": 60},
        )

        raw = response.text.strip()

    raw = re.sub(
        r"^```(?:json)?\s*",
        "",
        raw,
        flags=re.IGNORECASE,
    )

    raw = re.sub(r"\s*```$", "", raw)

    return json.loads(raw)



@app.route("/api/analyze", methods=["POST"])
def api_analyze():
    try:
        input_type = "text"
        text = ""

        if "pdf" in request.files and request.files["pdf"].filename:
            uploaded_file = request.files["pdf"]

            if not uploaded_file.filename.lower().endswith(".pdf"):
                return jsonify({
                    "error": "Only PDF files are accepted."
                }), 400

            pdf_bytes = uploaded_file.read()

            if len(pdf_bytes) > MAX_PDF_BYTES:
                return jsonify({
                    "error": "PDF exceeds the 10 MB size limit."
                }), 400

            try:
                text = extract_pdf_text(pdf_bytes)
            except ValueError as error:
                return jsonify({"error": str(error)}), 422

            input_type = "pdf"

        elif request.form.get("idea_text", "").strip():
            text = request.form["idea_text"].strip()
            text = text[:MAX_TEXT_CHARS]

        else:
            return jsonify({
                "error": "Please upload a PDF or enter a research idea."
            }), 400

        if DEMO_MODE:
            report_data = DEMO_REPORT
            is_demo = 1

        else:
            try:
                report_data = call_gemini(text)
                is_demo = 0

            except json.JSONDecodeError:
                return jsonify({
                    "error": "The AI returned an unexpected response. "
                             "Please try again."
                }), 502

            except Exception as error:
                app.logger.exception("Gemini API error")
                return jsonify({
                    "error": "AI API error: " + str(error)
                }), 502

        startup_name = (
            report_data.get("startup", {}).get("name")
            or report_data.get("name")
            or "Unnamed Startup"
        )

        snippet = report_data.get(
            "startup", {}
        ).get("description", "")[:200]

        db = get_db()

        cursor = db.execute(
            """
            INSERT INTO reports (
                created_at,
                input_type,
                startup_name,
                snippet,
                report_json,
                demo_mode,
                user_id
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            RETURNING id
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

        report_id = cursor.fetchone()["id"]
        db.commit()

        return jsonify({
            "report_id": report_id,
            "demo_mode": bool(is_demo),
        })

    except Exception:
        traceback.print_exc()
        return jsonify({
            "error": "An unexpected server error occurred. Please try again."
        }), 500


# --------------------------------------------------
# REPORT API
# --------------------------------------------------

@app.route("/api/reports", methods=["GET"])
def api_get_reports():
    db = get_db()

    rows = db.execute(
        """
        SELECT id, created_at, input_type, startup_name,
               snippet, demo_mode
        FROM reports
        WHERE user_id = %s
        ORDER BY id DESC
        """,
        (session["user_id"],),
    ).fetchall()

    return jsonify([dict(row) for row in rows])


@app.route("/api/reports/<int:report_id>", methods=["GET"])
def api_get_report(report_id):
    db = get_db()

    row = db.execute(
        """
        SELECT *
        FROM reports
        WHERE id = %s AND user_id = %s
        """,
        (report_id, session["user_id"]),
    ).fetchone()

    if not row:
        return jsonify({
            "error": "Report not found."
        }), 404

    data = dict(row)
    data["report_json"] = json.loads(data["report_json"])

    return jsonify(data)


@app.route("/api/reports/<int:report_id>", methods=["DELETE"])
def api_delete_report(report_id):
    db = get_db()

    db.execute(
        """
        DELETE FROM reports
        WHERE id = %s AND user_id = %s
        """,
        (report_id, session["user_id"]),
    )

    db.commit()

    return jsonify({"success": True})


# --------------------------------------------------
# WEBSITE PAGES
# --------------------------------------------------

@app.route("/")
def index():
    return render_template(
        "index.html",
        demo_mode=DEMO_MODE,
    )


@app.route("/analyze")
def analyze():
    return render_template(
        "analyze.html",
        demo_mode=DEMO_MODE,
    )


@app.route("/report/<int:report_id>")
def report(report_id):
    return render_template(
        "report.html",
        report_id=report_id,
        demo_mode=DEMO_MODE,
    )


@app.route("/history")
def history():
    return render_template(
        "history.html",
        demo_mode=DEMO_MODE,
    )


# --------------------------------------------------
# LOGIN AND SIGNUP
# --------------------------------------------------

EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def current_user():
    user_id = session.get("user_id")

    if not user_id:
        return None

    return get_db().execute(
        """
        SELECT id, name, email, created_at
        FROM users
        WHERE id = %s
        """,
        (user_id,),
    ).fetchone()


@app.context_processor
def inject_user():
    return {"current_user": current_user()}


PUBLIC_ENDPOINTS = {"login", "signup", "static"}


@app.before_request
def require_login():
    if request.endpoint in PUBLIC_ENDPOINTS or request.endpoint is None:
        return None

    user_id = session.get("user_id")

    if user_id:
        user = get_db().execute(
            "SELECT 1 FROM users WHERE id = %s",
            (user_id,),
        ).fetchone()

        if user:
            return None

    session.clear()

    if request.path.startswith("/api/"):
        return jsonify({
            "error": "Please log in first."
        }), 401

    return redirect(
        url_for("login", next=request.path)
    )


def login_required(view):
    @wraps(view)
    def wrapped(*args, **kwargs):
        if not session.get("user_id"):
            return redirect(
                url_for("login", next=request.path)
            )

        return view(*args, **kwargs)

    return wrapped


def safe_next(target):
    if target and target.startswith("/") and not target.startswith("//"):
        return target

    return url_for("index")


@app.route("/signup", methods=["GET", "POST"])
def signup():
    if session.get("user_id"):
        return redirect(url_for("index"))

    error = None
    form = {}

    if request.method == "POST":
        name = request.form.get("name", "").strip()
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        confirm = request.form.get("confirm", "")

        form = {
            "name": name,
            "email": email,
        }

        db = get_db()

        if not name or len(name) > 60:
            error = "Please enter your name (up to 60 characters)."

        elif not EMAIL_RE.match(email):
            error = "Please enter a valid email address."

        elif len(password) < 8:
            error = "Password must be at least 8 characters."

        elif password != confirm:
            error = "Passwords do not match."

        elif db.execute(
            "SELECT 1 FROM users WHERE email = %s",
            (email,),
        ).fetchone():
            error = "An account with this email already exists. Try logging in."

        else:
            cursor = db.execute(
                """
                INSERT INTO users (
                    name, email, password_hash, created_at
                )
                VALUES (%s, %s, %s, %s)
                RETURNING id
                """,
                (
                    name,
                    email,
                    generate_password_hash(password),
                    datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
                ),
            )

            user_id = cursor.fetchone()["id"]
            db.commit()

            session.clear()
            session["user_id"] = user_id

            return redirect(url_for("index"))

    return render_template(
        "signup.html",
        error=error,
        form=form,
        demo_mode=DEMO_MODE,
    )


@app.route("/login", methods=["GET", "POST"])
def login():
    if session.get("user_id"):
        return redirect(url_for("index"))

    error = None
    email = ""
    next_url = request.values.get("next", "")

    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")

        user = get_db().execute(
            "SELECT * FROM users WHERE email = %s",
            (email,),
        ).fetchone()

        if user and check_password_hash(
            user["password_hash"], password
        ):
            session.clear()
            session["user_id"] = user["id"]

            return redirect(safe_next(next_url))

        error = "Incorrect email or password."

    return render_template(
        "login.html",
        error=error,
        email=email,
        next=next_url,
        demo_mode=DEMO_MODE,
    )


@app.route("/logout", methods=["POST"])
def logout():
    session.clear()
    return redirect(url_for("index"))


# --------------------------------------------------
# USER PROFILE
# --------------------------------------------------

@app.route("/profile", methods=["GET", "POST"])
@login_required
def profile():
    user = current_user()

    if user is None:
        session.clear()
        return redirect(url_for("login"))

    db = get_db()
    message = None
    error = None

    if request.method == "POST":
        action = request.form.get("action")

        if action == "name":
            name = request.form.get("name", "").strip()

            if not name or len(name) > 60:
                error = "Please enter a name (up to 60 characters)."

            else:
                db.execute(
                    "UPDATE users SET name = %s WHERE id = %s",
                    (name, user["id"]),
                )
                db.commit()
                message = "Name updated."

        elif action == "password":
            row = db.execute(
                "SELECT password_hash FROM users WHERE id = %s",
                (user["id"],),
            ).fetchone()

            old_password = request.form.get("old_password", "")
            new_password = request.form.get("new_password", "")

            if not check_password_hash(
                row["password_hash"], old_password
            ):
                error = "Current password is incorrect."

            elif len(new_password) < 8:
                error = "New password must be at least 8 characters."

            else:
                db.execute(
                    """
                    UPDATE users
                    SET password_hash = %s
                    WHERE id = %s
                    """,
                    (
                        generate_password_hash(new_password),
                        user["id"],
                    ),
                )
                db.commit()
                message = "Password changed."

        user = current_user()

    reports = db.execute(
        """
        SELECT id, created_at, startup_name, snippet, demo_mode
        FROM reports
        WHERE user_id = %s
        ORDER BY id DESC
        """,
        (user["id"],),
    ).fetchall()

    return render_template(
        "profile.html",
        user=user,
        reports=reports,
        msg=message,
        error=error,
        demo_mode=DEMO_MODE,
    )


# --------------------------------------------------
# LOCAL DEVELOPMENT
# --------------------------------------------------

if __name__ == "__main__":
    init_db()

    print("=" * 55)
    print("Venture Forge AI")
    print(
        "Mode:",
        "DEMO (no Gemini key)" if DEMO_MODE else "LIVE (Gemini connected)"
    )
    print("URL: http://127.0.0.1:5000")
    print("=" * 55)

    app.run(debug=True, port=5000)
