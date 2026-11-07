import os
import json
import math
import time
import pickle
import secrets
import hashlib
import hmac
import smtplib
import threading
import asyncio
from contextlib import contextmanager, asynccontextmanager
from datetime import datetime, timedelta
from email.mime.text import MIMEText
from typing import Optional, Any

import httpx
import numpy as np
import mysql.connector
from mysql.connector import pooling
from pydantic import BaseModel, EmailStr
from fastapi import FastAPI, HTTPException, Header, Depends, Request, Query
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

# =====================================================================
# Configuration & Environment
# =====================================================================
DB_HOST = os.getenv("DB_HOST", "127.0.0.1")
DB_PORT = int(os.getenv("DB_PORT", "3306"))
DB_USER = os.getenv("DB_USER", "root")
DB_PASSWORD = os.getenv("DB_PASSWORD", "")
DB_NAME = os.getenv("DB_NAME", "smart_agriculture")

TOKEN_VALID_DAYS = 7

# --- SMTP CONFIGURATION (Port 587 STARTTLS for stable connection) ---
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587
SMTP_EMAIL = "greenpick69@gmail.com"
SMTP_PASSWORD = "mwgvrbcuoewzdics"  # 16 letters with NO SPACES

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "crop_recommendation_model - Copy.pkl")

# =====================================================================
# Database Setup & Connection Pool
# =====================================================================
db_pool = None
_pool_lock = threading.Lock()

def describe_db_error(e: Exception) -> str:
    errno = getattr(e, "errno", None)
    text = str(e)
    if errno == 1045 or "Access denied" in text:
        hint = "Invalid DB credentials. Verify DB_USER and DB_PASSWORD environment variables."
    elif errno in (2003, 2002) or "Can't connect" in text or "refused" in text:
        hint = f"MySQL server unreachable on {DB_HOST}:{DB_PORT}."
    else:
        hint = "Check MySQL service status and logs."
    return f"{text}\n>>> HINT: {hint}"

def init_database():
    global db_pool
    conn = mysql.connector.connect(
        host=DB_HOST, port=DB_PORT, user=DB_USER, password=DB_PASSWORD,
        connection_timeout=8,
    )
    cur = conn.cursor()
    cur.execute(
        f"CREATE DATABASE IF NOT EXISTS `{DB_NAME}` "
        f"CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci"
    )
    conn.commit()
    cur.close()
    conn.close()

    db_pool = pooling.MySQLConnectionPool(
        pool_name="agri_pool",
        pool_size=15,
        host=DB_HOST,
        port=DB_PORT,
        user=DB_USER,
        password=DB_PASSWORD,
        database=DB_NAME,
        connection_timeout=8,
    )

    with get_db() as (conn, cur):
        # Create users table
        cur.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INT AUTO_INCREMENT PRIMARY KEY,
                name VARCHAR(100) NOT NULL,
                email VARCHAR(255) NOT NULL UNIQUE,
                password_hash VARCHAR(128) NOT NULL,
                salt VARCHAR(64) NOT NULL,
                is_admin BOOLEAN DEFAULT FALSE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Schema migration: Ensure is_admin exists if table was pre-existing
        cur.execute("SHOW COLUMNS FROM users LIKE 'is_admin'")
        if not cur.fetchone():
            cur.execute("ALTER TABLE users ADD COLUMN is_admin BOOLEAN DEFAULT FALSE")

        cur.execute("""
            CREATE TABLE IF NOT EXISTS tokens (
                token VARCHAR(64) PRIMARY KEY,
                user_id INT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                expires_at DATETIME NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS history (
                id INT AUTO_INCREMENT PRIMARY KEY,
                user_id INT NOT NULL,
                action_type VARCHAR(50) NOT NULL,
                input_data JSON NULL,
                result_data JSON NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                INDEX idx_history_user (user_id, created_at),
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
            )
        """)
        
        # Seed initial admin account if no admin exists
        seed_default_admin(cur)

def seed_default_admin(cur):
    cur.execute("SELECT id FROM users WHERE is_admin = TRUE LIMIT 1")
    if not cur.fetchone():
        admin_email = "admin@greenpick.com"
        admin_pass = "Admin@123456"
        salt = secrets.token_hex(16)
        pw_hash = hash_password(admin_pass, salt)
        cur.execute(
            "INSERT INTO users (name, email, password_hash, salt, is_admin) "
            "VALUES (%s, %s, %s, %s, %s) "
            "ON DUPLICATE KEY UPDATE is_admin = TRUE",
            ("System Admin", admin_email, pw_hash, salt, True)
        )
        print(f"[ADMIN SETUP] Created default admin -> Email: {admin_email} | Password: {admin_pass}")

@contextmanager
def get_db():
    global db_pool
    if db_pool is None:
        with _pool_lock:
            if db_pool is None:
                try:
                    init_database()
                except Exception as e:
                    raise HTTPException(
                        status_code=503,
                        detail=f"Database unavailable: {describe_db_error(e)}",
                    )

    conn = None
    for _ in range(20):
        try:
            conn = db_pool.get_connection()
            break
        except mysql.connector.pooling.PoolError:
            time.sleep(0.1)

    if conn is None:
        raise HTTPException(
            status_code=503,
            detail="Database connection pool exhausted. Please retry shortly.",
        )
    cur = conn.cursor(dictionary=True)
    try:
        yield conn, cur
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        cur.close()
        conn.close()

# =====================================================================
# Lifespan Management
# =====================================================================
crop_model = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global crop_model
    try:
        await asyncio.to_thread(init_database)
        print("MySQL connected, tables verified, and admin account checked.")
    except Exception as e:
        print(f"Database initialization warning: {describe_db_error(e)}")
    
    if os.path.exists(MODEL_PATH):
        try:
            with open(MODEL_PATH, "rb") as f:
                crop_model = pickle.load(f)
            print("ML model loaded successfully.")
        except Exception as e:
            print(f"Error loading ML model: {e}")
    else:
        print(f"Warning: Model file not found at {MODEL_PATH}")
    
    yield

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.exception_handler(mysql.connector.Error)
def mysql_error_handler(request: Request, exc: mysql.connector.Error):
    return JSONResponse(status_code=503, content={"detail": f"Database error: {exc}"})

# =====================================================================
# Domain Data & Schemas
# =====================================================================
PROFIT_DATABASE = {
    "rice": {"cost_per_acre": 15000, "revenue_per_acre": 45000},
    "maize": {"cost_per_acre": 12000, "revenue_per_acre": 35000},
    "coconut": {"cost_per_acre": 20000, "revenue_per_acre": 60000},
    "banana": {"cost_per_acre": 25000, "revenue_per_acre": 85000},
    "kidneybeans": {"cost_per_acre": 30000, "revenue_per_acre": 40000},
    "chickpeas": {"cost_per_acre": 30000, "revenue_per_acre": 17000},
    "wheat": {"cost_per_acre": 50000, "revenue_per_acre": 100000},
    "sugarcane": {"cost_per_acre": 30000, "revenue_per_acre": 45000},
    "coffee": {"cost_per_acre": 38000, "revenue_per_acre": 90000},
}

class CropInput(BaseModel):
    nitrogen: float
    phosphorus: float
    potassium: float
    temperature: float
    humidity: float
    ph: float
    rainfall: float

class FertilizerInput(BaseModel):
    temperature: float
    moisture: float
    soil_type: str
    crop_type: str

class ProfitInput(BaseModel):
    crop_name: str
    land_area: float

class EmailHistoryRequest(BaseModel):
    email: EmailStr
    history_id: Optional[int] = None

class SignupInput(BaseModel):
    name: str
    email: EmailStr
    password: str

class LoginInput(BaseModel):
    email: EmailStr
    password: str

class CheckEmailInput(BaseModel):
    email: EmailStr

class ResetPasswordInput(BaseModel):
    email: EmailStr
    new_password: str

class HistoryCreate(BaseModel):
    action_type: str
    input_data: Optional[Any] = None
    result_data: Optional[Any] = None

class UserRoleUpdate(BaseModel):
    is_admin: bool

# =====================================================================
# Auth Helpers
# =====================================================================
def hash_password(password: str, salt: str) -> str:
    return hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), salt.encode("utf-8"), 100_000
    ).hex()

def _user_from_header(authorization: Optional[str]) -> Optional[dict]:
    if not authorization or not authorization.lower().startswith("bearer "):
        return None
    token = authorization.split(" ", 1)[1].strip()
    with get_db() as (conn, cur):
        cur.execute(
            "SELECT u.id, u.name, u.email, u.is_admin "
            "FROM tokens t JOIN users u ON u.id = t.user_id "
            "WHERE t.token = %s AND t.expires_at > %s",
            (token, datetime.utcnow()),
        )
        row = cur.fetchone()
    return row if row else None

def get_current_user_id(authorization: Optional[str] = Header(None)) -> int:
    user = _user_from_header(authorization)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    return user["id"]

def get_current_admin_user_id(authorization: Optional[str] = Header(None)) -> int:
    user = _user_from_header(authorization)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    if not user.get("is_admin"):
        raise HTTPException(status_code=403, detail="Access denied. Administrator privilege required.")
    return user["id"]

def get_optional_user_id(authorization: Optional[str] = Header(None)) -> Optional[int]:
    user = _user_from_header(authorization)
    return user["id"] if user else None

def save_history(user_id: Optional[int], action_type: str, input_data: Any, result_data: Any):
    if user_id is None:
        return
    try:
        with get_db() as (conn, cur):
            cur.execute(
                "INSERT INTO history (user_id, action_type, input_data, result_data) "
                "VALUES (%s, %s, %s, %s)",
                (user_id, action_type, json.dumps(input_data), json.dumps(result_data)),
            )
    except Exception as e:
        print(f"History logging failed: {e}")

# =====================================================================
# Page Delivery
# =====================================================================
ALLOWED_PAGES = {
    "homepg", "homepg2", "crop", "fertilizer", "profit", 
    "history", "history_table", "login", "signup", "signinpg", 
    "logout", "krishibhavan", "forgotpassword",
    "admin_login", "admin_dashboard"
}

def _serve(path: str, name: str):
    if not os.path.exists(path):
        raise HTTPException(
            status_code=404,
            detail=f"{name} not found at expected location.",
        )
    return FileResponse(path)

@app.get("/")
def serve_frontend():
    return _serve(os.path.join(BASE_DIR, "crop.html"), "crop.html")

@app.get("/{page}.html")
def serve_html_page(page: str):
    if page not in ALLOWED_PAGES:
        raise HTTPException(status_code=404, detail="Page not found")
    return _serve(os.path.join(BASE_DIR, f"{page}.html"), f"{page}.html")

# =====================================================================
# Application Core API Endpoints
# =====================================================================
@app.post("/predict")
def predict_crop(input_data: CropInput, user_id: Optional[int] = Depends(get_optional_user_id)):
    if crop_model is None:
        raise HTTPException(status_code=503, detail="Crop prediction model is unavailable")
    try:
        features = np.array([[
            input_data.nitrogen, input_data.phosphorus, input_data.potassium,
            input_data.temperature, input_data.humidity, input_data.ph, input_data.rainfall
        ]])
        prediction = crop_model.predict(features)
        result = {"recommended_crop": str(prediction[0])}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    save_history(user_id, "crop_prediction", input_data.model_dump(), result)
    return result

@app.post("/predict-fertilizer-db")
def predict_fertilizer(data: FertilizerInput, user_id: Optional[int] = Depends(get_optional_user_id)):
    soil = data.soil_type.strip().lower()
    crop = data.crop_type.strip().lower()
    temp = data.temperature
    moisture = data.moisture

    if soil == "sandy":
        if crop == "maize":
            recommended_fertilizer = "Urea (Nitrogen Rich)" if temp > 25 and moisture < 50 else "NPK 20:20:20"
        elif crop == "groundnuts":
            recommended_fertilizer = "Single Super Phosphate (SSP)" if moisture < 40 else "Gypsum"
        else:
            recommended_fertilizer = "Generic NPK for Sandy Soil"
    elif soil == "clayey":
        if crop == "rice":
            recommended_fertilizer = "DAP (Di-Ammonium Phosphate)" if temp > 28 and moisture > 60 else "Ammonium Sulfate"
        else:
            recommended_fertilizer = "NPK 10:26:26"
    elif soil == "loamy":
        if crop == "cotton":
            recommended_fertilizer = "MOP (Muriate of Potash)" if temp > 22 and moisture > 50 else "Urea + DAP Mix"
        elif crop == "sugarcane":
            recommended_fertilizer = "NPK 19:19:19"
        else:
            recommended_fertilizer = "Organic Compost / Vermicompost"
    elif soil == "black":
        if crop == "cotton":
            recommended_fertilizer = "Sulfate of Potash" if temp > 30 else "DAP"
        else:
            recommended_fertilizer = "NPK Complex Fertilizer"
    elif soil == "red":
        recommended_fertilizer = "Super Phosphate" if moisture < 45 else "NPK 15:15:15"
    else:
        recommended_fertilizer = "Balanced NPK (19:19:19) with Organic Manure"

    result = {"status": "success", "recommended_fertilizer": recommended_fertilizer}
    save_history(user_id, "fertilizer_prediction", data.model_dump(), result)
    return result

@app.post("/estimate-profit")
def estimate_profit(input_data: ProfitInput, user_id: Optional[int] = Depends(get_optional_user_id)):
    crop = input_data.crop_name.lower().strip()
    area = input_data.land_area

    if crop not in PROFIT_DATABASE:
        raise HTTPException(status_code=404, detail="Selected crop not found in profitability database")

    crop_data = PROFIT_DATABASE[crop]
    total_cost = crop_data["cost_per_acre"] * area
    total_revenue = crop_data["revenue_per_acre"] * area

    result = {
        "crop": crop,
        "land_area_acres": area,
        "total_cost": round(total_cost, 2),
        "total_revenue": round(total_revenue, 2),
        "net_profit": round(total_revenue - total_cost, 2),
    }

    save_history(user_id, "profit_estimate", input_data.model_dump(), result)
    return result

# =====================================================================
# Auth Routes
# =====================================================================
@app.post("/signup")
def signup(data: SignupInput):
    email = data.email.lower().strip()
    name = data.name.strip()

    if len(data.password) < 6:
        raise HTTPException(status_code=400, detail="Password must be at least 6 characters")

    salt = secrets.token_hex(16)
    pw_hash = hash_password(data.password, salt)

    try:
        with get_db() as (conn, cur):
            cur.execute("SELECT id FROM users WHERE email = %s", (email,))
            if cur.fetchone():
                raise HTTPException(status_code=409, detail="An account with this email already exists")

            cur.execute(
                "INSERT INTO users (name, email, password_hash, salt, is_admin) VALUES (%s, %s, %s, %s, %s)",
                (name, email, pw_hash, salt, False),
            )
    except mysql.connector.IntegrityError:
        raise HTTPException(status_code=409, detail="An account with this email already exists")

    return {"status": "success", "message": "Account created successfully", "user": {"name": name, "email": email}}

@app.post("/login")
def login(data: LoginInput):
    email = data.email.lower().strip()

    with get_db() as (conn, cur):
        cur.execute("SELECT id, name, password_hash, salt, is_admin FROM users WHERE email = %s", (email,))
        user = cur.fetchone()

        if user is None or not hmac.compare_digest(
            hash_password(data.password, user["salt"]), user["password_hash"]
        ):
            raise HTTPException(status_code=401, detail="Invalid email or password")

        token = secrets.token_hex(24)
        expires_at = datetime.utcnow() + timedelta(days=TOKEN_VALID_DAYS)
        cur.execute(
            "INSERT INTO tokens (token, user_id, expires_at) VALUES (%s, %s, %s)",
            (token, user["id"], expires_at),
        )

    return {
        "status": "success",
        "message": "Logged in successfully",
        "token": token,
        "user": {"name": user["name"], "email": email, "is_admin": bool(user["is_admin"])},
    }

@app.post("/logout")
def logout(authorization: Optional[str] = Header(None)):
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="Missing or malformed Authorization header")

    token = authorization.split(" ", 1)[1].strip()

    with get_db() as (conn, cur):
        cur.execute(
            "SELECT u.email FROM tokens t JOIN users u ON u.id = t.user_id WHERE t.token = %s",
            (token,),
        )
        row = cur.fetchone()
        if not row:
            raise HTTPException(status_code=401, detail="Token is invalid or already logged out")

        cur.execute("DELETE FROM tokens WHERE token = %s", (token,))

    return {"status": "success", "message": "Logged out successfully", "user": {"email": row["email"]}}

@app.post("/check-email")
def check_email(data: CheckEmailInput):
    email = data.email.lower().strip()

    with get_db() as (conn, cur):
        cur.execute("SELECT id FROM users WHERE email = %s", (email,))
        user = cur.fetchone()

    if user is None:
        raise HTTPException(status_code=404, detail="No account found with that email")

    return {"status": "success", "message": "Email verified"}

@app.post("/reset-password")
def reset_password(data: ResetPasswordInput):
    email = data.email.lower().strip()

    if len(data.new_password) < 6:
        raise HTTPException(status_code=400, detail="Password must be at least 6 characters")

    salt = secrets.token_hex(16)
    pw_hash = hash_password(data.new_password, salt)

    with get_db() as (conn, cur):
        cur.execute("SELECT id FROM users WHERE email = %s", (email,))
        user = cur.fetchone()
        if user is None:
            raise HTTPException(status_code=404, detail="No account found with that email")

        cur.execute(
            "UPDATE users SET password_hash = %s, salt = %s WHERE email = %s",
            (pw_hash, salt, email),
        )
        cur.execute("DELETE FROM tokens WHERE user_id = %s", (user["id"],))

    return {"status": "success", "message": "Password has been reset"}

# =====================================================================
# Admin Auth & Dashboard API Routes
# =====================================================================
@app.post("/admin/login")
def admin_login(data: LoginInput):
    email = data.email.lower().strip()

    with get_db() as (conn, cur):
        cur.execute("SELECT id, name, password_hash, salt, is_admin FROM users WHERE email = %s", (email,))
        user = cur.fetchone()

        if user is None or not hmac.compare_digest(
            hash_password(data.password, user["salt"]), user["password_hash"]
        ):
            raise HTTPException(status_code=401, detail="Invalid admin email or password")

        if not user["is_admin"]:
            raise HTTPException(status_code=403, detail="Access denied. Account is not an administrator.")

        token = secrets.token_hex(24)
        expires_at = datetime.utcnow() + timedelta(days=TOKEN_VALID_DAYS)
        cur.execute(
            "INSERT INTO tokens (token, user_id, expires_at) VALUES (%s, %s, %s)",
            (token, user["id"], expires_at),
        )

    return {
        "status": "success",
        "message": "Admin authentication successful",
        "token": token,
        "user": {"name": user["name"], "email": email, "is_admin": True},
    }

@app.get("/admin/stats")
def admin_get_stats(admin_id: int = Depends(get_current_admin_user_id)):
    with get_db() as (conn, cur):
        cur.execute("SELECT COUNT(*) AS total FROM users")
        total_users = cur.fetchone()["total"]

        cur.execute("SELECT COUNT(*) AS total FROM tokens WHERE expires_at > %s", (datetime.utcnow(),))
        active_sessions = cur.fetchone()["total"]

        cur.execute("SELECT COUNT(*) AS total FROM history")
        total_predictions = cur.fetchone()["total"]

        cur.execute("""
            SELECT action_type, COUNT(*) AS count 
            FROM history 
            GROUP BY action_type
        """)
        action_breakdown = cur.fetchall()

    return {
        "status": "success",
        "stats": {
            "total_users": total_users,
            "active_sessions": active_sessions,
            "total_predictions": total_predictions,
            "action_breakdown": {row["action_type"]: row["count"] for row in action_breakdown}
        }
    }

@app.get("/admin/users")
def admin_get_users(
    query: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
    admin_id: int = Depends(get_current_admin_user_id)
):
    with get_db() as (conn, cur):
        if query:
            q = f"%{query.strip().lower()}%"
            cur.execute(
                "SELECT id, name, email, is_admin, created_at FROM users "
                "WHERE LOWER(name) LIKE %s OR LOWER(email) LIKE %s "
                "ORDER BY id DESC LIMIT %s OFFSET %s",
                (q, q, limit, offset)
            )
            users = cur.fetchall()
            cur.execute(
                "SELECT COUNT(*) AS total FROM users WHERE LOWER(name) LIKE %s OR LOWER(email) LIKE %s",
                (q, q)
            )
            total = cur.fetchone()["total"]
        else:
            cur.execute(
                "SELECT id, name, email, is_admin, created_at FROM users "
                "ORDER BY id DESC LIMIT %s OFFSET %s",
                (limit, offset)
            )
            users = cur.fetchall()
            cur.execute("SELECT COUNT(*) AS total FROM users")
            total = cur.fetchone()["total"]

    formatted_users = [
        {
            "id": u["id"],
            "name": u["name"],
            "email": u["email"],
            "is_admin": bool(u["is_admin"]),
            "created_at": u["created_at"].isoformat() if u["created_at"] else None
        }
        for u in users
    ]

    return {"status": "success", "total": total, "users": formatted_users}

@app.put("/admin/users/{user_id}/role")
def admin_update_user_role(
    user_id: int, 
    role_data: UserRoleUpdate, 
    admin_id: int = Depends(get_current_admin_user_id)
):
    if user_id == admin_id and not role_data.is_admin:
        raise HTTPException(status_code=400, detail="You cannot revoke your own admin rights.")

    with get_db() as (conn, cur):
        cur.execute("UPDATE users SET is_admin = %s WHERE id = %s", (role_data.is_admin, user_id))
        if cur.rowcount == 0:
            raise HTTPException(status_code=404, detail="Target user not found")

    return {"status": "success", "message": f"User role updated successfully (is_admin={role_data.is_admin})"}

@app.delete("/admin/users/{user_id}")
def admin_delete_user(user_id: int, admin_id: int = Depends(get_current_admin_user_id)):
    if user_id == admin_id:
        raise HTTPException(status_code=400, detail="You cannot delete your own admin account.")

    with get_db() as (conn, cur):
        cur.execute("DELETE FROM users WHERE id = %s", (user_id,))
        if cur.rowcount == 0:
            raise HTTPException(status_code=404, detail="Target user not found")

    return {"status": "success", "message": "User account and related data purged successfully"}

@app.get("/admin/history")
def admin_get_history(
    user_id: Optional[int] = None,
    action_type: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
    admin_id: int = Depends(get_current_admin_user_id)
):
    where_clauses = []
    params = []

    if user_id:
        where_clauses.append("h.user_id = %s")
        params.append(user_id)
    if action_type:
        where_clauses.append("h.action_type = %s")
        params.append(action_type)

    where_sql = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""

    with get_db() as (conn, cur):
        sql = f"""
            SELECT h.id, h.user_id, u.name as user_name, u.email as user_email, 
                   h.action_type, h.input_data, h.result_data, h.created_at
            FROM history h
            JOIN users u ON u.id = h.user_id
            {where_sql}
            ORDER BY h.created_at DESC, h.id DESC
            LIMIT %s OFFSET %s
        """
        cur.execute(sql, (*params, limit, offset))
        rows = cur.fetchall()

    history = [
        {
            "id": r["id"],
            "user_id": r["user_id"],
            "user_name": r["user_name"],
            "user_email": r["user_email"],
            "action_type": r["action_type"],
            "input_data": json.loads(r["input_data"]) if r["input_data"] else None,
            "result_data": json.loads(r["result_data"]) if r["result_data"] else None,
            "timestamp": r["created_at"].isoformat(),
        }
        for r in rows
    ]

    return {"status": "success", "history": history}

@app.delete("/admin/history")
def admin_clear_history(
    history_id: Optional[int] = None, 
    admin_id: int = Depends(get_current_admin_user_id)
):
    with get_db() as (conn, cur):
        if history_id:
            cur.execute("DELETE FROM history WHERE id = %s", (history_id,))
            msg = f"History record #{history_id} deleted."
        else:
            cur.execute("DELETE FROM history")
            msg = "All activity logs purged system-wide."
        deleted = cur.rowcount

    return {"status": "success", "message": msg, "deleted": deleted}

# =====================================================================
# User History Routes
# =====================================================================
@app.post("/history/add")
def add_history(data: HistoryCreate, user_id: int = Depends(get_current_user_id)):
    with get_db() as (conn, cur):
        cur.execute(
            "INSERT INTO history (user_id, action_type, input_data, result_data) VALUES (%s, %s, %s, %s)",
            (user_id, data.action_type, json.dumps(data.input_data), json.dumps(data.result_data)),
        )
        new_id = cur.lastrowid

    return {"status": "success", "history": {"id": new_id, "action_type": data.action_type}}

@app.get("/history")
def get_history(user_id: int = Depends(get_current_user_id)):
    with get_db() as (conn, cur):
        cur.execute(
            "SELECT id, action_type, input_data, result_data, created_at "
            "FROM history WHERE user_id = %s ORDER BY created_at DESC, id DESC",
            (user_id,),
        )
        rows = cur.fetchall()

    history = [
        {
            "id": r["id"],
            "action_type": r["action_type"],
            "input_data": json.loads(r["input_data"]) if r["input_data"] else None,
            "result_data": json.loads(r["result_data"]) if r["result_data"] else None,
            "timestamp": r["created_at"].isoformat(),
        }
        for r in rows
    ]

    return {"status": "success", "history": history}

@app.delete("/history")
def clear_history(user_id: int = Depends(get_current_user_id)):
    with get_db() as (conn, cur):
        cur.execute("DELETE FROM history WHERE user_id = %s", (user_id,))
        deleted = cur.rowcount

    return {"status": "success", "message": "History cleared", "deleted": deleted}

@app.delete("/history/{history_id}")
def delete_history(history_id: int, user_id: int = Depends(get_current_user_id)):
    with get_db() as (conn, cur):
        cur.execute("DELETE FROM history WHERE id = %s AND user_id = %s", (history_id, user_id))
        deleted = cur.rowcount

    if deleted == 0:
        raise HTTPException(status_code=404, detail="History entry not found")

    return {"status": "success", "message": "History entry deleted"}

# =====================================================================
# SMTP History Delivery
# =====================================================================
FIELD_LABELS = {
    "nitrogen": "Nitrogen (N)", "phosphorus": "Phosphorus (P)", "potassium": "Potassium (K)",
    "temperature": "Temperature (°C)", "humidity": "Humidity (%)", "ph": "pH", "rainfall": "Rainfall (mm)",
    "moisture": "Moisture", "soil_type": "Soil type", "crop_type": "Crop type",
    "crop_name": "Crop", "land_area": "Land area (acres)",
}

ACTION_TITLES = {
    "crop_prediction": "Crop Recommendation",
    "fertilizer_prediction": "Fertilizer Recommendation",
    "profit_estimate": "Profit Estimation",
}

def format_history_item(action_type: str, input_data: Any, result_data: Any) -> list:
    lines = [f"--- {ACTION_TITLES.get(action_type, 'GreenPick Record')} ---", "Input Data:"]
    if isinstance(input_data, dict):
        for k, v in input_data.items():
            lines.append(f"  • {FIELD_LABELS.get(k, k)}: {v}")
            
    lines.append("Result:")
    if action_type == "crop_prediction" and isinstance(result_data, dict):
        lines.append(f"  Recommended Crop: {result_data.get('recommended_crop')}")
    elif action_type == "fertilizer_prediction" and isinstance(result_data, dict):
        lines.append(f"  Recommended Fertilizer: {result_data.get('recommended_fertilizer')}")
    elif action_type == "profit_estimate" and isinstance(result_data, dict):
        lines.append(f"  Total Cost: Rs. {result_data.get('total_cost')}")
        lines.append(f"  Total Revenue: Rs. {result_data.get('total_revenue')}")
        lines.append(f"  Net Profit: Rs. {result_data.get('net_profit')}")
    else:
        lines.append(f"  {result_data}")
        
    return lines

@app.post("/history/email")
def email_history_record(data: EmailHistoryRequest, user_id: int = Depends(get_current_user_id)):
    print(f"[DEBUG] SMTP CHECK: EMAIL='{SMTP_EMAIL}', HAS_PASS={bool(SMTP_PASSWORD)}")

    if not SMTP_EMAIL or not SMTP_PASSWORD or "your_actual_email" in SMTP_EMAIL or "your_16_char_app_pass" in SMTP_PASSWORD:
        raise HTTPException(
            status_code=503,
            detail="SMTP parameters are not configured in main.py. Please update SMTP_EMAIL and SMTP_PASSWORD.",
        )

    with get_db() as (conn, cur):
        if data.history_id:
            cur.execute(
                "SELECT id, action_type, input_data, result_data, created_at FROM history WHERE id = %s AND user_id = %s",
                (data.history_id, user_id),
            )
            rows = cur.fetchall()
            if not rows:
                raise HTTPException(status_code=404, detail="Selected history record was not found.")
        else:
            cur.execute(
                "SELECT id, action_type, input_data, result_data, created_at FROM history WHERE user_id = %s ORDER BY created_at DESC",
                (user_id,),
            )
            rows = cur.fetchall()
            if not rows:
                raise HTTPException(status_code=404, detail="No history records found to email.")

    lines = ["Hello,", "", "Here is your requested recommendation record from GreenPick:", ""]
    for row in rows:
        action_type = row["action_type"]
        input_data = json.loads(row["input_data"]) if row["input_data"] else {}
        result_data = json.loads(row["result_data"]) if row["result_data"] else {}
        timestamp = row["created_at"].strftime("%B %d, %Y - %I:%M %p")

        lines.append(f"Date: {timestamp}")
        lines.extend(format_history_item(action_type, input_data, result_data))
        lines.append("-" * 40)

    lines.extend(["", "Thank you for using GreenPick!", "- GreenPick Team"])

    msg = MIMEText("\n".join(lines), "plain", "utf-8")
    msg["Subject"] = f"GreenPick Record (ID: #{data.history_id})" if data.history_id else "Your GreenPick History Report"
    msg["From"] = SMTP_EMAIL
    msg["To"] = data.email

    try:
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT, timeout=15) as server:
            server.ehlo()
            server.starttls()
            server.ehlo()
            server.login(SMTP_EMAIL, SMTP_PASSWORD.replace(" ", ""))
            server.sendmail(SMTP_EMAIL, [data.email], msg.as_string())
    except Exception as e:
        print(f"[SMTP Error] Details: {e}")
        raise HTTPException(status_code=502, detail=f"Failed to send email: {e}")

    return {"status": "success", "message": f"History sent successfully to {data.email}"}

# =====================================================================
# Geo / Overpass API
# =====================================================================
def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))

@app.get("/krishi-bhavans")
async def nearby_krishi_bhavans(lat: float, lon: float, radius_km: float = 15):
    radius_m = int(max(1, min(radius_km, 50)) * 1000)
    query = f"""
    [out:json][timeout:15];
    (
      node["name"~"Krishi ?Bhavan",i](around:{radius_m},{lat},{lon});
      way["name"~"Krishi ?Bhavan",i](around:{radius_m},{lat},{lon});
    );
    out center;
    """
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.post("https://overpass-api.de/api/interpreter", data={"data": query})
            resp.raise_for_status()
            elements = resp.json().get("elements", [])
    except Exception as e:
        raise HTTPException(
            status_code=502,
            detail=f"Map data service unreachable: {e}",
        )

    results = []
    for el in elements:
        if el.get("type") == "node":
            la, lo = el.get("lat"), el.get("lon")
        else:
            center = el.get("center") or {}
            la, lo = center.get("lat"), center.get("lon")
        if la is None or lo is None:
            continue
        results.append({
            "name": (el.get("tags") or {}).get("name", "Krishi Bhavan"),
            "lat": la,
            "lon": lo,
            "distance_km": round(_haversine_km(lat, lon, la, lo), 1),
        })

    results.sort(key=lambda r: r["distance_km"])
    return {"status": "success", "count": len(results), "krishi_bhavans": results[:25]}