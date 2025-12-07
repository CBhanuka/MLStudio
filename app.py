from __future__ import annotations
import os
import sys
from pathlib import Path
import io
import json
import uuid
import time, math
import sqlite3
import joblib
import hashlib
import secrets
from datetime import datetime, timezone
from typing import List, Dict, Any, Tuple
from flask import send_from_directory

from flask import Flask, request, jsonify, send_file, abort
from flask_cors import CORS
import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    r2_score,
    mean_absolute_error,
    silhouette_score,
)
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.inspection import permutation_importance
from flask import Flask, request, jsonify, render_template

from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor

# Advanced ML libraries
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.model_selection import RandomizedSearchCV, cross_val_score
from sklearn.model_selection import KFold


# ---------- Config ----------
# BASE_DIR = os.path.abspath(os.path.dirname(__file__))
# STORE_DIR = os.path.join(BASE_DIR, "storage")
# MODELS_DIR = os.path.join(STORE_DIR, "models")
# DATASETS_DIR = os.path.join(STORE_DIR, "datasets")
# DB_PATH = os.path.join(STORE_DIR, "automl.db")

# os.makedirs(MODELS_DIR, exist_ok=True)
# os.makedirs(DATASETS_DIR, exist_ok=True)
# Fix for Render.com - ensure proper paths
BASE_DIR = os.path.abspath(os.path.dirname(__file__))

# CHANGE THIS: Use Render's /tmp directory for storage (writable)
if os.environ.get('RENDER'):
    # Render.com environment
    STORE_DIR = os.path.join(BASE_DIR, "storage")
else:
    # Local development
    STORE_DIR = os.path.join(BASE_DIR, "storage")

# Create storage directory
os.makedirs(STORE_DIR, exist_ok=True)

# Update database path
DB_PATH = os.path.join(STORE_DIR, "automl.db")

# Update model and dataset paths
MODELS_DIR = os.path.join(STORE_DIR, "models")
DATASETS_DIR = os.path.join(STORE_DIR, "datasets")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(DATASETS_DIR, exist_ok=True)

# app = Flask(__name__)
# CORS(app)

app = Flask(__name__, 
            static_folder='static',
            template_folder='templates',
            static_url_path='')
CORS(app)

# Add this configuration
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max file size
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev-secret-key-change-in-production')

# ---------- DB Utilities ----------

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS models (
    id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    problem_type TEXT NOT NULL,
    target TEXT,
    features TEXT NOT NULL,           -- JSON array
    algorithm TEXT NOT NULL,
    metric_name TEXT,
    metric_value REAL,
    api_key TEXT NOT NULL,
    model_path TEXT NOT NULL,
    dataset_path TEXT NOT NULL,
    model_size_kb INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS metadata (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    scenario TEXT,
    domain TEXT,
    target TEXT,
    problem_type TEXT,
    notes TEXT
);
"""

SCHEMA_ADVANCED_SQL = """
CREATE TABLE IF NOT EXISTS datasets_meta (
  model_id        TEXT PRIMARY KEY,
  created_at      TEXT,
  problem_type    TEXT,
  target          TEXT,
  n_rows          INTEGER,
  n_cols          INTEGER,
  pct_missing     REAL,
  pct_numeric     REAL,
  pct_categorical REAL,
  target_card     INTEGER,
  meta_json       TEXT
);

CREATE TABLE IF NOT EXISTS model_trials (
  id              INTEGER PRIMARY KEY AUTOINCREMENT,
  model_id        TEXT,
  algo_name       TEXT,
  metric_name     TEXT,
  metric_value    REAL,
  is_winner       INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS advanced_recommendations (
  id                INTEGER PRIMARY KEY AUTOINCREMENT,
  model_id          TEXT,
  recommended_algos TEXT,      -- JSON list
  rationale         TEXT,
  created_at        TEXT
);

CREATE TABLE IF NOT EXISTS advanced_runs (
  id                INTEGER PRIMARY KEY AUTOINCREMENT,
  basic_model_id    TEXT,
  advanced_model_id TEXT,
  metric_name       TEXT,
  metric_value      REAL,
  best_algo         TEXT,
  compared_to_basic REAL,      -- advanced - basic
  created_at        TEXT
);

CREATE TABLE IF NOT EXISTS advanced_feedback (
  id                INTEGER PRIMARY KEY AUTOINCREMENT,
  basic_model_id    TEXT,
  advanced_model_id TEXT,
  better_than_basic INTEGER,   -- 1/0
  comment           TEXT,
  created_at        TEXT
);
"""
def db_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

with db_conn() as conn:
    conn.executescript(SCHEMA_SQL)
    conn.executescript(SCHEMA_ADVANCED_SQL)
# ---------- DB Utilities ----------


# ADD THIS FUNCTION
def init_database():
    """Initialize database and create directories."""
    # Create storage directories
    os.makedirs(STORE_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(DATASETS_DIR, exist_ok=True)
    
    # Create/connect to database
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    
    # Execute schemas
    conn.executescript(SCHEMA_SQL)
    conn.executescript(SCHEMA_ADVANCED_SQL)
    conn.commit()
    conn.close()

# Call init on app startup
init_database()



# ---------- Helpers ----------

# def now_iso() -> str:
#     return datetime.utcnow().isoformat()
def now_iso(): 
    return datetime.now(timezone.utc).isoformat()


def random_id() -> str:
    return uuid.uuid4().hex


def make_api_key() -> str:
    # 32-byte urlsafe token
    return secrets.token_urlsafe(32)



def detect_problem_type(df: pd.DataFrame, target: str | None) -> str:
    """Smarter detection: only clustering if truly no target"""
    if not target:
        return "clustering"  # no target provided at all
    
    if target not in df.columns:
        # Don't assume clustering, just fail gracefully
        return "unknown"

    series = df[target].dropna()
    if series.empty:
        return "unknown"

    numericish = pd.api.types.is_numeric_dtype(series)
    uniq = series.astype(str).nunique(dropna=True)
    n = len(series)

    if numericish:
        if uniq < min(20, max(2, int(0.05 * n))):
            return "classification"
        return "regression"
    else:
        if uniq <= max(2, min(50, int(0.1 * n))):
            return "classification"
        return "clustering"




def build_preprocessor(
    df: pd.DataFrame,
    target: str | None,
    numeric_strategy: str = "median",
    cat_strategy: str = "most_frequent"
) -> Tuple[ColumnTransformer, List[str]]:
    """
    Build preprocessing pipeline with configurable imputation strategies.

    Parameters:
        df: pandas DataFrame
        target: target column name (optional)
        numeric_strategy: "mean", "median", "most_frequent", or "constant"
        cat_strategy: "most_frequent" or "constant"

    Returns:
        (ColumnTransformer, features)
    """
    # Select features
    features = [c for c in df.columns if c != target]
    X = df[features]

    # Split columns
    num_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    cat_cols = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]

    # --- Numeric pipeline ---
    if numeric_strategy == "constant":
        numeric_imputer = SimpleImputer(strategy="constant", fill_value=0)
    else:
        numeric_imputer = SimpleImputer(strategy=numeric_strategy)

    numeric_pipe = Pipeline([
        ("imputer", numeric_imputer),
        ("scaler", StandardScaler(with_mean=False)),  # with_mean=False keeps sparse-safe
    ])

    # --- Categorical pipeline ---
    if cat_strategy == "constant":
        cat_imputer = SimpleImputer(strategy="constant", fill_value="missing")
    else:
        cat_imputer = SimpleImputer(strategy=cat_strategy)

    cat_pipe = Pipeline([
        ("imputer", cat_imputer),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    # --- Combine ---
    pre = ColumnTransformer([
        ("num", numeric_pipe, num_cols),
        ("cat", cat_pipe, cat_cols),
    ])

    return pre, features




def choose_and_train(problem_type: str, X_train, y_train, X_val=None, y_val=None):
    """Return best (name, estimator, metric_name, metric_value, leaderboard)."""
    candidates = []
    metric_name = None

    if problem_type == "classification":
        metric_name = "f1"
        models = [
            ("LogisticRegression", LogisticRegression(max_iter=1000)),
            ("RandomForestClassifier", RandomForestClassifier(n_estimators=200, random_state=42)),
            ("SVC", SVC(probability=True)),
            ("KNN", KNeighborsClassifier(n_neighbors=5)),
            ("DecisionTreeClassifier", DecisionTreeClassifier(random_state=42)),
            ("GradientBoostingClassifier", GradientBoostingClassifier(random_state=42)),
            ("XGBClassifier", XGBClassifier(use_label_encoder=False, eval_metric="logloss")),
            ("LGBMClassifier", LGBMClassifier()),
            ("CatBoostClassifier", CatBoostClassifier(verbose=0)),
        ]
        for name, est in models:
            try:
                est.fit(X_train, y_train)
                preds = est.predict(X_val)
                score = f1_score(y_val, preds, average="weighted")
                candidates.append((name, est, float(score)))
            except Exception as e:
                print(f"[WARN] {name} failed: {e}")

    elif problem_type == "regression":
        metric_name = "r2"
        models = [
            ("LinearRegression", LinearRegression()),
            ("RandomForestRegressor", RandomForestRegressor(n_estimators=300, random_state=42)),
            ("SVR", SVR()),
            ("KNNRegressor", KNeighborsRegressor(n_neighbors=5)),
            ("DecisionTreeRegressor", DecisionTreeRegressor(random_state=42)),
            ("GradientBoostingRegressor", GradientBoostingRegressor(random_state=42)),
            ("XGBRegressor", XGBRegressor()),
            ("LGBMRegressor", LGBMRegressor()),
            ("CatBoostRegressor", CatBoostRegressor(verbose=0)),
        ]
        for name, est in models:
            try:
                est.fit(X_train, y_train)
                preds = est.predict(X_val)
                score = r2_score(y_val, preds)
                candidates.append((name, est, float(score)))
            except Exception as e:
                print(f"[WARN] {name} failed: {e}")

    else:  # clustering
        metric_name = "silhouette"
        best_est, best_score, best_name = None, -1.0, None
        leaderboard = []
        for k in (2, 3, 4, 5, 6, 7, 8, 10):
            try:
                est = KMeans(n_clusters=k, n_init=10, random_state=42)
                labels = est.fit_predict(X_train)
                score = silhouette_score(X_train, labels)
                leaderboard.append((f"KMeans(k={k})", est, float(score)))
                if score > best_score:
                    best_score, best_est, best_name = score, est, f"KMeans(k={k})"
            except Exception as e:
                print(f"[WARN] KMeans(k={k}) failed: {e}")
        return best_name, best_est, metric_name, float(best_score), leaderboard

    if not candidates:
        raise RuntimeError("No valid model candidates trained.")
    candidates.sort(key=lambda x: x[2], reverse=True)
    best_name, best_est, best_score = candidates[0]
    return best_name, best_est, metric_name, float(best_score), candidates




def model_feature_importance(pipeline: Pipeline, problem_type: str, X_val, y_val, feature_names: List[str]) -> List[Tuple[str, float]]:
    """Return a simple percentage importance. Falls back to permutation importance if needed."""
    try:
        est = pipeline.named_steps.get("est")
        pre = pipeline.named_steps.get("pre")
        # Expanded feature names after onehot
        try:
            expanded_names = pre.get_feature_names_out()
        except Exception:
            expanded_names = [f"f{i}" for i in range(X_val.shape[1])]

        if hasattr(est, "feature_importances_"):
            raw = est.feature_importances_
            vals = (raw / (raw.sum() + 1e-12)) * 100.0
            pairs = list(zip(expanded_names, vals))
        elif hasattr(est, "coef_"):
            coef = np.ravel(est.coef_)
            vals = np.abs(coef)
            vals = (vals / (vals.sum() + 1e-12)) * 100.0
            pairs = list(zip(expanded_names, vals))
        else:
            raise AttributeError
    except Exception:
        # permutation importance as fallback (can be slower)
        try:
            result = permutation_importance(pipeline, X_val, y_val, n_repeats=5, random_state=42)
            vals = (result.importances_mean / (np.sum(result.importances_mean) + 1e-12)) * 100.0
            expanded_names = pipeline.named_steps["pre"].get_feature_names_out()
            pairs = list(zip(expanded_names, vals))
        except Exception:
            # give empty list if not applicable (e.g., clustering)
            pairs = []

    # Summarize back to original feature prefixes for readability
    summary: Dict[str, float] = {}
    for name, perc in pairs:
        # name like 'cat__feature_value' or 'num__feature'
        original = name.split("__")[-1]
        base = original.split("_")[0] if original not in feature_names else original
        summary[base] = summary.get(base, 0.0) + float(perc)

    # Normalize to 100
    total = sum(summary.values())
    items = [(k, round(v * 100.0 / (total + 1e-12), 2)) for k, v in summary.items()]
    items.sort(key=lambda x: x[1], reverse=True)
    return items[:20]



def compute_dataset_stats(df: pd.DataFrame, target: str | None, problem_type: str) -> dict:
    n_rows = int(df.shape[0])
    n_cols = int(df.shape[1])
    pct_missing = float(df.isna().sum().sum() / (n_rows * n_cols) * 100) if n_rows and n_cols else 0.0

    # feature-only view
    feats = [c for c in df.columns if c != target]
    num_cols = [c for c in feats if pd.api.types.is_numeric_dtype(df[c])]
    cat_cols = [c for c in feats if c not in num_cols]
    total_feats = max(1, len(feats))
    pct_numeric = float(len(num_cols) / total_feats * 100.0)
    pct_categorical = float(len(cat_cols) / total_feats * 100.0)

    target_card = None
    if problem_type == "classification" and target in df.columns:
        target_card = int(df[target].dropna().astype(str).nunique())

    return {
        "n_rows": n_rows,
        "n_cols": n_cols,
        "pct_missing": pct_missing,
        "pct_numeric": pct_numeric,
        "pct_categorical": pct_categorical,
        "target_card": target_card,
    }

# A very simple recommender (uses history if available; else heuristics)

def recommend_algorithms(stats: dict, problem_type: str) -> tuple[list[str], str]:
    """
    Return (algorithms, rationale). Uses simple nearest-neighbor by (n_rows, n_cols, pct_missing, pct_numeric).
    Falls back to heuristics if no history.
    """
    recs = []
    rationale = ""

    # Try to use history:
    with db_conn() as conn:
        # join trials with meta of the same problem_type
        rows = conn.execute("""
            SELECT mt.algo_name, AVG(mt.metric_value) AS avg_score,
                   ABS(dm.n_rows - ?) + ABS(dm.n_cols - ?) +
                   ABS(dm.pct_missing - ?) + ABS(dm.pct_numeric - ?) AS distance
            FROM model_trials mt
            JOIN datasets_meta dm ON dm.model_id = mt.model_id
            WHERE dm.problem_type = ? AND mt.metric_value IS NOT NULL
            GROUP BY mt.algo_name
            ORDER BY distance ASC, avg_score DESC
            LIMIT 6
        """, (stats["n_rows"], stats["n_cols"], stats["pct_missing"], stats["pct_numeric"], problem_type)).fetchall()

    if rows:
        # top few by proximity and score
        recs = [r["algo_name"] for r in rows][:4]
        rationale = "Recommended from similar past datasets by proximity & historical scores."
    else:
        # Heuristics:
        if problem_type == "classification":
            if stats["n_rows"] > 50000:
                recs = ["LGBMClassifier", "XGBClassifier", "RandomForestClassifier"]
            elif (stats["target_card"] or 0) > 10:
                recs = ["XGBClassifier", "CatBoostClassifier", "RandomForestClassifier"]
            else:
                recs = ["RandomForestClassifier", "GradientBoostingClassifier", "LogisticRegression"]
            rationale = "Heuristic based on rows and target cardinality."
        elif problem_type == "regression":
            if stats["n_rows"] > 50000:
                recs = ["LGBMRegressor", "XGBRegressor", "RandomForestRegressor"]
            else:
                recs = ["RandomForestRegressor", "GradientBoostingRegressor", "LinearRegression"]
            rationale = "Heuristic based on dataset size."
        else:  # clustering
            recs = [f"KMeans(k={k})" for k in (3, 5, 8)]
            rationale = "Heuristic KMeans with several k."

    # Deduplicate, keep order
    seen = set()
    recs = [a for a in recs if not (a in seen or seen.add(a))]
    return recs, rationale






# ---------- API Endpoints ----------

@app.route('/')
def index():
    return render_template('indexnew.html')

@app.route("/advance.html")
def advance_page():
    return render_template("advance.html")

# 1. Serve dataset.html on root "/"
# @app.route('/')
# def serve_dataset():
#     frontend_path = os.path.join(os.path.dirname(__file__), '../frontend')
#     return send_from_directory(frontend_path, 'dataset.html')

# # 2. Serve all other static frontend files
# @app.route('/<path:path>')
# def serve_static_files(path):
#     frontend_path = os.path.join(os.path.dirname(__file__), '../frontend')
#     return send_from_directory(frontend_path, path)


@app.get("/api/ping")
def ping():
    return "pong", 200


@app.post("/api/metadata")
def save_metadata():
    data = request.get_json(silent=True) or {}
    row = (
        now_iso(),
        data.get("scenario"),
        data.get("domain"),
        data.get("target"),
        data.get("problem_type"),
        data.get("notes"),
    )
    with db_conn() as conn:
        conn.execute(
            "INSERT INTO metadata (created_at, scenario, domain, target, problem_type, notes) VALUES (?,?,?,?,?,?)",
            row,
        )
    return jsonify({"status": "ok"})


@app.post("/api/detect")
def detect():
    payload = request.get_json(silent=True) or {}
    sample = payload.get("sample") or []
    target = payload.get("target")
    try:
        df = pd.DataFrame(sample)
        p = detect_problem_type(df, target)
        return jsonify({"problem_type": p})
    except Exception as e:
        return jsonify({"problem_type": None, "error": str(e)}), 400


@app.post("/api/train")
def train():
    # multipart: file, target, problem_type
    if "file" not in request.files:
        return jsonify({"message": "CSV file is required as 'file'"}), 400

    f = request.files["file"]
    target = request.form.get("target")
    problem_type = request.form.get("problem_type")

    # Optional missing-value strategy
    strategy = request.form.get("missing_strategy", "auto")  
    numeric_strategy = request.form.get("numeric_strategy")
    cat_strategy = request.form.get("cat_strategy")

    # Save dataset
    model_id = random_id()
    dataset_path = os.path.join(DATASETS_DIR, f"{model_id}.csv")
    f.save(dataset_path)

    try:
        df = pd.read_csv(dataset_path)
    except Exception:
        df = pd.read_csv(dataset_path, encoding="latin-1")

    # Handle missing value strategy (ALWAYS executed, not only on exception)
    if strategy == "drop":
        df = df.dropna()
    elif strategy == "mean_mode":
        numeric_strategy, cat_strategy = "mean", "most_frequent"
    elif strategy == "median_mode":
        numeric_strategy, cat_strategy = "median", "most_frequent"
    elif strategy == "constant":
        numeric_strategy, cat_strategy = "constant", "constant"
    else:  # auto default
        if not numeric_strategy:
            numeric_strategy = "median"
        if not cat_strategy:
            cat_strategy = "most_frequent"

    # Detect problem type if not provided
    if not problem_type:
        problem_type = detect_problem_type(df, target)

   # ---------------- Save dataset stats for Advanced Feature ----------------
    n_rows, n_cols = df.shape
    pct_missing = float(df.isnull().sum().sum()) / (n_rows * n_cols) if n_rows * n_cols > 0 else 0
    pct_numeric = len([c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]) / n_cols if n_cols > 0 else 0
    pct_categorical = 1 - pct_numeric
    target_card = int(df[target].nunique()) if target and target in df.columns else None

    meta_json = {
        "columns": df.dtypes.astype(str).to_dict(),
        "sample": df.head(5).to_dict(orient="records")
    }

    with db_conn() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO datasets_meta (model_id, created_at, problem_type, target, n_rows, n_cols, pct_missing, pct_numeric, pct_categorical, target_card, meta_json) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (
                model_id,
                now_iso(),
                problem_type,
                target,
                n_rows,
                n_cols,
                pct_missing,
                pct_numeric,
                pct_categorical,
                target_card,
                json.dumps(meta_json),
            ),
        )
    # --

    # Build preprocessor with chosen strategies
    pre, features = build_preprocessor(df, target, numeric_strategy, cat_strategy)

    # ---------------- Training as before ----------------
    if problem_type in ("classification", "regression"):
        if target is None or target not in df.columns:
            return jsonify({"message": "Target column missing for supervised learning"}), 400
        y = df[target]
        X = df.drop(columns=[target])
        if y.dropna().empty:
            return jsonify({"message": "Target column has no valid values"}), 400

        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42,
            stratify=y if problem_type == "classification" else None
        )

        pipe = Pipeline([
            ("pre", pre),
            ("est", RandomForestClassifier() if problem_type == "classification" else RandomForestRegressor()),
        ])

        Xtr = pipe.named_steps["pre"].fit_transform(X_train)
        Xva = pipe.named_steps["pre"].transform(X_val)

        best_name, best_est, metric_name, metric_value, leaderboard = choose_and_train(
            problem_type, Xtr, y_train, Xva, y_val
        )

        # ---------------- Save leaderboard rows to model_trials ----------------
        with db_conn() as conn:
            for name, _est, score in leaderboard:
                conn.execute("""
                    INSERT INTO model_trials (model_id, algo_name, metric_name, metric_value, is_winner)
                    VALUES (?,?,?,?,?)
                """, (model_id, name, metric_name, float(score), 1 if name == best_name else 0))
        # -----------------------------------------------------------------------


        final_pipe = Pipeline([
            ("pre", pre),
            ("est", best_est),
        ])
        final_pipe.fit(X_train, y_train)

        try:
            importance = model_feature_importance(final_pipe, problem_type, X_val, y_val, features)
        except Exception:
            importance = []

    else:  # clustering
        X = df
        pipe = Pipeline([
            ("pre", pre),
        ])
        Xtr = pipe.named_steps["pre"].fit_transform(X)
        best_name, best_est, metric_name, metric_value, leaderboard = choose_and_train("clustering", Xtr, None)
        final_pipe = Pipeline([
            ("pre", pre),
            ("est", best_est),
        ])
        final_pipe.fit(X)
        importance = []

    # Save model
    model_path = os.path.join(MODELS_DIR, f"{model_id}.joblib")
    joblib.dump({
        "pipeline": final_pipe,
        "problem_type": problem_type,
        "target": target,
        "features": features,
    }, model_path)

    size_kb = int(os.path.getsize(model_path) / 1024)
    api_key = make_api_key()

    with db_conn() as conn:
        conn.execute(
            "INSERT INTO models (id, created_at, problem_type, target, features, algorithm, metric_name, metric_value, api_key, model_path, dataset_path, model_size_kb) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                model_id,
                now_iso(),
                problem_type,
                target,
                json.dumps(features),
                best_name,
                metric_name,
                float(metric_value) if metric_value is not None else None,
                api_key,
                model_path,
                dataset_path,
                size_kb,
            ),
        )

    sample_row = df.drop(columns=[target]).iloc[0].to_dict() if (problem_type != "clustering" and target in df.columns and len(df) > 0) else (df.iloc[0].to_dict() if len(df) > 0 else {})

    # Collect categorical values for dropdowns
    categories = {}
    for col in features:
        if df[col].dtype == "object" or str(df[col].dtype).startswith("category"):
            categories[col] = df[col].dropna().unique().tolist()

    # Convert leaderboard to JSON-safe format
    leaderboard_json = [(name, float(score)) for name, _est, score in leaderboard]        

    return jsonify({
        "id": model_id,
        "model_id": model_id,
        "best_model": best_name,
        "metric": f"{metric_name}: {round(metric_value, 4) if metric_value is not None else 'N/A'}",
        "api_key": api_key,
        "download_url": f"/api/model/{model_id}/download",
        "features": features,
        "target": target,
        "algorithm": best_name,
        "model_size_kb": size_kb,
        "importance": importance,
        "sample_row": sample_row,
        "csv_download_url": f"/api/dataset/{model_id}/download",
        "problem_type": problem_type,
        "categories": categories,
        "leaderboard": leaderboard_json,
    })



@app.post("/api/feedback/domain")
def domain_feedback():
    data = request.get_json(silent=True) or {}
    scenario  = data.get("scenario")
    domain    = data.get("domain") or data.get("suggested")
    feedback  = data.get("feedback")  # 'up' or 'down'
    print(f"[domain_feedback] scenario={scenario!r} domain={domain!r} feedback={feedback!r}")
    return jsonify({"ok": True})



@app.get("/api/model/<model_id>/download")
def download_model(model_id: str):
    with db_conn() as conn:
        cur = conn.execute("SELECT model_path FROM models WHERE id = ?", (model_id,))
        row = cur.fetchone()
        if not row:
            abort(404)
        path = row["model_path"]
    return send_file(path, as_attachment=True, download_name=f"model_{model_id}.joblib")


@app.get("/api/dataset/<model_id>/download")
def download_dataset(model_id: str):
    with db_conn() as conn:
        cur = conn.execute("SELECT dataset_path FROM models WHERE id = ?", (model_id,))
        row = cur.fetchone()
        if not row:
            abort(404)
        path = row["dataset_path"]
    return send_file(path, as_attachment=True, download_name=f"dataset_{model_id}.csv")


def _auth_model(model_id: str, token: str) -> Dict[str, Any]:
    if not token or not token.startswith("Bearer "):
        abort(401)
    key = token.split(" ", 1)[1].strip()
    with db_conn() as conn:
        cur = conn.execute("SELECT * FROM models WHERE id = ?", (model_id,))
        row = cur.fetchone()
        if not row:
            abort(404)
        if row["api_key"] != key:
            abort(403)
        return dict(row)
    
@app.route('/api/nlp/domain-detect', methods=['POST'])
def detect_domain():
    data = request.get_json()
    scenario = data.get("scenario", "")

    # --- Step 1: Grammar correction (dummy for now) ---
    corrected = scenario.strip().capitalize()

    # --- Step 2: Domain detection (dummy logic) ---
    # TODO: Replace with ML/NLP model
    if "student" in scenario.lower():
        suggested = "Education"
        alternatives = ["Academia", "EdTech", "Schools", "Training", "Research"]
    elif "customer" in scenario.lower():
        suggested = "Retail"
        alternatives = ["SaaS", "E-commerce", "Telecom", "Banking", "Insurance"]
    else:
        suggested = "General"
        alternatives = ["Finance", "Healthcare", "Retail", "SaaS", "Manufacturing"]

    return jsonify({
        "corrected_scenario": corrected,
        "suggested_domain": suggested,
        "alternatives": alternatives
    })



@app.post("/api/predict")
def predict():
    data = request.get_json(silent=True) or {}
    model_id = data.get("model_id")
    records = data.get("records")
    if not model_id or not isinstance(records, list) or len(records) == 0:
        return jsonify({"message": "Provide 'model_id' and non-empty 'records' array"}), 400

    auth = request.headers.get("Authorization", "")
    row = _auth_model(model_id, auth)

    # Load
    model_obj = joblib.load(row["model_path"])  # dict
    pipeline: Pipeline = model_obj["pipeline"]
    problem_type: str = model_obj.get("problem_type")

    X = pd.DataFrame.from_records(records)
    preds = pipeline.predict(X)

    # For regression return floats, for classification possibly proba
    result: Dict[str, Any] = {
        "predictions": preds.tolist(),
    }

    if problem_type == "classification" and hasattr(pipeline.named_steps.get("est"), "predict_proba"):
        try:
            proba = pipeline.predict_proba(X)
            result["probabilities"] = proba.tolist()
            if hasattr(pipeline.named_steps.get("est"), "classes_"):
                result["classes"] = pipeline.named_steps.get("est").classes_.astype(str).tolist()
        except Exception:
            pass

    return jsonify(result)



# ---------- Advanced API ----------

@app.post("/api/advance/recommend")
def advance_recommend():
    """Return algorithm recommendations for a given basic model."""
    data = request.json
    model_id = data.get("model_id")
    if not model_id:
        return jsonify({"message": "model_id required"}), 400

    with db_conn() as conn:
        recs = conn.execute(
            "SELECT recommended_algos, rationale FROM advanced_recommendations "
            "WHERE model_id=? ORDER BY id DESC LIMIT 1",
            (model_id,)
        ).fetchone()

    if recs:
        recs_json = json.loads(recs["recommended_algos"])
        rationale = recs["rationale"]
    else:
        # fallback defaults
        recs_json = [
            {"name": "CatBoostClassifier", "confidence": 0.65},
            {"name": "LGBMClassifier", "confidence": 0.55},
            {"name": "RandomForestClassifier", "confidence": 0.5},
        ]
        rationale = "Default heuristics applied."

    return jsonify({"recommendations": recs_json, "rationale": rationale})


@app.post("/api/advance/train")
def advance_train():
    """
    Advanced training endpoint:
    Request JSON:
      {
        "basic_model_id": "<id>" OR "model_id": "<id>",
        "algorithms": ["LGBMRegressor", "CatBoostRegressor"],   // optional - if omitted use recommender
        "cv": true/false,       // perform cross-validation scoring
        "tune": true/false,     // perform light hyperparameter search (RandomizedSearchCV)
        "budget_minutes": 3     // time budget (best-effort)
      }
    Response:
      {
        "advanced_model_id": "...",
        "best": {"name":"...","score":...,"model_id":"..."},
        "candidates": [{"name":"...", "score":..., "note":"val"}, ...],
        "metric": ...,
        "api_key":"...",
        "features":[...],
        "target":"...",
        "categories": {...},
        "sample_row": {...},
        "rationale": "..."
      }
    """
    data = request.get_json(silent=True) or {}
    basic_model_id = data.get("basic_model_id") or data.get("model_id")
    if not basic_model_id:
        return jsonify({"message": "basic_model_id (or model_id) required"}), 400

    algorithms_req = data.get("algorithms")  # optional
    use_cv = bool(data.get("cv", False))
    do_tune = bool(data.get("tune", False))
    budget_minutes = float(data.get("budget_minutes", 3))

    # Load basic model metadata
    with db_conn() as conn:
        row = conn.execute("SELECT * FROM models WHERE id = ?", (basic_model_id,)).fetchone()
    if not row:
        return jsonify({"message": "Unknown basic_model_id"}), 404

    dataset_path = row["dataset_path"]
    target = row["target"]
    problem_type = row["problem_type"]

    # Load dataset
    try:
        df = pd.read_csv(dataset_path)
    except Exception:
        df = pd.read_csv(dataset_path, encoding="latin-1")

    # Build preprocessor (reuse default strategies)
    pre, features = build_preprocessor(df, target, numeric_strategy="median", cat_strategy="most_frequent")

    # Compute stats + use recommender if needed
    stats = compute_dataset_stats(df, target, problem_type)
    algos, rationale = recommend_algorithms(stats, problem_type)
    # If caller provided algos, prefer that subset (but validate names)
    if algorithms_req and isinstance(algorithms_req, (list, tuple)) and len(algorithms_req) > 0:
        # intersection while preserving order of algorithms_req
        algos = [a for a in algorithms_req if make_estimator(a) is not None]

    # Simple time-bounded training loop (best-effort)
    start_ts = time.time()
    time_limit = budget_minutes * 60.0

    leaderboard = []
    best_score = -float("inf")
    best_name = None
    best_estimator = None
    metric_name = "f1" if problem_type == "classification" else ("r2" if problem_type == "regression" else "silhouette")

    # Prepare X/y depending on problem
    if problem_type in ("classification", "regression"):
        if target is None or target not in df.columns:
            return jsonify({"message": "Target column missing for supervised learning"}), 400
        y = df[target]
        X = df.drop(columns=[target])

        # Basic train/val split (consistent with basic mode)
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42,
            stratify=y if problem_type == "classification" else None
        )

        # Fit preprocessor and transform (once)
        Xtr = pre.fit_transform(X_train)
        Xva = pre.transform(X_val)

        for name in algos:
            # Check time budget
            elapsed = time.time() - start_ts
            if elapsed > time_limit:
                break

            est = make_estimator(name)
            if est is None:
                continue

            # If tuning is requested, run a lightweight randomized search (best-effort)
            try:
                if do_tune:
                    # simple parameter grids (keep small to be fast)
                    param_dist = {
                        "RandomForestRegressor": {"n_estimators": [50,100,200], "max_depth": [None, 6, 12]},
                        "RandomForestClassifier": {"n_estimators": [50,100,200], "max_depth": [None, 6, 12]},
                        "GradientBoostingRegressor": {"n_estimators": [50,100], "learning_rate": [0.05,0.1]},
                        "GradientBoostingClassifier": {"n_estimators": [50,100], "learning_rate": [0.05,0.1]},
                        "SVR": {"C": [0.1,1,10], "gamma": ["scale","auto"]},
                        "SVR": {"C": [0.1,1,10], "kernel": ["rbf","linear"]},
                        "XGBRegressor": {"n_estimators": [50,100], "learning_rate":[0.05,0.1]},
                        "XGBClassifier": {"n_estimators": [50,100], "learning_rate":[0.05,0.1]},
                        "LGBMRegressor": {"n_estimators":[50,100], "learning_rate":[0.05,0.1]},
                        "LGBMClassifier": {"n_estimators":[50,100], "learning_rate":[0.05,0.1]},
                        "CatBoostRegressor": {"iterations":[50,100], "learning_rate":[0.05,0.1]},
                        "CatBoostClassifier": {"iterations":[50,100], "learning_rate":[0.05,0.1]}
                    }.get(name, None)

                    if param_dist:
                        # RandomizedSearchCV on transformed space
                        n_iter = 6
                        # choose cv folds
                        cv_splits = 3 if use_cv else 2
                        rs = RandomizedSearchCV(est, param_distributions=param_dist, n_iter=n_iter,
                                                scoring=("f1_weighted" if problem_type=="classification" else "r2"),
                                                cv=cv_splits, random_state=42, n_jobs=1)
                        rs.fit(Xtr, y_train)
                        est_to_eval = rs.best_estimator_
                    else:
                        est_to_eval = est
                else:
                    est_to_eval = est

                # Evaluate via cross_val_score if requested, otherwise on Xva
                if use_cv:
                    cv = 3
                    if problem_type == "classification":
                        scs = cross_val_score(est_to_eval, Xtr, y_train, cv=cv, scoring="f1_weighted")
                        score = float(np.mean(scs))
                    else:
                        scs = cross_val_score(est_to_eval, Xtr, y_train, cv=cv, scoring="r2")
                        score = float(np.mean(scs))
                else:
                    est_to_eval.fit(Xtr, y_train)
                    preds = est_to_eval.predict(Xva)
                    if problem_type == "classification":
                        score = float(f1_score(y_val, preds, average="weighted"))
                    else:
                        score = float(r2_score(y_val, preds))

                leaderboard.append({"name": name, "score": score, "note": "cv" if use_cv else "val"})
                # persist trial row later (after model chosen) or persist here
                if score > best_score:
                    best_score = score
                    best_name = name
                    best_estimator = est_to_eval

                # check time budget again
                if time.time() - start_ts > time_limit:
                    break

            except Exception as e:
                print("[ADV][WARN]", name, "failed during train/eval:", e)
                # still continue to next algorithm

    else:
        # clustering case
        X = df.copy()
        Xtr = pre.fit_transform(X)
        for name in algos:
            if time.time() - start_ts > time_limit:
                break
            if name.startswith("KMeans("):
                try:
                    # parse k from name like KMeans(k=3)
                    k = int(name.split("KMeans(k=")[1].split(")")[0])
                except Exception:
                    # fallback k values
                    k = 3
                try:
                    est = KMeans(n_clusters=k, n_init=10, random_state=42)
                    labels = est.fit_predict(Xtr)
                    score = float(silhouette_score(Xtr, labels))
                    leaderboard.append({"name": f"KMeans(k={k})", "score": score, "note":"silhouette"})
                    if score > best_score:
                        best_score = score
                        best_name = f"KMeans(k={k})"
                        best_estimator = est
                except Exception as e:
                    print("[ADV][WARN] KMeans(k={}) failed: {}".format(k, e))

    if not leaderboard:
        return jsonify({"message": "No advanced candidates trained"}), 500

    # Build final pipeline (fit on full data)
    try:
        if problem_type in ("classification", "regression"):
            final_pipe = Pipeline([("pre", pre), ("est", best_estimator)])
            final_pipe.fit(X, y)
        else:
            final_pipe = Pipeline([("pre", pre), ("est", best_estimator)])
            final_pipe.fit(df)
    except Exception as e:
        print("[ADV][WARN] final pipeline fit failed:", e)
        return jsonify({"message": "Final pipeline fit failed", "error": str(e)}), 500

    # Persist advanced model as new model row
    adv_model_id = random_id()
    model_path = os.path.join(MODELS_DIR, f"{adv_model_id}.joblib")
    joblib.dump({
        "pipeline": final_pipe,
        "problem_type": problem_type,
        "target": target,
        "features": features,
    }, model_path)

    size_kb = int(os.path.getsize(model_path) / 1024)
    api_key = make_api_key()

    # Insert DB entries (models, advanced_recommendations, advanced_runs, model_trials)
    with db_conn() as conn:
        # insert model
        conn.execute("""
            INSERT INTO models (id, created_at, problem_type, target, features, algorithm, metric_name, metric_value, api_key, model_path, dataset_path, model_size_kb)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
        """, (
            adv_model_id, now_iso(), problem_type, target, json.dumps(features),
            best_name, metric_name, float(best_score), api_key, model_path, dataset_path, size_kb
        ))

        # insert recommendation record
        conn.execute("""
            INSERT INTO advanced_recommendations (model_id, recommended_algos, rationale, created_at)
            VALUES (?,?,?,?)
        """, (adv_model_id, json.dumps(algos), rationale, now_iso()))

        # insert advanced run record comparing to basic
        basic_metric = row["metric_value"] if row["metric_value"] is not None else None
        delta = None
        try:
            delta = float(best_score) - float(basic_metric) if basic_metric is not None else None
        except Exception:
            delta = None

        conn.execute("""
            INSERT INTO advanced_runs (basic_model_id, advanced_model_id, metric_name, metric_value, best_algo, compared_to_basic, created_at)
            VALUES (?,?,?,?,?,?,?)
        """, (basic_model_id, adv_model_id,
              metric_name, float(best_score), best_name, delta, now_iso()))

        # store per-algo trials
        for c in leaderboard:
            try:
                conn.execute("""
                    INSERT INTO model_trials (model_id, algo_name, metric_name, metric_value, is_winner)
                    VALUES (?,?,?,?,?)
                """, (adv_model_id, c["name"], metric_name, float(c["score"]), 1 if c["name"] == best_name else 0))
            except Exception:
                pass

    # categories and sample_row for frontend to build form
    categories = {}
    for col in features:
        if col in df.columns and (df[col].dtype == "object" or str(df[col].dtype).startswith("category")):
            categories[col] = df[col].dropna().astype(str).unique().tolist()

    sample_row = (df.drop(columns=[target]).iloc[0].to_dict()
                  if (problem_type != "clustering" and target in df.columns and len(df) > 0)
                  else (df.iloc[0].to_dict() if len(df) > 0 else {}))

    # best candidate object
    best = {"name": best_name, "score": float(best_score), "model_id": adv_model_id}

    return jsonify({
        "advanced_model_id": adv_model_id,
        "best": best,
        "candidates": leaderboard,
        "metric": float(best_score),
        "download_url": f"/api/model/{adv_model_id}/download",
        "api_key": api_key,
        "features": features,
        "target": target,
        "categories": categories,
        "sample_row": sample_row,
        "rationale": rationale
    })




@app.post("/api/advance/feedback")
def advance_feedback():
    """Collect user feedback comparing advanced vs basic models."""
    data = request.get_json(silent=True) or {}
    basic_id, adv_id = data.get("basic_model_id"), data.get("advanced_model_id")
    better = 1 if data.get("better_than_basic") else 0
    comment = data.get("comment", "")

    if not (basic_id and adv_id):
        return jsonify({"message": "basic_model_id and advanced_model_id required"}), 400

    with db_conn() as conn:
        conn.execute("""INSERT INTO advanced_feedback (basic_model_id, advanced_model_id,
                        better_than_basic, comment, created_at)
                        VALUES (?,?,?,?,?)""",
                     (basic_id, adv_id, better, comment, now_iso()))
    return jsonify({"ok": True})


@app.post("/api/advance/adopt")
def advance_adopt():
    """Mark an advanced model as adopted for a base model."""
    data = request.get_json(silent=True) or {}
    base_model_id, new_model_id = data.get("base_model_id"), data.get("new_model_id")
    if not (base_model_id and new_model_id):
        return jsonify({"message": "base_model_id and new_model_id required"}), 400

    with db_conn() as conn:
        conn.execute("""UPDATE advanced_runs SET compared_to_basic = 1
                        WHERE advanced_model_id = ?""", (new_model_id,))
    return jsonify({"message": f"Adopted {new_model_id} for base {base_model_id}"})


@app.get("/api/model/<model_id>/context")
def model_context(model_id):
    """Return dataset stats and context for a model."""
    with db_conn() as conn:
        row = conn.execute("SELECT * FROM models WHERE id=?", (model_id,)).fetchone()
    if not row:
        return jsonify({"message": "model not found"}), 404

    df = pd.read_csv(row["dataset_path"])
    stats = compute_dataset_stats(df, row["target"], row["problem_type"])

    return jsonify({
        "stats": stats,
        "problem_type": row["problem_type"],
        "target": row["target"],
        "features": json.loads(row["features"]),
    })



# helper for estimators
def make_estimator(name):
    mapping = {
        "LogisticRegression": LogisticRegression(max_iter=1000),
        "RandomForestClassifier": RandomForestClassifier(n_estimators=200, random_state=42),
        "SVC": SVC(probability=True),
        "KNNClassifier": KNeighborsClassifier(n_neighbors=5),
        "DecisionTreeClassifier": DecisionTreeClassifier(random_state=42),
        "GradientBoostingClassifier": GradientBoostingClassifier(random_state=42),
        "XGBClassifier": XGBClassifier(use_label_encoder=False, eval_metric="logloss"),
        "LGBMClassifier": LGBMClassifier(),
        "CatBoostClassifier": CatBoostClassifier(verbose=0),

        "LinearRegression": LinearRegression(),
        "RandomForestRegressor": RandomForestRegressor(n_estimators=300, random_state=42),
        "SVR": SVR(),
        "KNNRegressor": KNeighborsRegressor(n_neighbors=5),
        "DecisionTreeRegressor": DecisionTreeRegressor(random_state=42),
        "GradientBoostingRegressor": GradientBoostingRegressor(random_state=42),
        "XGBRegressor": XGBRegressor(),
        "LGBMRegressor": LGBMRegressor(),
        "CatBoostRegressor": CatBoostRegressor(verbose=0),
    }
    return mapping.get(name)

# if __name__ == "__main__":
#     # Optional: allow PORT env var
#     port = int(os.environ.get("PORT", 5500))
#     app.run(host="127.0.0.1", port=port, debug=True)


if __name__ == "__main__":
    # Get port from environment (Render sets PORT)
    port = int(os.environ.get("PORT", 5500))
    
    # Use 0.0.0.0 for production
    app.run(host="0.0.0.0", port=port, debug=False)

