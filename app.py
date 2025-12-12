# src/app.py
"""
Real Estate Investment Advisor (MLflow)

Gauge-based metric dashboard + aligned right-column "Model Input (final)" preview.
Model URIs and load status are shown in the sidebar only.
"""

import os
import json
import socket
import urllib.parse
from datetime import datetime
from typing import Optional, Any, List, Dict, Tuple

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import mlflow.pyfunc
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve,
    mean_squared_error, mean_absolute_error, r2_score
)

# ----- Config / paths -----
REG_JSON = os.path.join("config", "best_regression_registered.json")
CLF_JSON = os.path.join("config", "best_classification_registered.json")
DATA_CSV = os.path.join("data", "processed", "india_housing_cleaned.csv")
PREDICTIONS_LOG = os.path.join("reports", "metrics", "predictions.csv")
DEFAULT_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")

MODEL_FEATURES = [
    "State","City","Locality","Property_Type","BHK","Size_in_SqFt","Price_per_SqFt",
    "Nearby_Schools","Nearby_Hospitals","Public_Transport_Accessibility",
    "Parking_Space","Amenities_Count","Age_of_Property","Furnished_Status","Owner_Type",
    "Availability_Status","Floor_No","Total_Floors","Facing"
]

NUMERIC_FEATURES = {
    "BHK","Size_in_SqFt","Price_per_SqFt","Nearby_Schools","Nearby_Hospitals",
    "Floor_No","Total_Floors","Age_of_Property","Amenities_Count"
}

TRANSPORT_MAP = {"Low": 1, "Medium": 2, "High": 3}
PARKING_MAP = {"No": 0, "Yes": 1}

st.set_page_config(page_title="Real Estate Investment Advisor (MLflow)", layout="wide")
st.title("🏠 Real Estate Investment Advisor (MLflow)")

# ----- Helpers -----
def read_json(path: str) -> Optional[dict]:
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None
    return None

def normalize_uri(uri: str):
    if not isinstance(uri, str):
        return uri
    return uri.strip().replace("\\", "/")

def is_host_reachable(uri: str) -> bool:
    try:
        parsed = urllib.parse.urlparse(uri)
        host = parsed.hostname
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(1.0)
        sock.connect((host, port))
        sock.close()
        return True
    except Exception:
        return False

@st.cache_data(ttl=600)
def load_dataset(path: str) -> Optional[pd.DataFrame]:
    if not os.path.exists(path):
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None

@st.cache_resource
def load_mlflow_model_cached(uri: str):
    uri = normalize_uri(uri)
    if not uri:
        raise ValueError("Empty model URI")
    return mlflow.pyfunc.load_model(uri)

def sanitize_name(s: str) -> str:
    return s.replace(" ", "_").replace("(", "").replace(")", "").replace("/", "_").lower()

def prepare_input(df_row: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df_row, pd.Series):
        df_row = df_row.to_frame().T
    out = df_row.reindex(columns=MODEL_FEATURES, fill_value=np.nan).copy()
    for col in MODEL_FEATURES:
        if pd.isna(out.iloc[0].get(col)):
            out[col] = 0 if col in NUMERIC_FEATURES else "Unknown"
    for col in NUMERIC_FEATURES:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0)
    if "Public_Transport_Accessibility" in out.columns:
        v = out.iloc[0]["Public_Transport_Accessibility"]
        try:
            if isinstance(v, str):
                out.loc[:, "Public_Transport_Accessibility"] = TRANSPORT_MAP.get(v, np.nan)
            else:
                out.loc[:, "Public_Transport_Accessibility"] = pd.to_numeric(v, errors="coerce")
        except Exception:
            out.loc[:, "Public_Transport_Accessibility"] = pd.to_numeric(v, errors="coerce")
    if "Parking_Space" in out.columns:
        v = out.iloc[0]["Parking_Space"]
        if isinstance(v, str):
            out.loc[:, "Parking_Space"] = PARKING_MAP.get(v, np.nan)
        else:
            out.loc[:, "Parking_Space"] = pd.to_numeric(v, errors="coerce")
    for col in NUMERIC_FEATURES:
        out[col] = out[col].fillna(0)
    return out[MODEL_FEATURES]

def unwrap_reg(pred_raw):
    try:
        if isinstance(pred_raw, (list, tuple, np.ndarray, pd.Series)):
            return float(np.asarray(pred_raw).ravel()[0])
        if isinstance(pred_raw, pd.DataFrame):
            numeric_cols = pred_raw.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                return float(pred_raw[numeric_cols[0]].iloc[0])
            return float(pred_raw.iloc[0,0])
        return float(pred_raw)
    except Exception:
        return float("nan")

def unwrap_clf(model_obj, X):
    try:
        if hasattr(model_obj, "predict_proba"):
            proba = model_obj.predict_proba(X)
            arr = np.asarray(proba)
            if arr.ndim == 2 and arr.shape[1] >= 2:
                return float(arr[0,1]), arr.ravel()
            return float(arr.ravel()[0]), arr.ravel()
        pred = model_obj.predict(X)
        arr = np.asarray(pred)
        return float(arr.ravel()[0]), arr.ravel()
    except Exception:
        return None, None

def _safe(v, default=0.0):
    try:
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return float(default)
        return float(v)
    except Exception:
        return float(default)

# ----- Sidebar: model selection & info -----
reg_cfg = read_json(REG_JSON)
clf_cfg = read_json(CLF_JSON)

def tracking_uri_from_cfgs(r_cfg, c_cfg):
    env_uri = os.environ.get("MLFLOW_TRACKING_URI")
    if env_uri:
        return env_uri
    for cfg in (r_cfg, c_cfg):
        if isinstance(cfg, dict):
            for k in ("tracking_uri", "mlflow_tracking_uri", "tracking_server"):
                if cfg.get(k):
                    return cfg.get(k)
    return DEFAULT_TRACKING_URI

tracking_uri = tracking_uri_from_cfgs(reg_cfg, clf_cfg)
os.environ["MLFLOW_TRACKING_URI"] = tracking_uri

st.sidebar.header("MLflow & Model selection")
st.sidebar.markdown(f"**Tracking URI:** `{tracking_uri}`")
try:
    if tracking_uri.startswith("http") and not is_host_reachable(tracking_uri):
        st.sidebar.warning("MLflow tracking server not reachable (timeout).")
except Exception:
    pass

# keep the model info in the sidebar (main panel won't show URIs)
st.sidebar.markdown("---")
st.sidebar.markdown("Model info (selected)")

def build_options(cfg: Optional[dict]) -> List[Dict[str,str]]:
    opts = []
    if not isinstance(cfg, dict):
        return opts
    primary = cfg.get("registered_name") or cfg.get("registered_name") or cfg.get("registered_name")
    if primary:
        opts.append({"label": f"{primary} (primary)", "name": primary, "uri": f"models:/{primary}/Production"})
    for alt in cfg.get("alternatives", []) or []:
        an = str(alt)
        opts.append({"label": an, "name": an, "uri": f"models:/{an}/Production"})
    return opts

reg_options = build_options(reg_cfg)
clf_options = build_options(clf_cfg)

st.sidebar.subheader("Regression model")
reg_labels = [o["label"] for o in reg_options] if reg_options else []
if reg_labels:
    reg_sel_idx = st.sidebar.selectbox("Choose regression", options=list(range(len(reg_labels))), format_func=lambda i: reg_labels[i], index=0)
    reg_uri = reg_options[reg_sel_idx]["uri"]
else:
    reg_uri = st.sidebar.text_input("Custom regression URI", value="")

st.sidebar.subheader("Classification model")
clf_labels = [o["label"] for o in clf_options] if clf_options else []
if clf_labels:
    clf_sel_idx = st.sidebar.selectbox("Choose classification", options=list(range(len(clf_labels))), format_func=lambda i: clf_labels[i], index=0)
    clf_uri = clf_options[clf_sel_idx]["uri"]
else:
    clf_uri = st.sidebar.text_input("Custom classification URI", value="")

st.sidebar.markdown("---")
st.sidebar.code(f"Regression: {reg_uri}")
st.sidebar.code(f"Classification: {clf_uri}")

# ----- Safe session-state initialization -----
if "models_loaded" not in st.session_state:
    st.session_state["models_loaded"] = False
if "reg_model" not in st.session_state:
    st.session_state["reg_model"] = None
if "clf_model" not in st.session_state:
    st.session_state["clf_model"] = None
if "model_load_errors" not in st.session_state:
    st.session_state["model_load_errors"] = []

# Improved clear cache handler with diagnostics & safe fallbacks
if st.sidebar.button("Clear model cache"):
    errors = []
    try:
        st.cache_resource.clear()
    except Exception as e:
        errors.append(f"cache_resource.clear() failed: {e}")
    try:
        st.cache_data.clear()
    except Exception as e:
        errors.append(f"cache_data.clear() failed: {e}")

    # safe reset of session keys
    for k in ("models_loaded", "reg_model", "clf_model", "model_load_errors"):
        try:
            if k in st.session_state:
                del st.session_state[k]
        except Exception:
            pass

    if errors:
        st.sidebar.warning("Couldn't fully clear cache in this environment.")
        for e in errors:
            st.sidebar.error(e)
        try:
            st.experimental_rerun()
        except Exception:
            st.sidebar.info("Please refresh or restart the app to ensure cache is cleared.")
    else:
        st.sidebar.success("Cache cleared and state reset.")
        try:
            st.experimental_rerun()
        except Exception:
            st.sidebar.info("Cache cleared — please refresh the page.")

# Auto-load models immediately (cached)
def try_load_model_uri(uri: str, kind: str) -> Optional[Any]:
    if not uri:
        return None
    try:
        m = load_mlflow_model_cached(uri)
        st.sidebar.success(f"{kind} loaded")
        return m
    except Exception as e:
        err = f"{kind}: {str(e)}"
        # ensure list exists
        if "model_load_errors" not in st.session_state:
            st.session_state["model_load_errors"] = []
        st.session_state["model_load_errors"].append(err)
        st.sidebar.error(f"{kind} load failed")
        return None

# Use .get(...) with defaults to avoid KeyError if keys were removed
if not st.session_state.get("models_loaded", False):
    st.session_state["model_load_errors"] = []
    st.sidebar.info("Loading selected models (cached)...")
    try:
        st.session_state["reg_model"] = try_load_model_uri(reg_uri, "Regression") if reg_uri else None
        st.session_state["clf_model"] = try_load_model_uri(clf_uri, "Classification") if clf_uri else None
        st.session_state["models_loaded"] = True if not st.session_state.get("model_load_errors") else False
    except Exception as e:
        if "model_load_errors" not in st.session_state:
            st.session_state["model_load_errors"] = []
        st.session_state["model_load_errors"].append(f"Autoload failed: {e}")
        st.session_state["models_loaded"] = False

if st.session_state.get("models_loaded", False):
    st.sidebar.success("Models loaded (cached)")
else:
    if st.session_state.get("model_load_errors"):
        for e in st.session_state.get("model_load_errors", []):
            st.sidebar.error(e)
    else:
        st.sidebar.info("Models not loaded yet or no URI provided")

# ----- Load dataset -----
df_all = None
try:
    df_all = load_dataset(DATA_CSV)
except Exception as e:
    st.sidebar.error(f"Failed to load dataset: {e}")

if df_all is None:
    st.sidebar.info("Dataset not found. Metrics disabled.")
else:
    st.sidebar.success(f"Dataset loaded ({len(df_all)} rows)")

# ----- Metrics helpers for classification & regression -----
def df_signature(df: pd.DataFrame):
    if df is None:
        return (0, 0)
    rows = len(df)
    s = 0
    if "Good_Investment" in df.columns:
        try:
            s = int(df["Good_Investment"].sum())
        except Exception:
            s = 0
    return (rows, s)

@st.cache_data(ttl=600)
def compute_classification_metrics(uri: str, rows_sig: int, sum_sig: int) -> Dict[str, Any]:
    result = {"ok": False, "error": None, "metrics": None, "confusion": None, "roc": None, "model_name": uri}
    if not uri:
        result["error"] = "Empty URI"
        return result
    try:
        model_obj = load_mlflow_model_cached(uri)
    except Exception as e:
        result["error"] = f"Model load failed: {e}"
        return result
    if df_all is None or "Good_Investment" not in df_all.columns:
        result["error"] = "Dataset or target not available"
        return result

    try:
        X_all = df_all.reindex(columns=MODEL_FEATURES, fill_value=np.nan).copy()
        for col in NUMERIC_FEATURES:
            if col in X_all.columns:
                X_all[col] = pd.to_numeric(X_all[col], errors="coerce").fillna(0)
        if "Public_Transport_Accessibility" in X_all.columns:
            X_all["Public_Transport_Accessibility"] = X_all["Public_Transport_Accessibility"].map(TRANSPORT_MAP).fillna(X_all["Public_Transport_Accessibility"])
        if "Parking_Space" in X_all.columns:
            X_all["Parking_Space"] = X_all["Parking_Space"].map(PARKING_MAP).fillna(X_all["Parking_Space"])

        y_true = df_all["Good_Investment"].astype(int).values

        if hasattr(model_obj, "predict_proba"):
            proba_all = model_obj.predict_proba(X_all)
            if proba_all.ndim == 2 and proba_all.shape[1] >= 2:
                y_proba = proba_all[:, 1]
            else:
                y_proba = proba_all.ravel()
        else:
            preds_all = model_obj.predict(X_all)
            y_proba = np.asarray(preds_all).ravel()
    except Exception as e:
        result["error"] = f"Model inference failed: {e}"
        return result

    if len(y_proba) != len(y_true):
        result["error"] = f"Model returned {len(y_proba)} preds but dataset has {len(y_true)} rows"
        return result

    try:
        y_pred = (y_proba >= 0.5).astype(int)
        acc = float(accuracy_score(y_true, y_pred))
        prec = float(precision_score(y_true, y_pred, zero_division=0))
        rec = float(recall_score(y_true, y_pred, zero_division=0))
        f1 = float(f1_score(y_true, y_pred, zero_division=0))
        try:
            auc_v = float(roc_auc_score(y_true, y_proba))
        except Exception:
            try:
                auc_v = float(roc_auc_score(y_true, y_pred))
            except Exception:
                auc_v = float("nan")
        cm = confusion_matrix(y_true, y_pred)
        roc_pts = None
        try:
            fpr, tpr, thresholds = roc_curve(y_true, y_proba)
            roc_pts = {"fpr": fpr.tolist(), "tpr": tpr.tolist(), "thresholds": thresholds.tolist()}
        except Exception:
            roc_pts = None

        result["metrics"] = {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "auc": auc_v}
        result["confusion"] = cm.tolist()
        result["roc"] = roc_pts
        result["ok"] = True
        return result
    except Exception as e:
        result["error"] = f"Metric computation failed: {e}"
        return result

@st.cache_data(ttl=600)
def compute_regression_metrics(uri: str, rows_sig: int, sum_sig: int) -> Dict[str, Any]:
    result = {"ok": False, "error": None, "metrics": None, "model_name": uri}
    if not uri:
        result["error"] = "Empty URI"
        return result
    try:
        model_obj = load_mlflow_model_cached(uri)
    except Exception as e:
        result["error"] = f"Model load failed: {e}"
        return result
    if df_all is None:
        result["error"] = "Dataset not available"
        return result

    try:
        X_all = df_all.reindex(columns=MODEL_FEATURES, fill_value=np.nan).copy()
        for col in NUMERIC_FEATURES:
            if col in X_all.columns:
                X_all[col] = pd.to_numeric(X_all[col], errors="coerce").fillna(0)
        if "Public_Transport_Accessibility" in X_all.columns:
            X_all["Public_Transport_Accessibility"] = X_all["Public_Transport_Accessibility"].map(TRANSPORT_MAP).fillna(X_all["Public_Transport_Accessibility"])
        if "Parking_Space" in X_all.columns:
            X_all["Parking_Space"] = X_all["Parking_Space"].map(PARKING_MAP).fillna(X_all["Parking_Space"])

        # Determine regression target column if present
        target_candidates = ["Future_Price_5Y", "Price_in_Lakhs", "Target", "y"]
        tgt = None
        for c in target_candidates:
            if c in df_all.columns:
                tgt = c
                break
        if tgt is None:
            result["error"] = "No regression target column found. Expected one of: " + ", ".join(target_candidates)
            return result

        y_true = pd.to_numeric(df_all[tgt], errors="coerce").fillna(0).values
        preds_all = model_obj.predict(X_all)
        y_pred = np.asarray(preds_all).ravel()
    except Exception as e:
        result["error"] = f"Model inference failed: {e}"
        return result

    if len(y_pred) != len(y_true):
        result["error"] = f"Model returned {len(y_pred)} preds but dataset has {len(y_true)} rows"
        return result

    try:
        mse = float(mean_squared_error(y_true, y_pred))
        rmse = float(np.sqrt(mse))
        mae = float(mean_absolute_error(y_true, y_pred))
        r2 = float(r2_score(y_true, y_pred))
        result["metrics"] = {"RMSE": rmse, "MSE": mse, "MAE": mae, "R2": r2, "target_column": tgt}
        result["ok"] = True
        return result
    except Exception as e:
        result["error"] = f"Regression metric calculation failed: {e}"
        return result

# ----- Build comparisons and compute metrics ----- (we compute them early but render them after prediction)
sig_rows, sig_sum = df_signature(df_all)

compare_clf_uris = [o["uri"] for o in clf_options] if clf_options else ([clf_uri] if clf_uri else [])
clf_metrics_list = []
for uri in compare_clf_uris:
    try:
        clf_metrics_list.append({"uri": uri, "res": compute_classification_metrics(uri, sig_rows, sig_sum)})
    except Exception as e:
        clf_metrics_list.append({"uri": uri, "res": {"ok": False, "error": f"Failed to compute: {e}"}})

reg_res = {"ok": False, "error": "No regression URI"}
if reg_uri:
    try:
        reg_res = compute_regression_metrics(reg_uri, sig_rows, sig_sum)
    except Exception as e:
        reg_res = {"ok": False, "error": f"Failed to compute regression metrics: {e}"}

# ----- UI: manual prediction form -----
st.markdown("---")
st.header("Please Fill The Details Below To Get Investment Advice.")
top_cols = st.columns([2,1])

with top_cols[0]:
    st.info("Manual inputs (defaults). Models are chosen in the sidebar.")
    c1, c2, c3 = st.columns(3)

    def opts(col, defaults):
        if df_all is not None and col in df_all.columns:
            vals = df_all[col].dropna().unique().tolist()
            if len(vals) > 200:
                vals = sorted(vals)[:200]
            return [str(v) for v in vals]
        return defaults

    with c1:
        state = st.selectbox("State", opts("State", ["Unknown"]))
        city = st.selectbox("City", opts("City", ["Unknown"]))
        locality = st.selectbox("Locality", opts("Locality", ["Unknown"]))
    with c2:
        property_type = st.selectbox("Property Type", opts("Property_Type", ["Apartment", "Independent House", "Villa"]))
        bhk = st.selectbox("BHK", [0,1,2,3,4,5,6], index=2)
        size = st.number_input("Size_in_SqFt", min_value=10, max_value=20000, value=900)
    with c3:
        price_per_sqft = st.number_input("Price_per_SqFt", min_value=0.0, value=13000.0)
        nearby_schools = st.number_input("Nearby_Schools", min_value=0, max_value=50, value=3)
        nearby_hospitals = st.number_input("Nearby_Hospitals", min_value=0, max_value=20, value=1)

    s1, s2 = st.columns(2)
    with s1:
        transport = st.selectbox("Public_Transport_Accessibility", ["Low","Medium","High"], index=1)
        parking = st.selectbox("Parking_Space", ["No","Yes"], index=1)
    with s2:
        amenities_count = st.number_input("Amenities_Count", min_value=0, max_value=20, value=3)
        age = st.number_input("Age_of_Property", min_value=0, max_value=200, value=5)

    selected_row = pd.DataFrame([{ 
        "State": state, "City": city, "Locality": locality, "Property_Type": property_type,
        "BHK": bhk, "Size_in_SqFt": size, "Price_per_SqFt": price_per_sqft,
        "Nearby_Schools": nearby_schools, "Nearby_Hospitals": nearby_hospitals,
        "Public_Transport_Accessibility": transport, "Parking_Space": parking,
        "Amenities_Count": amenities_count, "Age_of_Property": age,
        "Furnished_Status": "Unfurnished", "Owner_Type": "Owner", "Availability_Status": "Ready_to_Move",
        "Floor_No": 1, "Total_Floors": 5, "Facing": "North"
    }])

# right column intentionally empty of model URIs or status (sidebar holds them)
with top_cols[1]:
    st.header("Model Input")
    st.write(" ")  # placeholder so the column keeps aligned; actual prepared input shows after Run

# Run prediction
run_clicked = st.button("Run prediction")

if run_clicked:
    X = prepare_input(selected_row.copy())

    # ---- Prediction & display: left and right ----
    reg_value = None
    clf_prob_single = None
    clf_raw_vec = None
    if not st.session_state.get("models_loaded", False):
        st.error("Models are not loaded correctly. Check sidebar.")
    else:
        # regression prediction
        try:
            if st.session_state.get("reg_model"):
                reg_raw = st.session_state["reg_model"].predict(X)
                reg_value = unwrap_reg(reg_raw)
            else:
                reg_value = None
        except Exception as e:
            st.error(f"Regression prediction failed: {e}")
            reg_value = None

        # classification prediction
        try:
            if st.session_state.get("clf_model"):
                clf_prob_single, clf_raw_vec = unwrap_clf(st.session_state.get("clf_model"), X)
            else:
                clf_prob_single, clf_raw_vec = None, None
        except Exception as e:
            st.error(f"Classification prediction failed: {e}")
            clf_prob_single, clf_raw_vec = None, None

    # Left: Estimated / Predicted future price (after 5 years) (lakhs)
    left_col, right_col = st.columns([2,1])
    with left_col:
        st.subheader("Estimated / Predicted future price (after 5 years) (lakhs)")
        if reg_value is not None and not np.isnan(reg_value):
            st.markdown(f"### {reg_value:.3f}")
        else:
            st.markdown("### Not available")

        # --- BAR CHART instead of line chart ---
        try:
            current_total_lakhs = (float(price_per_sqft) * float(size)) / 100000.0
        except Exception:
            current_total_lakhs = 0.0

        x_pts = ["Now", "Estimated (5Y)"]
        y_pts = [current_total_lakhs, reg_value if reg_value is not None and not np.isnan(reg_value) else current_total_lakhs]

        try:
            fig_price = go.Figure(data=[go.Bar(x=x_pts, y=y_pts, marker=dict(color=["#1f77b4", "#ff7f0e"]))])
            fig_price.update_layout(
                height=300,
                margin=dict(l=10, r=10, t=20, b=10),
                xaxis_title="",
                yaxis_title="Price (lakhs)",
                template="plotly_white"
            )
            st.plotly_chart(fig_price, use_container_width=True, config={"displayModeBar": False})
        except Exception as e:
            st.warning(f"Price chart failed to render: {e}")

    # Right: Investment decision + simplified YES/NO card
    with right_col:
        st.subheader("Is it good investment? (Yes / No)")
        if clf_prob_single is not None:
            try:
                decision = "Yes" if clf_prob_single >= 0.5 else "No"
                yes_pct = float(clf_prob_single)
                no_pct = max(0.0, 1.0 - yes_pct)
                fig_inv = go.Figure(data=[go.Pie(labels=["Yes", "No"], values=[yes_pct, no_pct], hole=0.6,
                                                marker=dict(colors=["#2ca02c", "#de425a"]))])
                fig_inv.update_traces(textinfo='none', sort=False)
                fig_inv.update_layout(height=260, margin=dict(l=10, r=10, t=10, b=10), showlegend=True)
                fig_inv.add_annotation(text="Yes" if yes_pct >= 0.5 else "No", x=0.5, y=0.5, showarrow=False, font=dict(size=20, color='white'))
                st.plotly_chart(fig_inv, use_container_width=True, config={"displayModeBar": False})
            except Exception as e:
                st.warning(f"Investment pie failed to render: {e}")
                st.markdown(f"### {decision}")
        else:
            st.markdown("### Not available")
            st.write("No classification output")

    # show prepared input table in the right column below decision (keeps alignment)
    with top_cols[1]:
        st.subheader("Provided Details Are:")
        try:
            st.dataframe(X.T)
        except Exception:
            st.write(X.T)

    # Save prediction to session_state
    try:
        st.session_state["last_prediction"] = {"X": X, "reg_value": reg_value, "clf_prob": clf_prob_single, "timestamp": datetime.utcnow().isoformat()}
        st.success("Prediction complete")
    except Exception:
        st.success("Prediction complete (failed to save to session state)")

# ---- Metrics: gauges (render AFTER the prediction block) ----
st.markdown("---")
st.subheader("Model-level metrics")

# --- Classification row (percent gauges) ---
st.markdown("**Classification metrics**")
chosen_clf_uri = None
if clf_metrics_list:
    chosen_clf_uri = (clf_options[clf_sel_idx]["uri"] if clf_options else clf_uri)
    chosen_clf_res = next((m["res"] for m in clf_metrics_list if m["uri"] == chosen_clf_uri), None)
    if chosen_clf_res and chosen_clf_res.get("ok"):
        cm = chosen_clf_res["metrics"]
        gauges = [
            ("Accuracy", _safe(cm.get("accuracy")) * 100),
            ("Precision", _safe(cm.get("precision")) * 100),
            ("Recall", _safe(cm.get("recall")) * 100),
            ("F1", _safe(cm.get("f1")) * 100),
            ("AUC", _safe(cm.get("auc")) * 100)
        ]
        clf_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
        row1 = st.columns(len(gauges))
        for i, (label, val) in enumerate(gauges):
            color = clf_colors[i % len(clf_colors)]
            try:
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=val,
                    number={"valueformat": ".0f", "suffix": "%", "font": {"color": "white"}},
                    gauge={
                        "axis": {"range": [0, 100], "tickfont": {"color": "white"}},
                        "steps": [{"range": [0, 100], "color": "#e6e6e6"}],
                        "bar": {"color": color, "thickness": 0.35}
                    },
                    title={"text": label, "font": {"size": 14, "color": "white"}, "align": "center"}
                ))
                fig.update_layout(height=200, margin=dict(l=6, r=6, t=48, b=6), paper_bgcolor="rgba(0,0,0,0)")
                row1[i].plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
            except Exception as e:
                row1[i].warning(f"Failed to render gauge: {e}")
    else:
        if chosen_clf_res and chosen_clf_res.get("error"):
            st.error(f"Classification metrics unavailable: {chosen_clf_res.get('error')}")
        else:
            st.info("Classification metrics not available for selected model.")
else:
    st.info("No classification model selected.")

# --- Regression row (numeric gauges) ---
st.markdown("---")
st.markdown("**Regression metrics**")
if reg_res and reg_res.get("ok"):
    rm = reg_res["metrics"]
    rmse_v = _safe(rm.get("RMSE"))
    mae_v = _safe(rm.get("MAE"))
    mse_v = _safe(rm.get("MSE"))
    r2_v = _safe(rm.get("R2"))

    max_err = max(rmse_v, mae_v, np.sqrt(mse_v) if mse_v>0 else 1.0, 1.0)
    rng_rmse = max(1.0, max_err * 2)
    rng_mae = rng_rmse
    rng_mse = max(1.0, mse_v * 2)
    rng_r2 = 1.0

    r_gauges = [("RMSE", rmse_v, rng_rmse), ("MAE", mae_v, rng_mae), ("MSE", mse_v, rng_mse), ("R²", r2_v, rng_r2)]
    reg_colors = ["#0b6e4f", "#1f77b4", "#9467bd", "#ff7f0e"]
    row2 = st.columns(len(r_gauges))
    for i, (label, val, rng) in enumerate(r_gauges):
        color = reg_colors[i % len(reg_colors)]
        try:
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=val,
                number={"valueformat": ".3f", "font": {"color": "white"}},
                gauge={
                    "axis": {"range": [0, float(rng)], "tickfont": {"color": "white"}},
                    "steps": [{"range": [0, float(rng)], "color": "#e6e6e6"}],
                    "bar": {"color": color, "thickness": 0.35}
                },
                title={"text": label, "font": {"size": 14, "color": "white"}, "align": "center"}
            ))
            fig.update_layout(height=200, margin=dict(l=6, r=6, t=48, b=6), paper_bgcolor="rgba(0,0,0,0)")
            row2[i].plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        except Exception as e:
            row2[i].warning(f"Failed to render gauge: {e}")
else:
    if reg_res and reg_res.get("error"):
        st.error(f"Regression metrics unavailable: {reg_res.get('error')}")
    else:
        st.info("Regression metrics not available.")

# ---- optional: classifier ROC & confusion after gauges (unchanged) ----
st.markdown("---")
st.subheader("Selected classifier details")
if clf_metrics_list:
    chosen_clf_res = None
    try:
        chosen_clf_res = next((m["res"] for m in clf_metrics_list if m["uri"] == chosen_clf_uri), None)
    except Exception:
        chosen_clf_res = None

    if not chosen_clf_res:
        st.info("Selected classifier metrics not available.")
    elif not chosen_clf_res.get("ok"):
        st.error(f"Metrics unavailable: {chosen_clf_res.get('error')}")
    else:
        mr = chosen_clf_res["metrics"]
        c1, c2 = st.columns([2,3])
        with c1:
            st.metric("Accuracy", f"{mr['accuracy']:.3f}")
            st.metric("Precision", f"{mr['precision']:.3f}")
            st.metric("Recall", f"{mr['recall']:.3f}")
            st.metric("F1", f"{mr['f1']:.3f}")
            st.metric("AUC", f"{mr['auc']:.3f}")
        with c2:
            if chosen_clf_res.get("roc"):
                try:
                    roc = chosen_clf_res["roc"]
                    figroc = go.Figure()
                    figroc.add_trace(go.Scatter(x=roc["fpr"], y=roc["tpr"], mode="lines", name="ROC"))
                    figroc.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines", line=dict(dash="dash"), showlegend=False))
                    figroc.update_layout(title="ROC Curve", xaxis_title="FPR", yaxis_title="TPR", height=300)
                    st.plotly_chart(figroc, use_container_width=True, config={"displayModeBar": False})
                except Exception as e:
                    st.warning(f"Failed to render ROC: {e}")
            else:
                st.info("ROC not available for this model.")
else:
    st.info("No classifier selected.")

st.markdown("---")
st.markdown("Notes: Dashboard shows classification and regression metrics for selected models. Use sidebar to change selection.")
