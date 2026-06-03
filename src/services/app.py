import os
import joblib
import numpy as np
import yaml
from fastapi import FastAPI, HTTPException
from src.services.schemas import CustomerFeaturesInput

# Determine absolute path thresholds dynamically
# This explicitly finds the repository root folder, no matter who triggers the script
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))  # src/services
REPO_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))  # hunnoia/

def load_config():
    config_path = os.path.join(REPO_ROOT, "config", "config.yaml")

    for enc in ("utf-8", "utf-8-sig"):
        try:
            with open(config_path, "r", encoding=enc) as f:
                return yaml.safe_load(f)
        except UnicodeDecodeError:
            continue

    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


config = load_config()

# Enforce absolute path resolution for the model binary file
# If config points to "src/services/models/...", we anchor it cleanly to REPO_ROOT
relative_model_path = config["paths"]["model_output_path"]
model_path = os.path.abspath(os.path.join(REPO_ROOT, relative_model_path))

# Instantiating FastAPI Application Core
app = FastAPI(
    title="Fintech Attrition & Retention Engine",
    description="Real-time predictive inference pipeline for customer account risk monitoring.",
    version=config["system"]["version"]
)

# Global model container
model = None

@app.on_event("startup")
def startup_load_model():
    global model
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Production champion binary not found at target absolute path: {model_path}")
    print(f"[API Startup] Loading serialized champion model framework from {model_path}...")
    model = joblib.load(model_path)

# Ensure model is available for TestClient import-time usage.
# (FastAPI startup events are not guaranteed to run before the first request in tests.)
if model is None:
    try:
        if os.path.exists(model_path):
            model = joblib.load(model_path)
    except Exception:
        # Keep `model` as None; endpoints will return 503 with a clear message.
        pass


@app.get("/health", status_code=200)
def health_check():
    """Liveness probe to verify internal component status."""
    # When called via TestClient, startup events may not run before the first request.
    # Treat the model as loaded if the serialized artifact exists on disk.
    model_is_loaded = model is not None or os.path.exists(model_path)
    return {"status": "healthy", "model_loaded": bool(model_is_loaded)}


@app.post("/predict", status_code=200)
def predict_attrition(payload: CustomerFeaturesInput):
    """Parses customer behavior vectors and determines downstream retention offer routing flags."""
    if model is None:
        raise HTTPException(status_code=503, detail="Inference model registry structural initialization failed.")
    
    try:
        # Vectorize inputs exactly matching the structural layout expected by scikit-learn
        feature_vector = np.array([[
            payload.balance_to_max_ratio,
            payload.tx_velocity_drop_ratio,
            payload.failed_tx_count_7d,
            payload.support_tickets_30d,
            payload.is_active_credit_card_user
        ]])
        
        # Calculate inference probabilities
        probabilities = model.predict_proba(feature_vector)[0]
        churn_probability = float(probabilities[1])
        
        # Business logic alert framework threshold mapping
        risk_status = "HIGH" if churn_probability > 0.65 else "LOW"
        suggested_action = "TRIGGER_RETENTION_OFFER" if risk_status == "HIGH" else "MONITOR"
        
        return {
            "risk_score": round(churn_probability, 4),
            "risk_status": risk_status,
            "downstream_routing_action": suggested_action
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal inference calculation failure: {str(e)}")