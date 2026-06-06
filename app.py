"""
Fintech Customer Attrition Engine
──────────────────────────────────
Tab 1 — ML Risk Scorer   : sliders → champion model → Gemini explanation + retention plan
Tab 2 — AI Risk Advisor : free-text chat powered by Gemini with model-in-the-loop
"""

import os
import joblib
import numpy as np
import gradio as gr
from google import genai
from google.genai import types

# ── Model loading ─────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "src", "services", "models", "champion_model.joblib")

try:
    model        = joblib.load(MODEL_PATH)
    MODEL_LOADED = True
except Exception as e:
    model        = None
    MODEL_LOADED = False
    print(f"[WARNING] Model not loaded: {e}")

# ── Gemini client — reads secret injected by system or HF Spaces ──────────────
def get_gemini_client():
    key = os.getenv("GEMINI_API_KEY", "").strip()
    if not key:
        return None
    # Google GenAI client automatically detects GEMINI_API_KEY from environment variables
    return genai.Client()


# ═══════════════════════════════════════════════════════════════════════════════
# Helper — run ML model inference
# ═══════════════════════════════════════════════════════════════════════════════
def run_model(balance_to_max_ratio, tx_velocity_drop_ratio,
              failed_tx_count_7d, support_tickets_30d, is_active_credit_card_user):
    if not MODEL_LOADED or model is None:
        return None, None, None

    features   = np.array([[
        balance_to_max_ratio,
        tx_velocity_drop_ratio,
        failed_tx_count_7d,
        support_tickets_30d,
        is_active_credit_card_user,
    ]])
    proba      = model.predict_proba(features)[0]
    churn_prob = float(proba[1])
    risk       = "HIGH" if churn_prob > 0.65 else "LOW"
    action     = "TRIGGER_RETENTION_OFFER" if risk == "HIGH" else "MONITOR"
    return churn_prob, risk, action


# ═══════════════════════════════════════════════════════════════════════════════
# Helper — build HTML risk card
# ═══════════════════════════════════════════════════════════════════════════════
def risk_card(churn_prob, risk, action):
    colour  = "#e74c3c" if risk == "HIGH" else "#27ae60"
    emoji   = "🔴" if risk == "HIGH" else "🟢"
    bar_pct = int(churn_prob * 100)
    return f"""
    <div style="font-family:sans-serif;padding:16px;border-radius:10px;
                border:2px solid {colour};background:#1a1a2e;color:#eee;">
      <h2 style="margin:0 0 8px;color:{colour};">{emoji} {risk} RISK</h2>
      <p style="margin:0 0 10px;font-size:13px;color:#aaa;">Churn probability</p>
      <div style="background:#333;border-radius:8px;height:20px;width:100%;">
        <div style="background:{colour};width:{bar_pct}%;height:20px;border-radius:8px;"></div>
      </div>
      <p style="margin:10px 0 0;font-size:26px;font-weight:bold;">{churn_prob:.1%}</p>
      <hr style="border-color:#444;margin:10px 0"/>
      <p style="margin:0;font-size:13px;color:#aaa;">
        Action: <strong style="color:{colour};">{action}</strong>
      </p>
    </div>"""


# ═══════════════════════════════════════════════════════════════════════════════
# Tab 1 — Predict + Gemini explanation
# ════════════════════════════════