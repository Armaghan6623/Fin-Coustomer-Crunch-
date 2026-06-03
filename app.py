"""
Fintech Customer Attrition Engine — Interactive Gradio UI
Loads the champion model directly; no separate backend needed.
"""

import os
import joblib
import numpy as np
import gradio as gr

# ── Model loading ────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "src", "services", "models", "champion_model.joblib")

try:
    model = joblib.load(MODEL_PATH)
    MODEL_LOADED = True
except Exception as e:
    model = None
    MODEL_LOADED = False
    print(f"[WARNING] Could not load model: {e}")


# ── Inference logic ──────────────────────────────────────────────────────────
def predict(
    balance_to_max_ratio: float,
    tx_velocity_drop_ratio: float,
    failed_tx_count_7d: int,
    support_tickets_30d: int,
    is_active_credit_card_user: int,
):
    if not MODEL_LOADED or model is None:
        return (
            "❌ Model not loaded",
            "—",
            "—",
            "<div style='color:red'>Model binary could not be found. Please check deployment.</div>",
        )

    features = np.array([[
        balance_to_max_ratio,
        tx_velocity_drop_ratio,
        failed_tx_count_7d,
        support_tickets_30d,
        is_active_credit_card_user,
    ]])

    proba = model.predict_proba(features)[0]
    churn_prob = float(proba[1])

    risk_status = "🔴 HIGH RISK" if churn_prob > 0.65 else "🟢 LOW RISK"
    action = "TRIGGER RETENTION OFFER" if churn_prob > 0.65 else "MONITOR"
    score_display = f"{churn_prob:.1%}"

    # Colour-coded summary card
    colour = "#e74c3c" if churn_prob > 0.65 else "#27ae60"
    bar_pct = int(churn_prob * 100)
    html = f"""
    <div style="font-family:sans-serif; padding:16px; border-radius:10px;
                border: 2px solid {colour}; background:#1a1a2e; color:#eee;">
      <h2 style="margin:0 0 8px; color:{colour};">{risk_status}</h2>
      <p style="margin:0 0 12px; font-size:14px;">Churn probability score</p>
      <div style="background:#333; border-radius:8px; height:22px; width:100%;">
        <div style="background:{colour}; width:{bar_pct}%; height:22px;
                    border-radius:8px; transition:width 0.4s;"></div>
      </div>
      <p style="margin:10px 0 0; font-size:22px; font-weight:bold;">{score_display}</p>
      <hr style="border-color:#444; margin:12px 0"/>
      <p style="margin:0; font-size:13px; color:#aaa;">
        Recommended action: <strong style="color:{colour};">{action}</strong>
      </p>
    </div>
    """
    return score_display, risk_status, action, html


# ── Gradio interface ─────────────────────────────────────────────────────────
with gr.Blocks(
    theme=gr.themes.Base(
        primary_hue="blue",
        neutral_hue="slate",
    ),
    title="Fintech Attrition Engine",
    css="""
    #title { text-align: center; margin-bottom: 4px; }
    #subtitle { text-align: center; color: #888; margin-bottom: 20px; font-size: 14px; }
    """,
) as demo:

    gr.Markdown("# 🏦 Fintech Customer Attrition Engine", elem_id="title")
    gr.Markdown(
        "Real-time churn risk prediction for customer retention teams.",
        elem_id="subtitle",
    )

    with gr.Row():
        # ── Left column: inputs ──────────────────────────────────────────────
        with gr.Column(scale=1):
            gr.Markdown("### 📋 Customer Behaviour Signals")

            balance_slider = gr.Slider(
                minimum=0.0,
                maximum=1.5,
                step=0.01,
                value=0.45,
                label="Balance-to-Max Ratio",
                info="Current balance ÷ historical peak balance (0 = empty, 1 = at peak)",
            )
            velocity_slider = gr.Slider(
                minimum=0.0,
                maximum=1.0,
                step=0.01,
                value=0.3,
                label="Transaction Velocity Drop Ratio",
                info="How much transaction frequency has dropped (0 = none, 1 = stopped entirely)",
            )
            failed_tx = gr.Slider(
                minimum=0,
                maximum=30,
                step=1,
                value=1,
                label="Failed Transactions (last 7 days)",
                info="Number of declined or failed payment attempts",
            )
            support_tickets = gr.Slider(
                minimum=0,
                maximum=20,
                step=1,
                value=0,
                label="Support Tickets (last 30 days)",
                info="Count of customer support disputes filed",
            )
            active_cc = gr.Radio(
                choices=[(("Yes", 1)), (("No", 0))],
                value=1,
                label="Active Credit Card User",
                info="Is the customer actively using their credit card?",
            )

            with gr.Row():
                clear_btn = gr.Button("🔄 Reset", variant="secondary")
                predict_btn = gr.Button("🔍 Analyse Risk", variant="primary")

        # ── Right column: outputs ────────────────────────────────────────────
        with gr.Column(scale=1):
            gr.Markdown("### 📊 Risk Assessment")

            result_html = gr.HTML(
                value="<div style='color:#888; font-family:sans-serif; padding:20px;'>"
                      "Fill in the customer signals and click <strong>Analyse Risk</strong>.</div>"
            )

            with gr.Row():
                score_box = gr.Textbox(label="Churn Score", interactive=False)
                status_box = gr.Textbox(label="Risk Status", interactive=False)
                action_box = gr.Textbox(label="Recommended Action", interactive=False)

    # ── Example presets ──────────────────────────────────────────────────────
    gr.Markdown("### 💡 Example Presets")
    gr.Examples(
        examples=[
            [0.15, 0.82, 5, 4, 0],   # High risk
            [0.90, 0.05, 0, 0, 1],   # Low risk
            [0.50, 0.50, 2, 1, 1],   # Borderline
        ],
        inputs=[balance_slider, velocity_slider, failed_tx, support_tickets, active_cc],
        label="Click a row to populate the form",
    )

    # ── Wire up events ───────────────────────────────────────────────────────
    inputs = [balance_slider, velocity_slider, failed_tx, support_tickets, active_cc]
    outputs = [score_box, status_box, action_box, result_html]

    predict_btn.click(fn=predict, inputs=inputs, outputs=outputs)
    clear_btn.click(
        fn=lambda: (0.45, 0.3, 1, 0, 1, "", "", "",
                    "<div style='color:#888; font-family:sans-serif; padding:20px;'>"
                    "Fill in the customer signals and click <strong>Analyse Risk</strong>.</div>"),
        inputs=[],
        outputs=[balance_slider, velocity_slider, failed_tx, support_tickets,
                 active_cc, score_box, status_box, action_box, result_html],
    )


if __name__ == "__main__":
    demo.launch()
